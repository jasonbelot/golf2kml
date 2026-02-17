#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Golf2KML
Segmentation automatique d'une image de trou de golf (fairway / rough / forêt / green / départs)
+ OB = ligne extérieure (contour) de l'union du rough
+ export en KML.

Dépendances (serveur/headless):
  opencv-python-headless
  numpy
  scikit-learn
  shapely
  simplekml

Usage:
  python golf2kml.py --image trou.png --out trou.kml
  python golf2kml.py --image trou.png --out trou_geo.kml --georef "lat,lon;lat,lon;lat,lon;lat,lon"

Notes:
- Sans georef: export en coordonnées locales (mètres) dans un repère arbitraire.
- Avec georef (4 coins TL;TR;BR;BL), export en WGS84 utilisable dans Google Earth.
"""

import argparse
import sys
from dataclasses import dataclass

import numpy as np

# --- OpenCV import (headless-friendly)
try:
    import cv2
except Exception as e:
    raise ImportError(
        "Impossible d'importer OpenCV (cv2).\n"
        "Sur Streamlit Cloud / serveur, utilise 'opencv-python-headless' dans requirements.txt\n"
        "et force Python 3.11 via runtime.txt.\n"
        f"Détail: {e}"
    )

from sklearn.cluster import KMeans
import simplekml

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# Compat Shapely 2.x / 1.8
try:
    from shapely.validation import make_valid as _make_valid
except Exception:
    _make_valid = None


def make_valid(poly):
    if poly.is_valid:
        return poly
    if _make_valid is not None:
        try:
            return _make_valid(poly)
        except Exception:
            return poly
    # fallback shapely<2: buffer(0) corrige souvent
    try:
        return poly.buffer(0)
    except Exception:
        return poly


# -----------------------------
# Config
# -----------------------------

@dataclass
class Params:
    k: int = 7
    known_line_m: float = 10.0
    sample_pixels: int = 200_000
    min_area_px: int = 6000
    simplify_px: float = 3.0
    debug_prefix: str | None = None
    max_side: int = 1600  # downscale (stabilité + vitesse). Met 0 pour désactiver.


# -----------------------------
# Scale line detection (yellow)
# -----------------------------

def detect_yellow_scale_line_mpp(bgr: np.ndarray, known_length_m: float) -> tuple[float, np.ndarray, tuple]:
    """
    Détecte la plus grande composante "jaune" et estime sa longueur en pixels.
    Retourne (meters_per_pixel, yellow_mask, rect).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Jaune (à ajuster si besoin)
    lower = np.array([15, 80, 120], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Nettoyage
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(
            "Trait jaune non détecté. "
            "Vérifie que le trait est bien jaune (HSV) ou ajuste les seuils dans detect_yellow_scale_line_mpp()."
        )

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)  # center, (w,h), angle
    (_, _), (w, h), _ = rect
    length_px = float(max(w, h))

    if length_px < 5:
        raise RuntimeError("Trait jaune détecté mais trop court (bruit).")

    meters_per_pixel = known_length_m / length_px
    return meters_per_pixel, mask, rect


# -----------------------------
# Features + segmentation
# -----------------------------

def compute_features(bgr: np.ndarray) -> np.ndarray:
    """
    Features par pixel : HSV + texture (variance locale sur V).
    Output: (H,W,4) float32 normalisé.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Texture: variance locale sur V
    v_u8 = v.astype(np.uint8)
    v_blur = cv2.GaussianBlur(v_u8, (9, 9), 0)
    v_sq = cv2.GaussianBlur((v_u8.astype(np.float32) ** 2), (9, 9), 0)
    var = np.maximum(v_sq - (v_blur.astype(np.float32) ** 2), 0.0)

    feat = np.dstack([
        h / 180.0,
        s / 255.0,
        v / 255.0,
        np.clip(var / (255.0 ** 2), 0, 1),
    ]).astype(np.float32)

    return feat


def kmeans_segment(feat: np.ndarray, k: int, sample_pixels: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    KMeans sur échantillon, puis prédiction sur tous pixels.
    Return labels (H,W) and centers (k,4)
    """
    H, W, C = feat.shape
    X = feat.reshape(-1, C)
    n = X.shape[0]

    rng = np.random.default_rng(seed)
    if n > sample_pixels:
        idx = rng.choice(n, size=sample_pixels, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    km.fit(Xs)

    labels = km.predict(X).reshape(H, W)
    centers = km.cluster_centers_
    return labels, centers


# -----------------------------
# Cluster -> classes (heuristics)
# -----------------------------

def assign_clusters_to_classes(centers: np.ndarray) -> dict[int, str]:
    """
    Heuristiques basées sur (H,S,V,texture):
    - foret: sombre + texture
    - green: lumineux + très peu texturé
    - departs: très uniforme + lumineux (2e meilleur)
    - fairway: plutôt uniforme + lumineux
    - rough: plus texturé que fairway
    """
    k = centers.shape[0]
    h = centers[:, 0]
    s = centers[:, 1]
    v = centers[:, 2]
    t = centers[:, 3]

    mapping: dict[int, str] = {}

    # Forêt: sombre et/ou texturée
    forest_idx = int(np.argmin(v + 0.25 * (1 - t)))
    mapping[forest_idx] = "foret"

    remaining = [i for i in range(k) if i != forest_idx]

    # Green: lumineux + uniforme
    green_idx = max(remaining, key=lambda i: (v[i] - 1.0 * t[i] + 0.15 * s[i]))
    mapping[green_idx] = "green"
    remaining.remove(green_idx)

    # Départs: très uniforme et lumineux (second meilleur score "uniforme & lumineux")
    if remaining:
        tee_idx = max(remaining, key=lambda i: (v[i] - 1.3 * t[i] + 0.05 * s[i]))
        mapping[tee_idx] = "departs"
        remaining.remove(tee_idx)

    # Fairway: lumineux + plutôt uniforme
    if remaining:
        fairway_idx = max(remaining, key=lambda i: (v[i] - 0.6 * t[i]))
        mapping[fairway_idx] = "fairway"
        remaining.remove(fairway_idx)

    # Rough: plus texturé
    for i in remaining:
        mapping[i] = "rough"

    return mapping


# -----------------------------
# Masks -> polygons
# -----------------------------

def cleanup_mask(mask_bool: np.ndarray, open_ksize: int = 5, close_ksize: int = 11) -> np.ndarray:
    m = (mask_bool.astype(np.uint8) * 255)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2, iterations=2)
    return (m > 0)


def mask_to_contours(mask_bool: np.ndarray) -> list[np.ndarray]:
    m = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def contours_to_polygons(contours: list[np.ndarray], min_area_px: int) -> list[Polygon]:
    polys: list[Polygon] = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        pts = cnt.reshape(-1, 2)
        poly = Polygon(pts)
        poly = make_valid(poly)
        if poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            polys.extend([p for p in poly.geoms if (not p.is_empty)])
        else:
            polys.append(poly)
    return polys


def simplify_geom(g, tol_px: float):
    try:
        sg = g.simplify(tol_px, preserve_topology=True)
        sg = make_valid(sg)
        return sg
    except Exception:
        return g


# -----------------------------
# Georef mapping
# -----------------------------

def parse_georef(s: str):
    parts = s.strip().split(";")
    if len(parts) != 4:
        raise ValueError('georef doit être "lat,lon;lat,lon;lat,lon;lat,lon" (TL;TR;BR;BL)')
    pts = []
    for p in parts:
        lat_str, lon_str = p.split(",")
        pts.append((float(lat_str), float(lon_str)))
    return pts  # [(lat,lon)*4]


def make_px_to_coord_fn(H: int, W: int, meters_per_pixel: float, georef=None):
    if georef is None:
        # repère local en mètres
        def fn(x_px, y_px):
            x_m = float(x_px) * meters_per_pixel
            y_m = float(y_px) * meters_per_pixel
            return (x_m, -y_m)  # lon=x, lat=-y pour "nord vers le haut"
        return fn

    (lat_tl, lon_tl), (lat_tr, lon_tr), (lat_br, lon_br), (lat_bl, lon_bl) = georef

    def fn(x_px, y_px):
        u = float(x_px) / float(max(1, W - 1))
        v = float(y_px) / float(max(1, H - 1))

        lat_top = lat_tl + u * (lat_tr - lat_tl)
        lon_top = lon_tl + u * (lon_tr - lon_tl)
        lat_bot = lat_bl + u * (lat_br - lat_bl)
        lon_bot = lon_bl + u * (lon_br - lon_bl)

        lat = lat_top + v * (lat_bot - lat_top)
        lon = lon_top + v * (lon_bot - lon_top)
        return (lon, lat)
    return fn


# -----------------------------
# KML export
# -----------------------------

def polygon_to_kml_coords(poly: Polygon, px_to_coord_fn):
    ext = [(float(x), float(y)) for x, y in np.array(poly.exterior.coords)]
    ext_ll = [px_to_coord_fn(x, y) for x, y in ext]

    holes_ll = []
    for ring in poly.interiors:
        ring_pts = [(float(x), float(y)) for x, y in np.array(ring.coords)]
        holes_ll.append([px_to_coord_fn(x, y) for x, y in ring_pts])

    return ext_ll, holes_ll


def add_polygons_folder(kml: simplekml.Kml, name: str, geoms, px_to_coord_fn, line_color: str, fill_color: str):
    folder = kml.newfolder(name=name)
    for g in geoms:
        if g.is_empty:
            continue
        if isinstance(g, MultiPolygon):
            geom_list = list(g.geoms)
        else:
            geom_list = [g]
        for poly in geom_list:
            if poly.is_empty:
                continue
            ext_ll, holes_ll = polygon_to_kml_coords(poly, px_to_coord_fn)
            p = folder.newpolygon(
                outerboundaryis=ext_ll,
                innerboundaryis=holes_ll if holes_ll else None
            )
            p.style.linestyle.color = line_color
            p.style.linestyle.width = 2
            p.style.polystyle.color = fill_color
    return folder


# -----------------------------
# Downscale helper
# -----------------------------

def maybe_downscale(bgr: np.ndarray, max_side: int):
    if not max_side or max_side <= 0:
        return bgr, 1.0
    H, W = bgr.shape[:2]
    m = max(H, W)
    if m <= max_side:
        return bgr, 1.0
    scale = max_side / float(m)
    newW = int(round(W * scale))
    newH = int(round(H * scale))
    resized = cv2.resize(bgr, (newW, newH), interpolation=cv2.INTER_AREA)
    return resized, scale


# -----------------------------
# Main pipeline
# -----------------------------

def run(image_path: str, out_kml: str, georef_str: str | None, p: Params):
    bgr0 = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr0 is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {image_path}")

    # downscale
    bgr, scale = maybe_downscale(bgr0, p.max_side)

    H, W = bgr.shape[:2]

    # scale line detection
    mpp, yellow_mask, rect = detect_yellow_scale_line_mpp(bgr, p.known_line_m)

    # si downscale appliqué, mpp doit être ajusté (car px plus grands)
    # (si l'échelle est mesurée sur l'image downscalée, c'est déjà cohérent)
    meters_per_pixel = mpp

    # features + segmentation
    feat = compute_features(bgr)
    labels, centers = kmeans_segment(feat, p.k, p.sample_pixels, seed=0)

    # mapping
    cluster_to_class = assign_clusters_to_classes(centers)

    classes = ["fairway", "rough", "foret", "green", "departs"]
    masks = {c: np.zeros((H, W), dtype=bool) for c in classes}

    for cid, cname in cluster_to_class.items():
        if cname in masks:
            masks[cname] |= (labels == cid)

    # clean masks
    for c in classes:
        masks[c] = cleanup_mask(masks[c])

    # masks -> polygons
    class_geoms = {}
    for c in classes:
        contours = mask_to_contours(masks[c])
        polys = contours_to_polygons(contours, p.min_area_px)
        polys = [simplify_geom(poly, p.simplify_px) for poly in polys]
        class_geoms[c] = polys

    # OB = extérieur union rough
    rough_union = unary_union(class_geoms["rough"]) if class_geoms["rough"] else None
    rough_union = make_valid(rough_union) if rough_union else None

    # georef mapping
    georef = parse_georef(georef_str) if georef_str else None
    px_to_coord_fn = make_px_to_coord_fn(H, W, meters_per_pixel, georef=georef)

    # KML
    kml = simplekml.Kml()

    # Styles KML: aabbggrr
    style = {
        "fairway": ("ff006600", "6600ff00"),
        "rough":   ("ff003300", "6600aa00"),
        "foret":   ("ff002200", "66005500"),
        "green":   ("ff00aa00", "6600ff66"),
        "departs": ("ff00cccc", "6600ffff"),
        "ob":      ("ff0000ff", None),
    }

    for c in classes:
        line, fill = style[c]
        add_polygons_folder(kml, c, class_geoms[c], px_to_coord_fn, line, fill)

    if rough_union and (not rough_union.is_empty):
        folder_ob = kml.newfolder(name="OB")
        geoms = list(rough_union.geoms) if isinstance(rough_union, MultiPolygon) else [rough_union]
        for g in geoms:
            if g.is_empty:
                continue
            ext = [(float(x), float(y)) for x, y in np.array(g.exterior.coords)]
            ext_ll = [px_to_coord_fn(x, y) for x, y in ext]
            ls = folder_ob.newlinestring(coords=ext_ll)
            ls.style.linestyle.color = style["ob"][0]
            ls.style.linestyle.width = 3

    kml.save(out_kml)

    # Debug exports
    if p.debug_prefix:
        vis = (labels.astype(np.float32) / max(1, p.k - 1) * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        cv2.imwrite(p.debug_prefix + "_labels.png", vis)
        cv2.imwrite(p.debug_prefix + "_yellowmask.png", (yellow_mask > 0).astype(np.uint8) * 255)

        dbg = bgr.copy()
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(dbg, [box], -1, (0, 0, 255), 2)
        cv2.imwrite(p.debug_prefix + "_scale_detect.png", dbg)

        for c in classes:
            cv2.imwrite(p.debug_prefix + f"_{c}.png", masks[c].astype(np.uint8) * 255)

    print(f"[OK] KML écrit: {out_kml}")
    print(f"[INFO] Échelle estimée: {meters_per_pixel:.6f} m/px")
    if georef is None:
        print("[INFO] KML non géoréférencé (coordonnées locales). Fournis --georef pour Google Earth.")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Chemin image (png/jpg)")
    ap.add_argument("--out", required=True, help="Chemin sortie .kml")
    ap.add_argument("--k", type=int, default=7, help="Nb clusters KMeans (6-10 souvent bien)")
    ap.add_argument("--known_line_m", type=float, default=10.0, help="Longueur (m) du trait jaune")
    ap.add_argument("--min_area_px", type=int, default=6000, help="Aire min en px² pour conserver une zone")
    ap.add_argument("--simplify_px", type=float, default=3.0, help="Tolérance simplification polygones (px)")
    ap.add_argument("--sample_pixels", type=int, default=200000, help="Nb pixels échantillonnés pour KMeans")
    ap.add_argument("--max_side", type=int, default=1600, help="Downscale si image trop grande (0 = off)")
    ap.add_argument("--georef", type=str, default=None, help='TL;TR;BR;BL: "lat,lon;lat,lon;lat,lon;lat,lon"')
    ap.add_argument("--debug_prefix", type=str, default=None, help="Prefix de fichiers debug (ex: debug/hole)")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    p = Params(
        k=args.k,
        known_line_m=args.known_line_m,
        sample_pixels=args.sample_pixels,
        min_area_px=args.min_area_px,
        simplify_px=args.simplify_px,
        debug_prefix=args.debug_prefix,
        max_side=args.max_side,
    )

    run(args.image, args.out, args.georef, p)


if __name__ == "__main__":
    main()

