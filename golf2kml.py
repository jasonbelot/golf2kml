#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Segmentation automatique d'un trou de golf (fairway / rough / forêt / green / départs)
+ OB = contour extérieur de l'union du rough
+ export en KML.

Dépendances:
  pip install opencv-python numpy simplekml shapely scikit-learn

Usage:
  python golf2kml.py --image /path/image.png --out hole.kml

Option géoréférencement (pour superposition Google Earth):
  --georef "lat_tl,lon_tl;lat_tr,lon_tr;lat_br,lon_br;lat_bl,lon_bl"
Ex:
  --georef "43.123,1.234;43.123,1.245;43.115,1.245;43.115,1.234"

Note:
- Sans --georef, le KML est en coordonnées locales (x,y) ~ (lon,lat) arbitraires.
  Il restera "valide" mais pas positionné sur la Terre.
"""

import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
import simplekml
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid


# -----------------------------
# Utilitaires géométrie / KML
# -----------------------------

def contours_to_polygons(contours, min_area_px=5000):
    """Convertit des contours OpenCV en polygones shapely, en filtrant les petites zones."""
    polys = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        pts = cnt.reshape(-1, 2)
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue
        # shapely peut renvoyer MultiPolygon si "make_valid" split
        if isinstance(poly, MultiPolygon):
            polys.extend([p for p in poly.geoms if not p.is_empty])
        else:
            polys.append(poly)
    return polys


def simplify_polygon(poly, tol_px=2.0):
    """Simplifie un polygone (tolérance en pixels)."""
    try:
        sp = poly.simplify(tol_px, preserve_topology=True)
        if not sp.is_valid:
            sp = make_valid(sp)
        return sp
    except Exception:
        return poly


def polygon_to_kml_coords(poly, px_to_coord_fn):
    """
    Convertit un polygone shapely (en pixels) en liste de coordonnées KML (lon,lat).
    px_to_coord_fn: (x_px, y_px) -> (lon, lat)
    """
    ext = [(float(x), float(y)) for x, y in np.array(poly.exterior.coords)]
    ext_ll = [px_to_coord_fn(x, y) for x, y in ext]

    holes_ll = []
    for ring in poly.interiors:
        ring_pts = [(float(x), float(y)) for x, y in np.array(ring.coords)]
        holes_ll.append([px_to_coord_fn(x, y) for x, y in ring_pts])

    return ext_ll, holes_ll


def add_polygons_folder(kml, name, polygons, px_to_coord_fn, line_color="ff000000", fill_color="7f00ff00"):
    folder = kml.newfolder(name=name)
    for poly in polygons:
        if poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            geoms = list(poly.geoms)
        else:
            geoms = [poly]

        for g in geoms:
            if g.is_empty:
                continue
            ext_ll, holes_ll = polygon_to_kml_coords(g, px_to_coord_fn)
            p = folder.newpolygon(outerboundaryis=ext_ll, innerboundaryis=holes_ll if holes_ll else None)
            p.style.linestyle.color = line_color
            p.style.linestyle.width = 2
            p.style.polystyle.color = fill_color
    return folder


# -----------------------------
# Détection du trait jaune (échelle)
# -----------------------------

def detect_yellow_scale_line_mpp(bgr, known_length_m=10.0):
    """
    Détecte la plus grande composante jaune et estime sa longueur en pixels.
    Retourne meters_per_pixel.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Seuils jaune (peuvent être ajustés si besoin)
    lower = np.array([15, 80, 120], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Nettoyage
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Trait jaune non détecté. Ajuste les seuils HSV ou fournis l'échelle manuellement.")

    cnt = max(contours, key=cv2.contourArea)

    # Longueur via boîte englobante orientée (minAreaRect)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    length_px = max(w, h)
    if length_px < 5:
        raise RuntimeError("Trait jaune détecté mais trop court / bruité.")

    meters_per_pixel = known_length_m / float(length_px)
    return meters_per_pixel, mask, rect


# -----------------------------
# Features + Clustering
# -----------------------------

def compute_features(bgr):
    """
    Features par pixel : HSV + texture (variance locale sur V).
    Retour: feature_map (H,W,C)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Texture : variance locale sur V (fenêtre 9x9)
    v_u8 = v.astype(np.uint8)
    v_blur = cv2.GaussianBlur(v_u8, (9, 9), 0)
    v_sq = cv2.GaussianBlur((v_u8.astype(np.float32) ** 2), (9, 9), 0)
    var = np.maximum(v_sq - (v_blur.astype(np.float32) ** 2), 0.0)

    # Normalisation douce
    h_n = h / 180.0
    s_n = s / 255.0
    v_n = v / 255.0
    var_n = np.clip(var / (255.0 ** 2), 0, 1)

    feat = np.dstack([h_n, s_n, v_n, var_n]).astype(np.float32)
    return feat


def kmeans_segment(feat, k=6, sample=200000, seed=0):
    """
    KMeans sur un échantillon de pixels, puis prédiction sur tous les pixels.
    Retour: labels (H,W), centers (k,C)
    """
    H, W, C = feat.shape
    X = feat.reshape(-1, C)

    n = X.shape[0]
    if n > sample:
        idx = np.random.default_rng(seed).choice(n, size=sample, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    km.fit(Xs)

    labels = km.predict(X).reshape(H, W)
    centers = km.cluster_centers_
    return labels, centers


# -----------------------------
# Attribution clusters -> classes
# -----------------------------

def assign_clusters_to_classes(centers):
    """
    Heuristiques basées sur (H,S,V,texture):
    - forêt: v bas + texture plus élevée
    - green: v haut, s moyen/haut, texture faible (surface uniforme)
    - fairway: v moyen/haut, s moyen, texture faible/moyenne
    - rough: v moyen, texture plus élevée que fairway
    - départs: proche green/fairway mais souvent très uniforme; ici: texture très faible + v haut
    Retour: dict cluster_id -> class_name
    """
    # centers: [k,4] => h,s,v,var
    k = centers.shape[0]
    h = centers[:, 0]
    s = centers[:, 1]
    v = centers[:, 2]
    t = centers[:, 3]

    mapping = {}

    # Forêt: les plus sombres (v faible) et/ou les plus "texturés"
    forest_idx = int(np.argmin(v + 0.3 * (1 - t)))  # favorise sombre
    mapping[forest_idx] = "foret"

    remaining = [i for i in range(k) if i != forest_idx]

    # Green: lumineux et très peu texturé
    green_idx = max(remaining, key=lambda i: (v[i] - 0.8 * t[i] + 0.2 * s[i]))
    mapping[green_idx] = "green"
    remaining.remove(green_idx)

    # Départs: très uniformes, souvent lumineux aussi (second meilleur "uniforme & lumineux")
    if remaining:
        tee_idx = max(remaining, key=lambda i: (v[i] - 1.2 * t[i] + 0.1 * s[i]))
        mapping[tee_idx] = "departs"
        remaining.remove(tee_idx)

    # Fairway vs Rough:
    # fairway: plus uniforme (t plus faible) + plutôt lumineux
    # rough: plus texturé (t plus fort) + un peu moins lumineux
    if len(remaining) >= 2:
        fairway_idx = max(remaining, key=lambda i: (v[i] - 0.6 * t[i]))
        mapping[fairway_idx] = "fairway"
        remaining.remove(fairway_idx)

        rough_idx = max(remaining, key=lambda i: (t[i] + 0.2 * (1 - v[i])))
        mapping[rough_idx] = "rough"
        remaining.remove(rough_idx)

    # Tout le reste -> rough par défaut (ou "autre")
    for i in remaining:
        mapping[i] = "rough"

    return mapping


# -----------------------------
# Post-traitement masques
# -----------------------------

def cleanup_mask(mask, close_ksize=9, open_ksize=5):
    mask_u8 = (mask.astype(np.uint8) * 255)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k1, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k2, iterations=2)
    return (mask_u8 > 0)


def mask_to_contours(mask_bool):
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# -----------------------------
# Géoréférencement (optionnel)
# -----------------------------

def make_px_to_coord_fn(H, W, meters_per_pixel, georef=None):
    """
    Si georef=None : renvoie (lon=x_m, lat=y_m) dans un repère local (mètres).
    Si georef fourni (4 coins lat/lon): TL,TR,BR,BL -> mapping bilinéaire vers WGS84.
    """
    if georef is None:
        # repère local: origine en haut-gauche, x vers la droite, y vers le bas (converti en lat "y")
        def fn(x_px, y_px):
            x_m = x_px * meters_per_pixel
            y_m = y_px * meters_per_pixel
            # KML attend lon,lat -> on met x_m en "lon", y_m en "lat" (repère arbitraire)
            return (x_m, -y_m)  # -y pour que "nord" soit vers le haut
        return fn

    # georef: [(lat,lon) TL, TR, BR, BL]
    (lat_tl, lon_tl), (lat_tr, lon_tr), (lat_br, lon_br), (lat_bl, lon_bl) = georef

    def fn(x_px, y_px):
        u = x_px / float(W - 1)
        v = y_px / float(H - 1)
        # bilinéaire sur lat/lon
        lat_top = lat_tl + u * (lat_tr - lat_tl)
        lon_top = lon_tl + u * (lon_tr - lon_tl)
        lat_bot = lat_bl + u * (lat_br - lat_bl)
        lon_bot = lon_bl + u * (lon_br - lon_bl)

        lat = lat_top + v * (lat_bot - lat_top)
        lon = lon_top + v * (lon_bot - lon_top)
        return (lon, lat)

    return fn


def parse_georef(s):
    # "lat,lon;lat,lon;lat,lon;lat,lon"
    parts = s.strip().split(";")
    if len(parts) != 4:
        raise ValueError("georef doit avoir 4 points: TL;TR;BR;BL au format lat,lon")
    pts = []
    for p in parts:
        lat_str, lon_str = p.split(",")
        pts.append((float(lat_str), float(lon_str)))
    return pts


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Chemin image (png/jpg)")
    ap.add_argument("--out", required=True, help="Chemin sortie .kml")
    ap.add_argument("--k", type=int, default=6, help="Nb clusters KMeans (defaut 6)")
    ap.add_argument("--known_line_m", type=float, default=10.0, help="Longueur (m) du trait jaune (defaut 10)")
    ap.add_argument("--min_area_px", type=float, default=6000, help="Aire min en px² pour conserver un polygone")
    ap.add_argument("--simplify_px", type=float, default=3.0, help="Tolérance simplification polygones (px)")
    ap.add_argument("--georef", type=str, default=None, help='TL;TR;BR;BL en "lat,lon;lat,lon;lat,lon;lat,lon"')
    ap.add_argument("--debug_prefix", type=str, default=None, help="Si défini, exporte des PNG de debug avec ce prefix")
    args = ap.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {args.image}")

    H, W = bgr.shape[:2]

    # 1) échelle via trait jaune
    meters_per_pixel, yellow_mask, rect = detect_yellow_scale_line_mpp(bgr, known_length_m=args.known_line_m)

    # 2) features + segmentation
    feat = compute_features(bgr)
    labels, centers = kmeans_segment(feat, k=args.k)

    # 3) mapping clusters -> classes
    cluster_to_class = assign_clusters_to_classes(centers)

    # 4) masques par classe
    classes = ["fairway", "rough", "foret", "green", "departs"]
    masks = {c: np.zeros((H, W), dtype=bool) for c in classes}

    for cid, cname in cluster_to_class.items():
        if cname in masks:
            masks[cname] |= (labels == cid)

    # nettoyage morpho
    for c in classes:
        masks[c] = cleanup_mask(masks[c])

    # 5) contours -> polygons shapely (en pixels)
    class_polys = {}
    for c in classes:
        contours = mask_to_contours(masks[c])
        polys = contours_to_polygons(contours, min_area_px=args.min_area_px)
        polys = [simplify_polygon(p, tol_px=args.simplify_px) for p in polys]
        class_polys[c] = polys

    # 6) OB = contour extérieur de l'union du rough
    rough_union = unary_union(class_polys["rough"]) if class_polys["rough"] else None
    ob_poly = None
    if rough_union and not rough_union.is_empty:
        if isinstance(rough_union, MultiPolygon):
            # OB: contour extérieur global -> enveloppe de l'union (convexe ou concave?)
            # Ici: on prend boundary de l'union, puis on garde l'exterior du polygone englobant.
            rough_union = unary_union([g for g in rough_union.geoms])
        # "OB" demandé = ligne extérieure des zones de rough -> extérieur de l'union
        # On stocke le polygone rough_union lui-même et on met un style "ligne" dans le KML
        ob_poly = rough_union

    # 7) coords mapping pour KML
    georef = parse_georef(args.georef) if args.georef else None
    px_to_coord_fn = make_px_to_coord_fn(H, W, meters_per_pixel, georef=georef)

    # 8) KML export
    kml = simplekml.Kml()

    # Styles (KML: aabbggrr)
    # (tu peux modifier)
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
        add_polygons_folder(
            kml, c, class_polys[c], px_to_coord_fn,
            line_color=line,
            fill_color=fill
        )

    # OB en ligne (pas rempli)
    if ob_poly and not ob_poly.is_empty:
        folder_ob = kml.newfolder(name="OB")
        # OB peut être MultiPolygon
        geoms = list(ob_poly.geoms) if isinstance(ob_poly, MultiPolygon) else [ob_poly]
        for g in geoms:
            if g.is_empty:
                continue
            # On trace uniquement l'extérieur
            ext = [(float(x), float(y)) for x, y in np.array(g.exterior.coords)]
            ext_ll = [px_to_coord_fn(x, y) for x, y in ext]
            ls = folder_ob.newlinestring(coords=ext_ll)
            ls.style.linestyle.color = style["ob"][0]
            ls.style.linestyle.width = 3

    kml.save(args.out)

    # Debug exports
    if args.debug_prefix:
        # image labels colorisée simple
        vis = (labels.astype(np.float32) / max(1, args.k - 1) * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        cv2.imwrite(args.debug_prefix + "_labels.png", vis)
        cv2.imwrite(args.debug_prefix + "_yellowmask.png", (yellow_mask > 0).astype(np.uint8) * 255)

        dbg = bgr.copy()
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(dbg, [box], -1, (0, 0, 255), 2)
        cv2.imwrite(args.debug_prefix + "_scale_detect.png", dbg)

        for c in classes:
            cv2.imwrite(args.debug_prefix + f"_{c}.png", masks[c].astype(np.uint8) * 255)

    print(f"[OK] KML écrit: {args.out}")
    print(f"[INFO] Échelle estimée: {meters_per_pixel:.4f} m/px")
    if georef is None:
        print("[INFO] KML en coordonnées locales (non géoréférencé). Fournis --georef pour superposition Google Earth.")


if __name__ == "__main__":
    main()

