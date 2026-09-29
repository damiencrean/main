"""Clean, finished render of the Ducati suit on all three mannequins (front, back, side).

Every panel is redrawn as a smooth vector shape (supersampled for crisp edges), left/right
panels are mirrored so front and back are symmetric, and all views share one palette,
one shading model and the same seam treatment.

Inputs : template.jpg        (blank back + side mannequins)
         front_back_v2.jpg   (hand-drawn front/back design: used for the front silhouette/shading)
Output : ducati_suit_v3.png
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage as nd

OUT = 'ducati_suit_v3.png'
SS = 4                                   # supersampling for panel edges
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

RED = (218, 36, 42)
BLACK = (28, 28, 31)
GREY = (160, 162, 166)      # stretch accordion + shoulder caps
SLIDER = (128, 130, 134)    # knee sliders
ELBOW = (52, 52, 56)        # elbow armour
CUP = (14, 14, 16)          # knee cup centre
INSERT = (226, 226, 228)    # shin armour insert (off-white)

tmpl = np.asarray(Image.open('template.jpg').convert('RGB')).astype(float)
v2 = np.asarray(Image.open('front_back_v2.jpg').convert('RGB')).astype(float)
H, W = tmpl.shape[:2]
FRONT_X = 560                            # columns < FRONT_X belong to the front view


# ------------------------------------------------------------------ silhouettes
def silhouette(img, x0, x1):
    g = img.mean(2)
    m = g < 247
    m[:, :x0] = False
    m[:, x1:] = False
    m = nd.binary_closing(m, iterations=1)
    holes = nd.binary_fill_holes(m) & ~m
    lab, n = nd.label(holes)
    if n:
        sizes = nd.sum(holes, lab, range(1, n + 1))
        m |= np.isin(lab, 1 + np.where(sizes < 600)[0])
    m = nd.binary_opening(m, iterations=1)
    lab, n = nd.label(m)
    sizes = nd.sum(m, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.where(sizes > 5000)[0])


front_m = silhouette(v2, 0, FRONT_X)
back_m = silhouette(tmpl, FRONT_X, 1060)
side_m = silhouette(tmpl, 1060, W)
body = front_m | back_m | side_m
soft_clip = lambda m: np.clip(nd.gaussian_filter(m.astype(float), 0.6) * 1.15, 0, 1) * nd.binary_dilation(m)

# ------------------------------------------------------------------ base image + shading
# Back and side come straight from the blank template. The front only exists painted, so
# rebuild its bare-mannequin tone: fill painted pixels from the nearest bare-white pixel,
# smooth, then add back the rim darkening the bare mannequins show near their outline.
base = tmpl.copy()
f = v2.copy()
fg = f.mean(2)
painted_front = front_m & (((f.max(2) - f.min(2)) > 22) | (fg < 200))
painted_front = nd.binary_dilation(painted_front, iterations=3) & front_m
known = front_m & ~painted_front
_, (iy, ix) = nd.distance_transform_edt(~known, return_indices=True)
filled = fg[iy, ix]
filled = nd.gaussian_filter(filled, 6)
dist = nd.distance_transform_edt(front_m)
# rim profile measured on the bare back view
bd = nd.distance_transform_edt(back_m)
bl = tmpl.mean(2)
prof = np.array([np.median(bl[back_m & (bd > k) & (bd <= k + 1)]) for k in range(12)])
prof = prof / prof[-1]
rim = np.interp(dist, np.arange(12) + 0.5, prof)
front_tone = np.where(painted_front, np.clip(filled, 0, 245) * rim, fg)
for c in range(3):
    base[..., c] = np.where(front_m, front_tone, base[..., c])
# anything outside the silhouettes on the front side -> clean background
bg = np.median(tmpl[5:40, 5:200].reshape(-1, 3), axis=0)
base[:, :FRONT_X][~front_m[:, :FRONT_X]] = bg

lum = base.mean(2)
REF = np.median(lum[body])
shade = np.clip(nd.gaussian_filter(lum / REF, 0.8), 0.55, 1.08)

# ------------------------------------------------------------------ panel drawing
col = np.zeros((H, W, 3))
cov = np.zeros((H, W))           # coverage of painted panels
pid = np.zeros((H, W), int)      # panel id (for seams)
_next = [1]


def catmull(pts, closed=True, n=10):
    """Smooth a control polygon with a Catmull-Rom spline."""
    P = np.array(pts, float)
    if len(P) < 3:
        return [tuple(p) for p in P]
    out = []
    N = len(P)
    rng = range(N) if closed else range(N - 1)
    for i in rng:
        p0, p1, p2, p3 = P[(i - 1) % N], P[i], P[(i + 1) % N], P[(i + 2) % N]
        if not closed:
            p0 = P[max(i - 1, 0)]
            p3 = P[min(i + 2, N - 1)]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)))
    if not closed:
        out.append(tuple(P[-1]))
    return out


def raster(draw_fn, bbox):
    """Supersampled coverage (0..1) of a shape inside bbox, returned as full-size array slice."""
    x0, y0, x1, y1 = [int(v) for v in (np.floor(bbox[0]) - 2, np.floor(bbox[1]) - 2,
                                        np.ceil(bbox[2]) + 2, np.ceil(bbox[3]) + 2)]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    im = Image.new('L', ((x1 - x0) * SS, (y1 - y0) * SS), 0)
    draw_fn(ImageDraw.Draw(im), x0, y0)
    im = im.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    return (slice(y0, y1), slice(x0, x1)), np.asarray(im).astype(float) / 255


def put(sl, a, color, clip):
    a = a * clip[sl]
    i = _next[0]
    _next[0] += 1
    c = np.array(color, float)
    col[sl] = col[sl] * (1 - a[..., None]) + c * a[..., None]
    cov[sl] = cov[sl] + (1 - cov[sl]) * a
    pid[sl] = np.where(a > 0.5, i, pid[sl])


def poly(pts, color, clip, smooth=True):
    p = catmull(pts) if smooth else pts
    xs, ys = [q[0] for q in p], [q[1] for q in p]
    sl, a = raster(lambda d, x0, y0: d.polygon([((x - x0) * SS, (y - y0) * SS) for x, y in p], fill=255),
                   (min(xs), min(ys), max(xs), max(ys)))
    put(sl, a, color, clip)


def ell(box, color, clip):
    (ax, ay), (bx, by) = box
    ax, bx = min(ax, bx), max(ax, bx)
    sl, a = raster(lambda d, x0, y0: d.ellipse([((ax - x0) * SS, (ay - y0) * SS),
                                                 ((bx - x0) * SS, (by - y0) * SS)], fill=255),
                   (ax, ay, bx, by))
    put(sl, a, color, clip)


def mir(pts, cx):
    return [(2 * cx - x, y) for x, y in pts]


def pair(fn, pts, color, clip, cx, **kw):
    fn(pts, color, clip, **kw)
    fn(mir(pts, cx), color, clip, **kw)


texts = []   # (string, centre, size, colour, angle, clip)


# ================================================================== FRONT  (centre x = 333)
C = 333.0
F = soft_clip(front_m)
# black: arm (inner/front half, shoulder to wrist)
pair(poly, [(226, 206), (262, 206), (262, 232), (256, 262), (246, 300), (238, 345), (232, 400),
            (226, 450), (222, 494), (206, 494), (207, 440), (208, 385), (210, 340), (214, 296),
            (220, 256), (222, 220)], BLACK, F, C)
# black: torso side stripe running down to the hip
pair(poly, [(250, 206), (266, 214), (268, 290), (266, 340), (262, 385), (254, 398), (248, 392),
            (248, 340), (250, 290), (252, 250)], BLACK, F, C)
# red shoulder yoke along the trapezius
pair(poly, [(302, 156), (286, 166), (262, 179), (240, 192), (232, 210), (244, 223), (258, 224),
            (266, 204), (284, 184), (304, 166)], RED, F, C)
# grey shoulder cap
pair(poly, [(206, 204), (220, 190), (242, 188), (248, 206), (246, 224), (226, 226), (206, 222)], GREY, F, C)
# red outer forearm (DUCATI sleeve panel)
pair(poly, [(198, 330), (210, 346), (210, 400), (211, 450), (212, 494), (186, 494), (186, 440),
            (188, 380), (192, 345)], RED, F, C)
# red hip flank
pair(poly, [(254, 384), (262, 394), (266, 424), (264, 452), (254, 470), (240, 484), (238, 462),
            (244, 430), (248, 402)], RED, F, C)
# black V-band from hip to crotch, continuing down the inner thigh to the knee
pair(poly, [(240, 470), (252, 454), (266, 446), (288, 462), (312, 480), (333, 488), (333, 510),
            (330, 530), (318, 560), (306, 600), (300, 640), (280, 640), (282, 600), (288, 550),
            (294, 512), (282, 494), (262, 478), (246, 482)], BLACK, F, C)
# silver accordion band above the knee
pair(poly, [(234, 638), (270, 634), (310, 636), (310, 668), (270, 666), (234, 670)], GREY, F, C, smooth=False)
# red knee panel with black lower border
pair(poly, [(234, 668), (272, 664), (310, 666), (308, 700), (300, 722), (286, 736), (262, 745),
            (242, 745), (234, 732)], BLACK, F, C)
pair(poly, [(234, 668), (272, 664), (310, 666), (306, 698), (298, 718), (284, 731), (262, 739),
            (243, 739), (234, 727)], RED, F, C)
# knee cup + slider puck
pair(ell, [(234, 690), (278, 742)], BLACK, F, C)
pair(ell, [(242, 700), (270, 734)], CUP, F, C)
pair(ell, [(250, 664), (284, 698)], SLIDER, F, C)
# shin armour insert
pair(poly, [(243, 752), (283, 752), (273, 834), (255, 834)], INSERT, F, C, smooth=False)
# ankle cuff: red band with black tab on the inside
pair(poly, [(244, 888), (268, 886), (292, 886), (292, 912), (268, 913), (244, 914)], RED, F, C, smooth=False)
pair(poly, [(278, 866), (289, 866), (292, 912), (280, 912)], BLACK, F, C, smooth=False)

texts += [('DUCATI', (C, 262), 29, BLACK, 0, F),
          ('DUCATI', (199, 424), 20, (255, 255, 255), -90, F),
          ('DUCATI', (2 * C - 199, 424), 20, (255, 255, 255), 90, F)]

# ================================================================== BACK  (centre x = 795)
C = 795.0
B = soft_clip(back_m)
# red shoulders
pair(poly, [(752, 168), (726, 180), (704, 192), (690, 208), (694, 224), (716, 224), (744, 212),
            (764, 200), (764, 170)], RED, B, C)
# grey shoulder caps
pair(poly, [(662, 204), (676, 192), (698, 190), (704, 208), (700, 226), (680, 228), (662, 224)], GREY, B, C)
# black arm-back + lat stripe down to the waist
pair(poly, [(700, 222), (730, 214), (742, 240), (740, 300), (736, 350), (740, 385), (744, 404),
            (724, 408), (712, 380), (702, 345), (690, 330), (682, 290), (686, 250)], BLACK, B, C)
# red forearm
pair(poly, [(652, 330), (690, 326), (706, 340), (705, 420), (703, 488), (652, 488), (648, 420)],
     RED, B, C)
# black inner-forearm strip
pair(poly, [(692, 330), (706, 334), (706, 410), (703, 488), (694, 488), (696, 410)], BLACK, B, C)
# elbow armour
pair(ell, [(652, 336), (690, 392)], ELBOW, B, C)
# red flanks + waist belt
poly([(716, 408), (722, 388), (738, 398), (760, 408), (795, 410), (830, 408), (852, 398), (868, 388),
      (874, 408), (884, 440), (882, 480), (868, 478), (830, 476), (795, 478), (760, 476), (722, 478),
      (708, 480), (706, 440)], RED, B, smooth=False)
# black seat / back-of-thigh panel
pair(poly, [(795, 500), (784, 494), (772, 520), (752, 558), (730, 590), (712, 615), (702, 642),
            (795, 642)], BLACK, B, C)
# silver accordion band
pair(poly, [(698, 638), (750, 636), (795, 638), (795, 668), (750, 666), (700, 670)], GREY, B, C, smooth=False)
# red back-of-knee with black centre
pair(poly, [(698, 666), (795, 666), (795, 708), (760, 710), (730, 714), (704, 718), (698, 700)],
     RED, B, C, smooth=False)
pair(poly, [(722, 670), (770, 670), (764, 708), (730, 712)], BLACK, B, C, smooth=False)
# ankle: red cuff with black heel cup
pair(poly, [(706, 864), (728, 862), (750, 864), (750, 914), (728, 916), (706, 914)], RED, B, C, smooth=False)
pair(poly, [(714, 880), (726, 872), (740, 878), (744, 908), (728, 911), (712, 908)], BLACK, B, C)
# black collar / yoke panel (last on the upper back so it sits on top)
poly([(772, 138), (818, 138), (826, 168), (852, 178), (842, 222), (822, 268), (768, 268),
      (748, 222), (738, 178), (764, 168)], BLACK, B, smooth=False)

texts += [('DUCATI', (C, 198), 22, (255, 255, 255), 0, B),
          ('DUCATI', (674, 425), 18, (255, 255, 255), 90, B),
          ('DUCATI', (2 * C - 674, 425), 18, (255, 255, 255), -90, B)]

# ================================================================== SIDE (facing right)
S_ = soft_clip(side_m)
arm = np.zeros((H, W))
sl, a = raster(lambda d, x0, y0: d.polygon([((x - x0) * SS, (y - y0) * SS) for x, y in catmull(
    [(1140, 196), (1175, 186), (1212, 196), (1224, 214), (1221, 245), (1216, 262), (1210, 300),
     (1203, 340), (1204, 380), (1210, 420), (1220, 460), (1238, 492), (1192, 497), (1180, 480),
     (1172, 460), (1165, 440), (1160, 420), (1156, 400), (1150, 360), (1138, 300), (1134, 250)])],
    fill=255), (1120, 180, 1250, 510))
arm[sl] = a
hand = np.zeros((H, W))
sl, a = raster(lambda d, x0, y0: d.polygon([((x - x0) * SS, (y - y0) * SS) for x, y in catmull(
    [(1190, 494), (1238, 489), (1252, 520), (1260, 552), (1250, 578), (1224, 582), (1203, 572),
     (1194, 540)])], fill=255), (1180, 480, 1270, 590))
hand[sl] = a
ARM = arm * (1 - hand) * S_
TOR = (1 - arm) * (1 - hand) * S_

# torso / legs
poly([(1166, 150), (1184, 150), (1192, 180), (1178, 194), (1150, 194)], BLACK, TOR)            # collar
poly([(1206, 360), (1211, 390), (1220, 425), (1232, 450), (1222, 460), (1200, 466), (1160, 474),
      (1136, 476), (1136, 418), (1160, 412), (1190, 402), (1206, 380)], RED, TOR, smooth=False)  # flank + belt
poly([(1190, 460), (1222, 458), (1244, 472), (1262, 490), (1270, 504), (1260, 510), (1240, 490),
      (1216, 476), (1190, 474)], BLACK, TOR)                                                   # V-band
poly([(1140, 505), (1158, 510), (1176, 540), (1194, 578), (1204, 612), (1208, 640), (1150, 640)],
     BLACK, TOR)                                                                               # back of thigh
poly([(1150, 638), (1200, 634), (1255, 632), (1250, 668), (1200, 670), (1150, 674)], GREY, TOR, smooth=False)
poly([(1145, 668), (1200, 666), (1250, 664), (1244, 705), (1236, 742), (1220, 757), (1196, 756),
      (1170, 745), (1145, 736)], BLACK, TOR)
poly([(1145, 668), (1200, 666), (1250, 664), (1243, 703), (1234, 736), (1219, 750), (1196, 749),
      (1172, 739), (1145, 730)], RED, TOR)                                                     # knee
poly([(1140, 674), (1166, 673), (1170, 700), (1164, 724), (1140, 724)], BLACK, TOR)           # back of knee
ell([(1196, 684), (1244, 744)], BLACK, TOR)
ell([(1206, 696), (1238, 734)], CUP, TOR)
ell([(1210, 654), (1248, 694)], SLIDER, TOR)
poly([(1193, 758), (1213, 758), (1206, 834), (1195, 834)], INSERT, TOR, smooth=False)       # shin insert
poly([(1150, 866), (1215, 864), (1220, 912), (1150, 914)], RED, TOR, smooth=False)          # ankle cuff
poly([(1150, 876), (1164, 872), (1176, 884), (1180, 912), (1150, 914)], BLACK, TOR)          # heel cup

# arm (outer face: white upper arm edged in black, red forearm, pucks)
poly([(1126, 214), (1228, 208), (1222, 262), (1210, 305), (1204, 350), (1170, 354), (1140, 350),
      (1130, 300)], BLACK, ARM)                                                                # upper arm
poly([(1140, 344), (1180, 350), (1210, 346), (1216, 400), (1246, 500), (1180, 506), (1150, 420)],
     RED, ARM)
poly([(1206, 346), (1214, 346), (1222, 400), (1246, 496), (1236, 498), (1210, 400)], BLACK, ARM)
poly([(1140, 350), (1156, 352), (1166, 410), (1180, 460), (1196, 504), (1180, 506), (1160, 450),
      (1146, 400)], BLACK, ARM)                                                                # back-edge piping
poly([(1128, 212), (1150, 190), (1176, 182), (1200, 178), (1222, 184), (1232, 198), (1214, 214),
      (1180, 230), (1150, 236), (1128, 236)], RED, S_ * (1 - hand))                              # shoulder yoke
ell([(1156, 196), (1200, 234)], GREY, ARM)
ell([(1146, 334), (1180, 388)], ELBOW, ARM)
texts.append(('DUCATI', (1192, 428), 19, (255, 255, 255), -76, ARM))

# ------------------------------------------------------------------ lettering
for s, (cx, cy), size, colr, ang, clip in texts:
    fnt = ImageFont.truetype(FONT, size * SS)
    bb = fnt.getbbox(s)
    t = Image.new('L', (bb[2] - bb[0] + 8 * SS, bb[3] - bb[1] + 8 * SS), 0)
    ImageDraw.Draw(t).text((4 * SS - bb[0], 4 * SS - bb[1]), s, font=fnt, fill=255)
    t = t.rotate(ang, expand=True, resample=Image.BICUBIC)
    w, h = t.width // SS, t.height // SS
    t = t.resize((w, h), Image.LANCZOS)
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    put(sl, np.asarray(t).astype(float) / 255, colr, clip)

# ------------------------------------------------------------------ composite
S3 = shade[..., None]
lit = col * np.clip(S3, 0.55, 1.08) ** 1.25
# soft leather sheen on the highlights
sheen = np.clip((shade - 0.985) / 0.07, 0, 1)[..., None]
lit = lit + (255 - lit) * 0.22 * sheen * (col.max(2, keepdims=True) < 200)
lit = np.clip(lit, 0, 255)
a = np.clip(cov, 0, 1)[..., None]
out = base * (1 - a) + lit * a

# seams: darken 1px where two different panels (or panel and bare suit) meet
lab = np.where(cov > 0.5, pid, 0)
edge = (nd.maximum_filter(lab, 3) != nd.minimum_filter(lab, 3)) & body
edge &= nd.binary_dilation(cov > 0.5, iterations=1)
soft = nd.gaussian_filter(edge.astype(float), 0.5)
out *= (1 - 0.28 * np.clip(soft * 1.6, 0, 1))[..., None]
# side view: faint outline where the arm overlaps the torso
ae = nd.binary_dilation(arm > 0.5, iterations=2) & ~(arm > 0.5) & side_m & ~nd.binary_dilation(hand > 0.5, iterations=3)
ae[:240] = False                         # shoulder yoke spans arm + torso: no outline there
out[ae] *= 0.4

Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(OUT)
print('saved', OUT)
