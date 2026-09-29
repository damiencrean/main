"""Paint the side view of front_back_v2.jpg to match its front and back designs.

Front and back views are left untouched; only the right-hand (side) mannequin is painted.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from scipy import ndimage as nd

SRC = 'front_back_v2.jpg'
OUT = 'ducati_suit_v2_front_back_side.png'

src = Image.open(SRC).convert('RGB')
A = np.asarray(src).astype(float)
H, W = A.shape[:2]
lum = A.mean(2)

# body silhouette of the side mannequin
m = lum < 249
m = nd.binary_closing(m, iterations=3)
m = nd.binary_fill_holes(m)
m = nd.binary_opening(m, iterations=2)
m[:, :1080] = False
lab, n = nd.label(m)
mask = lab == (1 + np.argmax(nd.sum(m, lab, range(1, n + 1))))

# colours sampled from the front/back views
RED = (234, 55, 54); BLACK = (36, 36, 36); GREY = (168, 168, 168)
SLIDER = (150, 150, 150); ELBOW = (60, 58, 58); CUP = (22, 22, 22)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def poly_mask(pts, smooth=3):
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).polygon(pts, fill=255)
    if smooth:
        im = im.filter(ImageFilter.GaussianBlur(smooth)).point(lambda v: 255 if v > 127 else 0)
    return np.asarray(im) > 0


def ell_mask(box):
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).ellipse(box, fill=255)
    return np.asarray(im) > 0


col = np.full((H, W, 3), -1.0)   # -1 = unpainted (keep mannequin white)


def paint(region, c, clip):
    r = region & clip
    col[r] = c
    return r


# ---- arm (sits in front of the torso) ----
arm = poly_mask([(1140, 196), (1175, 186), (1212, 196), (1224, 214), (1221, 245), (1216, 262),
                 (1210, 300), (1203, 340), (1204, 380), (1210, 420), (1220, 460), (1238, 492),
                 (1192, 497), (1180, 480), (1172, 460), (1165, 440), (1160, 420), (1156, 400),
                 (1150, 360), (1138, 300), (1134, 250)], 2) & mask
hand = poly_mask([(1190, 494), (1238, 489), (1252, 520), (1260, 552), (1250, 578), (1224, 582),
                  (1203, 572), (1194, 540)], 2) & mask
arm &= ~hand
torso = mask & ~arm & ~hand

# ---- torso / legs ----
# black collar at the back of the neck (continuation of the back yoke panel)
paint(poly_mask([(1160, 152), (1184, 150), (1190, 176), (1172, 196), (1140, 200)]), BLACK, torso)
# red yoke over the top of the shoulder, wrapping from front to back
paint(poly_mask([(1140, 196), (1172, 188), (1190, 176), (1212, 176), (1238, 196), (1234, 214),
                 (1222, 222), (1180, 232), (1136, 234)]), RED, torso)
# black side stripe running down from the armpit, just in front of the arm
paint(poly_mask([(1214, 224), (1230, 232), (1224, 270), (1216, 320), (1210, 350), (1206, 350)]), BLACK, torso)
# red flank: tapers up from the hip and wraps round to the red waist belt on the back
paint(poly_mask([(1206, 360), (1211, 390), (1220, 425), (1232, 450), (1222, 460), (1200, 466),
                 (1160, 474), (1136, 476), (1136, 418), (1160, 412), (1190, 402), (1206, 380)]), RED, torso)
# black V-band: from the hip, sweeping down to the crotch
paint(poly_mask([(1190, 460), (1222, 458), (1244, 472), (1262, 490), (1270, 504), (1260, 510),
                 (1240, 490), (1216, 476), (1190, 474)], 2), BLACK, torso)
# black back-of-thigh panel from under the seat down to the knee
paint(poly_mask([(1140, 505), (1158, 510), (1176, 540), (1194, 578), (1204, 612), (1208, 640),
                 (1150, 640)]), BLACK, torso)
# silver accordion stretch band above the knee
paint(poly_mask([(1150, 638), (1255, 632), (1250, 668), (1150, 674)], 1), GREY, torso)
# red knee surround with black lower edge
knee_red = poly_mask([(1145, 672), (1250, 666), (1244, 705), (1236, 740), (1220, 754), (1196, 752),
                      (1172, 740), (1145, 730)])
knee_edge = poly_mask([(1145, 726), (1172, 736), (1196, 748), (1220, 750), (1236, 736), (1240, 744),
                       (1222, 758), (1194, 757), (1168, 746), (1145, 736)], 1)
paint(knee_red, RED, torso)
paint(knee_edge, BLACK, torso)
# black centre panel at the back of the knee (flanked by red, as on the back view)
paint(poly_mask([(1140, 674), (1166, 673), (1170, 700), (1164, 724), (1140, 724)]), BLACK, torso)
# knee slider: black cup with darker centre, grey top slider
paint(ell_mask([(1196, 682), (1244, 744)]), BLACK, torso)
paint(ell_mask([(1206, 694), (1238, 736)]), CUP, torso)
paint(ell_mask([(1208, 652), (1246, 694)]), SLIDER, torso)
# ankle: red cuff, black vertical strip, black heel cup
paint(poly_mask([(1150, 866), (1215, 864), (1220, 912), (1150, 914)], 1), RED, torso)
paint(poly_mask([(1184, 862), (1194, 862), (1198, 914), (1188, 914)], 1), BLACK, torso)
paint(poly_mask([(1150, 878), (1164, 876), (1174, 890), (1176, 914), (1150, 914)], 2), BLACK, torso)

# ---- arm panels ----
# red over the top of the shoulder
paint(poly_mask([(1130, 180), (1225, 180), (1230, 214), (1200, 222), (1170, 228), (1130, 232)]), RED, arm)
# black upper arm down to the elbow
paint(poly_mask([(1128, 222), (1170, 228), (1200, 222), (1232, 214), (1225, 300), (1206, 350),
                 (1180, 352), (1150, 346), (1128, 300)], 2), BLACK, arm)
# red forearm down to the wrist
paint(poly_mask([(1140, 346), (1180, 352), (1206, 350), (1215, 400), (1245, 500), (1180, 505),
                 (1150, 420)], 2), RED, arm)
# grey shoulder puck and dark elbow patch
paint(ell_mask([(1156, 198), (1200, 234)]), GREY, arm)
paint(ell_mask([(1146, 336), (1180, 386)]), ELBOW, arm)

# ---- composite with mannequin shading ----
painted = col[..., 0] >= 0
ref = np.median(lum[mask])
shade = np.clip(nd.gaussian_filter(lum / ref, 1.0), 0.6, 1.1)[..., None]
lit = np.clip(np.where(painted[..., None], col, 0) * shade ** 0.9, 0, 255)

# forearm lettering (white, reading top-to-bottom like the front/back sleeves)
txt = Image.new('RGBA', (W, H), (0, 0, 0, 0))
f = ImageFont.truetype(FONT, 21)
bb = f.getbbox('DUCATI')
t = Image.new('RGBA', (bb[2] + 10, bb[3] + 10), (0, 0, 0, 0))
ImageDraw.Draw(t).text((5 - bb[0], 5 - bb[1]), 'DUCATI', font=f, fill=(255, 255, 255, 255))
t = t.rotate(-76, expand=True, resample=Image.BICUBIC)
txt.alpha_composite(t, (int(1196 - t.width / 2), int(426 - t.height / 2)))
T = np.asarray(txt).astype(float)
ta = (T[..., 3] / 255.0 * arm)[..., None]

alpha = nd.gaussian_filter(painted.astype(float), 0.6)[..., None] * mask[..., None]
out = A * (1 - alpha) + lit * alpha
out = out * (1 - ta) + T[..., :3] * ta

# thin seams between panels
key = np.where(painted, col[..., 0] * 7 + col[..., 1] * 3 + col[..., 2], -1)
edge = (nd.maximum_filter(key, 3) != nd.minimum_filter(key, 3)) & mask
edge &= nd.binary_dilation(painted, iterations=1)
out[edge] *= 0.82
# arm outline where it overlaps the torso
ae = (nd.binary_dilation(arm, iterations=1) & ~arm) & torso & ~nd.binary_dilation(hand, iterations=3)
out[ae] *= 0.7

# shin armour outline (as on the front view)
res = Image.fromarray(out.astype(np.uint8))
ImageDraw.Draw(res).line([(1192, 758), (1212, 758), (1205, 834), (1194, 834), (1192, 758)],
                         fill=(40, 40, 40), width=1)
res.save(OUT)
print('saved', OUT)
