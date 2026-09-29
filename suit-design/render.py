from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from scipy import ndimage as nd

src = Image.open('template.jpg').convert('RGB')
A = np.asarray(src).astype(float)
mask = np.load('mask.npy')
H, W = mask.shape
lum = A.mean(2)

RED = (200, 16, 26); BLACK = (16, 16, 18); SILVER = (150, 152, 158); DGREY = (70, 72, 76); WHITE = (238, 238, 238)
FONT = '/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf'

ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
extra = np.zeros((H, W), bool)   # area allowed outside the body silhouette (aero hump)

def shape(draw_fn, c, smooth=4):
    """Draw a shape, round its corners (blur + threshold), composite onto overlay."""
    m = Image.new('L', (W, H), 0)
    draw_fn(ImageDraw.Draw(m))
    if smooth:
        m = m.filter(ImageFilter.GaussianBlur(smooth)).point(lambda v: 255 if v > 127 else 0)
        m = m.filter(ImageFilter.GaussianBlur(0.7))
    layer = Image.new('RGBA', (W, H), c + (255,))
    layer.putalpha(m)
    ov.alpha_composite(layer)
    return m

def P(pts, c, s=4): return shape(lambda d: d.polygon(pts, fill=255), c, s)
def E(box, c, s=0): return shape(lambda d: d.ellipse(box, fill=255), c, s)
def mirror(pts, cx=800): return [(2 * cx - x, y) for x, y in pts]
ident = lambda p: p

def text(s, center, size, color, angle=0):
    f = ImageFont.truetype(FONT, size)
    bb = f.getbbox(s)
    t = Image.new('RGBA', (bb[2] + 10, bb[3] + 10), (0, 0, 0, 0))
    ImageDraw.Draw(t).text((5 - bb[0], 5 - bb[1]), s, font=f, fill=color)
    t = t.rotate(angle, expand=True, resample=Image.BICUBIC)
    ov.alpha_composite(t, (int(center[0] - t.width / 2), int(center[1] - t.height / 2)))

# ================= BACK VIEW (centre x = 800) =================
for S in (ident, mirror):
    # red shoulder yoke wrapping over from the front
    P(S([(772, 148), (745, 166), (705, 180), (674, 198), (660, 232), (666, 262), (700, 252), (738, 222), (770, 198), (792, 184)]), RED)
    # red side / lat panel
    P(S([(700, 262), (714, 272), (724, 360), (718, 425), (700, 420), (698, 300)]), RED)
    # black waist band sweeping down to the seat
    P(S([(700, 412), (720, 420), (762, 440), (800, 452), (800, 474), (756, 462), (704, 442)]), BLACK, 3)
    # black inner rear-thigh strip
    P(S([(776, 522), (800, 514), (800, 545), (792, 600), (786, 635), (770, 635), (772, 580)]), BLACK)
    # silver stretch accordion behind the knee
    P(S([(691, 632), (789, 632), (792, 685), (696, 685)]), SILVER, 3)
    # red calf panel with black lower edge
    P(S([(694, 682), (792, 682), (788, 705), (758, 752), (732, 782), (712, 786), (698, 752)]), RED)
    P(S([(698, 752), (712, 786), (732, 782), (728, 796), (706, 800), (694, 772)]), BLACK, 2)
    # boot + red heel counter
    P(S([(688, 848), (800, 848), (800, 1000), (620, 1000), (620, 900)]), BLACK, 2)
    P(S([(716, 850), (748, 850), (745, 902), (720, 902)]), RED, 3)
    # arm: black outer upper arm, red forearm, black cuff
    P(S([(660, 240), (700, 234), (706, 300), (703, 345), (680, 352), (650, 350), (649, 290)]), BLACK)
    P(S([(650, 350), (680, 352), (703, 345), (706, 420), (701, 478), (646, 478), (644, 410)]), RED)
    P(S([(644, 474), (703, 474), (703, 490), (644, 490)]), BLACK, 2)
    # gloves
    P(S([(630, 488), (697, 488), (697, 600), (630, 600)]), BLACK, 0)
# shoulder pucks + elbow sliders
for bx in (656, 912):
    E([(bx, 206), (bx + 32, 244)], SILVER)
for bx in (654, 912):
    E([(bx, 320), (bx + 34, 352)], DGREY)

# aero hump: black rim, white body, red cap
P([(750, 176), (768, 158), (832, 158), (850, 176), (856, 285), (800, 306), (744, 285)], BLACK, 8)
P([(756, 181), (772, 166), (828, 166), (844, 181), (849, 281), (800, 299), (751, 281)], WHITE, 8)
P([(756, 181), (772, 166), (828, 166), (844, 181), (846, 214), (754, 214)], RED, 6)
text('DUCATI', (800, 334), 27, BLACK)
text('DUCATI', (675, 413), 22, (255, 255, 255), 90)
text('DUCATI', (925, 413), 22, (255, 255, 255), -90)

# ================= SIDE VIEW (facing right) =================
# aero hump bulge behind the neck
hm = E([(1126, 162), (1186, 300)], WHITE)
hm_np = np.asarray(hm) > 0
hm_np &= (np.arange(W)[None, :] < 1178)
extra |= hm_np
# red cap on top of hump
capm = Image.new('L', (W, H), 0); ImageDraw.Draw(capm).ellipse([(1126, 162), (1186, 300)], fill=255)
cap = (np.asarray(capm) > 0) & (np.arange(H)[:, None] < 205) & (np.arange(W)[None, :] < 1178)
ovn = np.asarray(ov).copy(); ovn[cap] = RED + (255,); ov = Image.fromarray(ovn)
# shoulder yoke
P([(1148, 192), (1185, 184), (1222, 190), (1240, 206), (1228, 226), (1180, 226), (1148, 216)], RED)
# red hip wedge + black band sloping from lower back to crotch
P([(1145, 330), (1170, 330), (1178, 420), (1145, 420)], RED)
P([(1136, 416), (1180, 416), (1262, 468), (1266, 500), (1240, 514), (1176, 452), (1136, 442)], BLACK, 3)
# outer-thigh red piping
P([(1182, 460), (1192, 460), (1228, 625), (1219, 627)], RED, 1)
# knee: silver accordion, red surround, black slider + grey cap
P([(1180, 628), (1248, 622), (1240, 672), (1186, 674)], SILVER, 3)
P([(1178, 672), (1240, 668), (1238, 740), (1214, 766), (1186, 756), (1176, 718)], RED)
E([(1190, 678), (1236, 742)], BLACK)
E([(1204, 672), (1234, 698)], DGREY)
# calf flash
P([(1140, 730), (1170, 720), (1180, 760), (1160, 792), (1140, 792)], RED)
# boot with red front and rear panels
P([(1130, 848), (1240, 848), (1320, 1000), (1100, 1000)], BLACK, 2)
P([(1196, 850), (1226, 850), (1230, 905), (1204, 905)], RED, 3)
P([(1150, 852), (1168, 852), (1168, 905), (1150, 905)], RED, 3)
# arm on top: black upper arm, red forearm, cuff, glove, pucks
P([(1150, 212), (1190, 204), (1226, 218), (1222, 262), (1214, 340), (1162, 348), (1148, 280)], BLACK)
P([(1162, 348), (1214, 340), (1230, 400), (1244, 486), (1192, 492), (1172, 430)], RED)
P([(1190, 484), (1245, 478), (1248, 500), (1193, 505)], BLACK, 2)
P([(1195, 500), (1248, 496), (1256, 535), (1249, 572), (1226, 579), (1203, 572), (1196, 535)], BLACK, 3)
E([(1170, 222), (1208, 262)], SILVER)
E([(1156, 318), (1186, 350)], DGREY)
text('DUCATI', (1206, 415), 22, (255, 255, 255), -82)

# ================= FRONT VIEW additions: gloves + boots =================
P([(170, 488), (238, 488), (238, 600), (170, 600)], BLACK, 0)
P([(432, 488), (500, 488), (500, 600), (432, 600)], BLACK, 0)
P([(200, 850), (300, 850), (300, 1000), (200, 1000)], BLACK, 0)
P([(360, 850), (470, 850), (470, 1000), (360, 1000)], BLACK, 0)

# ================= composite =================
O = np.asarray(ov).astype(float)
alpha = O[..., 3] / 255.0
allowed = mask | extra
F = A.astype(int)
isred = (F[..., 0] > 170) & (F[..., 1] < 120) & (F[..., 2] < 120)
keep_front_red = (np.arange(W)[None, :] < 520) & isred
alpha = alpha * allowed * ~keep_front_red
alpha = nd.gaussian_filter(alpha, 0.5)  # soften clip edges
alpha = alpha[..., None]

# shading from the mannequin: normalise so its typical tone -> 1
ref = np.median(lum[mask])
shade = lum / ref
hump_out = extra & ~mask
xx = np.arange(W)[None, :].repeat(H, 0)
shade[hump_out] = np.clip(0.78 + (xx[hump_out] - 1126) / 160.0, 0.78, 1.0)
shade = np.clip(nd.gaussian_filter(shade, 1.0), 0.35, 1.15)[..., None]
col = O[..., :3]
dark = shade ** 2.2
gloss = 38 * np.clip((shade - 0.99) / 0.12, 0, 1)            # leather sheen on highlights
lit = np.clip(col * dark + gloss, 0, 255)
# white panels keep the mannequin's own tone
is_white = (col.min(2) > 230)[..., None]
lit = np.where(is_white, A, lit)
hump_all = extra[..., None]
hshade = np.clip(0.80 + (xx - 1126) / 150.0, 0.8, 1.02)[..., None]
lit = np.where(hump_all & is_white, np.clip(242 * hshade, 0, 255), lit)
lit = np.where(hump_all & ~is_white, np.clip(col * hshade ** 2, 0, 255), lit)
out = A * (1 - alpha) + lit * alpha

# darken panel seams slightly
key = (O[..., 0] // 50) * 100 + (O[..., 1] // 50) * 10 + O[..., 2] // 50
key = np.where(O[..., 3] > 128, key, -1)
edge = (nd.maximum_filter(key, 3) != nd.minimum_filter(key, 3)) & allowed & (O[..., 3] > 0)
out[edge] *= 0.8
# outline hump where it leaves the body silhouette
hb = extra & ~nd.binary_erosion(extra, iterations=2) & ~nd.binary_erosion(mask, iterations=4)
out[hb] = out[hb] * 0.55

Image.fromarray(out.astype(np.uint8)).save('ducati_suit_front_back_side.png')
