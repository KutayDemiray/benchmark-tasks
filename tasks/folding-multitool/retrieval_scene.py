"""M6 DIN 6334 coupling nut, rectangular maintenance recess and catch tray.

Dimensions are SI. The hole is a real opening in both rendering and collision;
the nut is not a solid hexagon. Threads and edge chamfers are omitted.
"""
import numpy as np
from geometry import box, prism
from compound_asset import Compound

TABLE_Z = .7475
POCKET_XY = np.array([.24, 0.])
POCKET_HALF = np.array([.012, .012])
COUPON_HEIGHT = .038
POCKET_DEPTH = .028
POCKET_FLOOR = TABLE_Z+COUPON_HEIGHT-POCKET_DEPTH
POCKET_RIM = TABLE_Z+COUPON_HEIGHT
NUT_AF, NUT_HEIGHT, BORE_RADIUS = .010, .018, .0025
# Accu HTBN-M6-A2 catalogue: 957 g per 100 pieces; do not tune mass for stability.
NUT_MASS = .00957
NUT_SOURCE = "https://www.accu.co.uk/hexagonal-stud-connector-nuts/62290-HTBN-M6-A2"
TRAY_XY = np.array([.49, 0.])
TRAY_HALF = .050
TRAY_FLOOR = TABLE_Z+.003
TRAY_TOP = TRAY_FLOOR+.025


def nut_spec():
    pieces = []
    angles = np.linspace(0., 2*np.pi, 13)
    normals = np.pi/2+np.arange(6)*np.pi/3
    def outer(a):
        return NUT_AF/2/np.cos(a-normals).max()*np.array([np.cos(a), np.sin(a)])
    for i, (a, b) in enumerate(zip(angles[:-1], angles[1:])):
        xy = np.array([BORE_RADIUS*np.array([np.cos(a), np.sin(a)]), outer(a),
                       outer(b), BORE_RADIUS*np.array([np.cos(b), np.sin(b)])])
        pieces.append(prism(f"nut_sector_{i:02d}", xy, 0., NUT_HEIGHT, (.68, .67, .62)))
    return Compound(name="nut", pos=(*POCKET_XY, POCKET_FLOOR+.0002),
                    size=(NUT_AF/np.cos(np.pi/6), NUT_AF, NUT_HEIGHT),
                    mass=NUT_MASS, pieces=tuple(pieces))


def fixtures():
    color = (.40, .44, .48)
    base_h = COUPON_HEIGHT-POCKET_DEPTH
    x, y = POCKET_HALF
    parts = [box("pocket_floor", (-.04, -.03, 0.), (.04, .03, base_h), color),
             box("pocket_left", (-.04, -.03, base_h), (-x, .03, COUPON_HEIGHT), color),
             box("pocket_right", (x, -.03, base_h), (.04, .03, COUPON_HEIGHT), color),
             box("pocket_front", (-x, -.03, base_h), (x, -y, COUPON_HEIGHT), color),
             box("pocket_back", (-x, y, base_h), (x, .03, COUPON_HEIGHT), color)]
    coupon = Compound(name="bolted_coupon", pos=(*POCKET_XY, TABLE_Z),
                      size=(.08, .06, COUPON_HEIGHT), mass=None, pieces=tuple(parts), box_primitives=True)
    c = (.16, .30, .42)
    tray_parts = [box("tray_floor", (-.053, -.053, 0.), (.053, .053, .003), c),
                  box("tray_left", (-.053, -.053, .003), (-.05, .053, .028), c),
                  box("tray_right", (.05, -.053, .003), (.053, .053, .028), c),
                  box("tray_front", (-.05, -.053, .003), (.05, -.05, .028), c),
                  box("tray_back", (-.05, .05, .003), (.05, .053, .028), c)]
    tray = Compound(name="tray", pos=(*TRAY_XY, TABLE_Z), size=(.106, .106, .028),
                    mass=None, pieces=tuple(tray_parts), friction=(.45, .35), box_primitives=True)
    return [coupon, tray]
