"""Inactive load-test fixture geometry; deployment must pass before spawning it."""
from geometry import box
from compound_asset import Compound

STEEL_DENSITY = 7850.
GRAVITY = 9.81
MASS = 1./GRAVITY
BASE_WIDTH = .030
STEM_WIDTH = .010
STEM_HEIGHT = .010
BASE_HEIGHT = (MASS/STEEL_DENSITY-STEM_WIDTH**2*STEM_HEIGHT)/BASE_WIDTH**2
BITE_HEIGHT = BASE_HEIGHT+STEM_HEIGHT/2


def gauge_spec():
    # A broader base is stable on the table. The 10 mm square stem is pinched
    # above the shoulder, so a carried weight cannot merely hang on a jaw.
    pieces = (box("steel_base", (-BASE_WIDTH/2, -BASE_WIDTH/2, 0.),
                  (BASE_WIDTH/2, BASE_WIDTH/2, BASE_HEIGHT)),
              box("gripping_stem", (-STEM_WIDTH/2, -STEM_WIDTH/2, BASE_HEIGHT),
                  (STEM_WIDTH/2, STEM_WIDTH/2, BASE_HEIGHT+STEM_HEIGHT)))
    return Compound(name="load_gauge", pos=(.24, 0., .7475),
                    size=(BASE_WIDTH, BASE_WIDTH, BASE_HEIGHT+STEM_HEIGHT),
                    mass=MASS, pieces=pieces)
