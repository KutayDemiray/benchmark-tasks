"""Unfold the multitool and use its working jaws to retrieve the recessed nut
and release it into the tray. Deployment and load-gate diagnostic modes omit
the nut. All tool joints are passive and all control uses stock YAM actions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from yam_arena.task_file import Box
from mechanism import load_spec
from geometry import reset_root, SUPPORT_HEIGHT
from tool_asset import FoldTool
import adapter

SCENE = "station"
SUCCESS_HOLD_S = 0.
SCORE_OVER = "end"
SPEC = load_spec()
OBJECTS = [FoldTool(name="multitool", pos=tuple(float(v) for v in reset_root(SPEC)), mass=.1885,
                    size=(.1016, .029972, .01651), spec=SPEC)]
for name, x in (("support_near", .342), ("support_far", .418)):
    OBJECTS.append(Box(name=name, pos=(x, 0., .7475+SUPPORT_HEIGHT/2), size=(.008, .045, SUPPORT_HEIGHT),
                       mass=None, color=(.56, .36, .17)))
if adapter.PAYLOAD is not None:
    OBJECTS.append(adapter.PAYLOAD)
OBJECTS.extend(adapter.FIXTURES)


def score(objects):
    return adapter.monitor.score


def success(objects):
    return adapter.monitor.score == 1.
