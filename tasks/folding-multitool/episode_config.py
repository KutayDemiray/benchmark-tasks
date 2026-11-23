"""Shared submission-episode contract for scripted and Dream evaluation."""
INSTRUCTION = (
    "Unfold the folded butterfly multitool pliers. Use the working jaws to grip "
    "the loose coupling nut in the rectangular recess, lift it out, carry it to the blue tray, "
    "and release it inside. Move the tool and both robot grippers clear and leave "
    "the nut resting in the tray. Complete the task within 180 seconds."
)
CAMERAS = ("top_cam", "left_cam", "right_cam")
WIDTH, HEIGHT = 320, 176
CONTROL_HZ = 20
MAX_STEPS = 3600
SEED = 0
MODEL = "dreamlabs/dream-1-beta"
TOP_CAMERA_VARIANT = "demo"  # Native fixed camera: 15 cm lower, 5 degrees re-aimed.
REVISION = "coupling-nut-rectangular-recess-v1"

# Files that define the common scene, reset, passive mechanics and scorer.
# Controllers are intentionally excluded: a demonstration is allowed privileged
# feedback, while Dream is restricted to the evaluation.py observation boundary.
EPISODE_FILES = (
    "episode_config.py", "mechanism.json", "mechanism.py", "geometry.py",
    "tool_asset.py", "passive_actuator.py", "compound_asset.py", "adapter.py",
    "retrieval_scene.py", "task.py", "scorer.py", "retrieval_score.py",
    "jaw_contact.py", "load_score.py", "clearance.py", "stock_geometry.py",
)


def contract(directory):
    import hashlib
    from pathlib import Path
    return {
        "revision": REVISION, "instruction": INSTRUCTION, "cameras": list(CAMERAS),
        "top_camera_variant": TOP_CAMERA_VARIANT,
        "rgb_shape": [HEIGHT, WIDTH, 3], "control_hz": CONTROL_HZ,
        "max_actions": MAX_STEPS, "seed": SEED, "physics_hz": 240,
        "source_hashes": {name: hashlib.sha256((Path(directory)/name).read_bytes()).hexdigest()
                          for name in EPISODE_FILES},
    }
