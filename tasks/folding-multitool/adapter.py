"""Process-local native integration and read-only measurement; no environment edits."""
import numpy as np
from mechanism import BODIES, load_spec, jaw_gap
from scorer import DeploymentScorer

monitor = DeploymentScorer()
last_record = {}
PAD_NAMES = ("left_tip_left", "left_tip_right", "right_tip_left", "right_tip_right")
_step = 0
MODE = "deployment-gate"
PAYLOAD = None
PAYLOAD_PARTNERS = []
FIXTURES = []


def configure(mode):
    global MODE, PAYLOAD, FIXTURES
    MODE = mode
    PAYLOAD, FIXTURES = None, []
    if mode == "load-gate":
        from load_gauge import gauge_spec
        PAYLOAD = gauge_spec()
    elif mode == "retrieval":
        from retrieval_scene import nut_spec, fixtures
        PAYLOAD, FIXTURES = nut_spec(), fixtures()


def array(value):
    value = value.torch if hasattr(value, "torch") else value
    return value.detach().cpu().numpy().astype(float)


def install():
    """Extend this process's object factory; keep installed files unmodified."""
    from yam_arena import scene
    from tool_asset import FoldTool, make_object
    from compound_asset import Compound, make_object as make_compound
    original_object, original_build, original_states = scene.task_object, scene.build_task_environment, scene.object_states

    def task_object(spec):
        if isinstance(spec, Compound):
            return make_compound(spec)
        return make_object(spec) if isinstance(spec, FoldTool) else original_object(spec)

    def build(embodiment, task, control_hz=None):
        add_sensors(embodiment)
        return original_build(embodiment, task, control_hz)

    def states(env, names):
        global _step
        _step += 1
        measure(env, _step)
        # Native bounds querying does not expose individual articulation links.
        # The score consumes measured link poses and contacts instead.
        return original_states(env, [n for n in names if n != "multitool"])

    scene.task_object, scene.build_task_environment, scene.object_states = task_object, build, states


def add_sensors(embodiment):
    global PAYLOAD_PARTNERS
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_arena.utils.configclass import make_configclass
    for side, prim in (("left", "LeftArm"), ("right", "RightArm")):
        for tip in ("tip_left", "tip_right"):
            cfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/"+prim+embodiment.link_paths[tip],
                  filter_prim_paths_expr=["{ENV_REGEX_NS}/multitool/"+b for b in BODIES],
                  update_period=0., history_length=0)
            setattr(embodiment.scene_config, side+"_"+tip, cfg)
        setattr(embodiment.scene_config, side+"_all_contacts", ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/"+prim+"/.*", history_length=0))
    if PAYLOAD is not None:
        paths = ["{ENV_REGEX_NS}/multitool/"+b for b in BODIES]
        PAYLOAD_PARTNERS = list(BODIES)
        for side, prim in (("left", "LeftArm"), ("right", "RightArm")):
            for name, path in embodiment.link_paths.items():
                paths.append("{ENV_REGEX_NS}/"+prim+path)
                PAYLOAD_PARTNERS.append(side+":"+name)
        setattr(embodiment.scene_config, "payload_contacts", ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/"+PAYLOAD.name, filter_prim_paths_expr=paths,
            track_contact_points=True, max_contact_data_count_per_prim=128, history_length=0))
    fields = [(n, type(v), v) for n, v in vars(embodiment.scene_config).items() if not n.startswith("_")]
    embodiment.scene_config = make_configclass("FoldingMultitoolSceneCfg", fields)()


def measure(env, step):
    global last_record
    from yam_arena.urdf import quat_xyzw_to_matrix
    scene = env.unwrapped.scene
    art = scene["multitool"]
    q = dict(zip(art.joint_names, array(art.data.joint_pos)[0].tolist()))
    poses = dict(zip(art.body_names, array(art.data.body_link_pose_w)[0]))
    root = poses["jaw_a"]
    r = quat_xyzw_to_matrix(root[3:])
    pads = [np.linalg.norm(array(scene[n].data.normal_force_matrix_w)[0, 0], axis=-1).tolist() for n in PAD_NAMES]
    last_record = {"step": step, "q": q,
                   "qvel": dict(zip(art.joint_names, array(art.data.joint_vel)[0].tolist())),
                   "link_poses": {k: p.tolist() for k, p in poses.items()},
                   "root_pos": root[:3].tolist(), "tip_axis": r[:, 0].tolist(),
                   "gap_m": jaw_gap(load_spec(), q["jaw"]), "pads_n": pads}
    last_record["passive_updates"] = getattr(art, "passive_update_count", 0)
    last_record["passive_effort_nm"] = array(art.last_passive_effort)[0].tolist() if hasattr(art, "last_passive_effort") else None
    for side in ("left", "right"):
        sensor = scene[side+"_all_contacts"]
        last_record[side+"_contact_names"] = list(sensor.body_names)
        last_record[side+"_contact_forces_n"] = array(sensor.data.net_normal_forces_w)[0].tolist()
        robot = scene[side+"_robot"]
        last_record[side+"_joint_names"] = list(robot.joint_names)
        last_record[side+"_joint_positions"] = array(robot.data.joint_pos)[0].tolist()
        if MODE == "retrieval":
            last_record[side+"_body_poses"] = dict(zip(robot.body_names,
                array(robot.data.body_link_pose_w)[0].tolist()))
    if PAYLOAD is not None:
        obj = scene[PAYLOAD.name]
        pose = array(obj.data.root_link_pose_w)[0]
        rot = quat_xyzw_to_matrix(pose[3:])
        vertices = np.concatenate([p.vertices for p in PAYLOAD.pieces])@rot.T+pose[:3]
        sensor = scene["payload_contacts"]
        forces = array(sensor.data.normal_force_matrix_w)[0, 0]
        points = array(sensor.data.contact_pos_w)[0, 0]
        if len(forces) != len(PAYLOAD_PARTNERS):
            raise RuntimeError("Payload contact filters did not preserve the explicit partner list")
        last_record["payload"] = {"name": PAYLOAD.name, "pose": pose.tolist(),
            "velocity": array(obj.data.root_com_vel_w)[0].tolist(),
            "mass_kg": float(array(obj.data.body_mass).reshape(-1)[0]),
            "low": vertices.min(0).tolist(), "high": vertices.max(0).tolist(),
            "contact_partners": PAYLOAD_PARTNERS, "contact_forces_n": forces.tolist(),
            "contact_points_w": [[float(v) if np.isfinite(v) else None for v in p] for p in points]}
        net = array(sensor.data.net_normal_forces_w)[0, 0]
        last_record["payload"]["unattributed_normal_n"] = (net-forces.sum(0)).tolist()
        if MODE == "retrieval":
            from clearance import distance_lower_bound
            last_record["payload"]["tool_robot_clearance_m"] = distance_lower_bound(
                vertices.min(0), vertices.max(0), last_record["link_poses"],
                [last_record[s+"_body_poses"] for s in ("left", "right")])
    monitor.update(last_record)
    last_record["scoring"] = monitor.snapshot()
    return last_record


def reset(env):
    global monitor, _step
    monitor, _step = DeploymentScorer(), 0
    return measure(env, 0)
