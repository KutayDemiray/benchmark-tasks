"""Explicit control and physics clocks shared by recording and scoring."""


def control_hz(cfg):
    value = cfg.get('completion', {}).get('control_hz', 10)
    assert value in (10, 20), 'Unsupported control rate'
    return int(value)


def physics_stride(cfg):
    hz = control_hz(cfg)
    physics = cfg['simulation']['physics_hz']
    assert physics % hz == 0, 'Non-integral physics decimation'
    return int(physics // hz)


def max_actions(cfg):
    return round(cfg.get('completion', {}).get('run_limit_seconds', 300)*control_hz(cfg))


def retention_samples(cfg):
    return round(cfg['tabletop']['released_observation_seconds']*control_hz(cfg))+1
