"""The existing force-limited laboratory press and physical pedal trajectory."""


def targets(t, request, origin_z):
    """Return laboratory drive targets and phase, without changing passive joints."""
    travel=origin_z-.003+request.get('press_overtravel',0.)
    target=0.;phase='folded'
    if 1<=t<4:
        s=(t-1)/3;target=-travel*s*s*(3-2*s);phase='press'
    elif 4<=t<6:
        target=-travel;phase='flat_hold'
    elif 6<=t<9:
        s=(t-6)/3
        target=-travel+(travel+request.get('retention_raise',0.))*s*s*(3-2*s);phase='lift'
    elif t>=9:
        target=request.get('retention_raise',0.);phase='retention'
    pusher=0.;u=t-request.get('retention_extension',0.)
    if request.get('mechanism_unlock') and u>=12:
        floor_travel=origin_z-.003
        if u<15:
            s=(u-12)/3;target=-floor_travel*s*s*(3-2*s);phase='set_down_for_unlock'
        elif u<20:
            target=-floor_travel+request.get('unlock_lift',0.);phase='pedal_press'
            if u<17:
                s=(u-15)/2
                pusher=-(.180+request.get('pedal_overtravel',0.))*s*s*(3-2*s)
                target=-floor_travel+request.get('unlock_lift',0.)*s*s*(3-2*s)
            elif u<18:
                pusher=-(.180+request.get('pedal_overtravel',0.))
            elif u<19:
                s=u-18;pusher=-(.180+request.get('pedal_overtravel',0.))*(1-s*s*(3-2*s))
            else:
                phase='pedal_released'
        elif u<23:
            s=(u-20)/3
            target=(-floor_travel+request.get('unlock_lift',0.))*(1-s*s*(3-2*s));phase='lift_after_unlock'
        else:
            phase='unlocked_hold'
    return target,pusher,phase
