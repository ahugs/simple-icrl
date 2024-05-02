def wall_behind(pos, obs, acs, info):
    return (info['x_position'] < pos)

def wall_infront(pos, obs, acs, info):
    return (info['x_position'] > pos)

def zero(obs, acs, info):
    return 0