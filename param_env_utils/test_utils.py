
def get_empty_env_ranges():
    return {'roughness':None,
          'stump_height':None,#stump_levels = [[0., 0.66], [0.66, 1.33], [1.33, 2.]]
          'tunnel_height':None,
          'obstacle_spacing':None,
          'gap_width':None,
          'step_height':None,
          'step_number':None}


def get_test_set_name(env_ranges):
    name = ''
    for k, v in env_ranges.items():
        if (v is not None) and (k is not 'env_param_input'):
            if k is "stump_height": # always same test set for stump height experiments
                name += k + str(v[0]) + "_3.0"
            else:
                name += k + str(v[0]) + "_" + str(v[1])
    return name