# -*- coding: utf-8 -*-


from bag.io import read_yaml

from ckt_dsn.mos.core import MOSDBDiscrete
from ckt_dsn.analog.amplifier.components import InputGm


def print_dsn_info(info):
    if info is None:
        print('No solution found')
    else:
        for key, val in info.items():
            if isinstance(val, list):
                print('%s = [%s]' % (key, ', '.join(('%.3g' % v for v in val))))
            elif isinstance(val, str):
                print('%s = %s' % (key, val))
            else:
                print('%s = %.3g' % (key, val))


if __name__ == '__main__':
    pch_config = 'specs_mos_char/mos_char_pch_stack_w2.yaml'
    gm_specs = 'specs_dsn/input_gm.yaml'

    noise_fstart = 20e3
    noise_fstop = noise_fstart + 500
    noise_scale = 1.0
    noise_temp = 310

    gm_specs = read_yaml(gm_specs)

    print('create transistor database')
    pch_db = MOSDBDiscrete([pch_config], noise_fstart, noise_fstop,
                           noise_scale=noise_scale, noise_temp=noise_temp)
    print('create design class')
    gm_dsn = InputGm(pch_db)

    print('design gm')
    gm_dsn.design(**gm_specs)
    gm_info = gm_dsn.get_dsn_info()
    print('gm info:')
    print_dsn_info(gm_info)

    print('done')
