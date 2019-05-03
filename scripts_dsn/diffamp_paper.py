# -*- coding: utf-8 -*-

"""This script designs a simple diff amp with gain/bandwidth spec for BAG CICC paper."""

import math
import pprint

import yaml
import numpy as np
import scipy.optimize as sciopt

from bag.core import BagProject
from bag.io import read_yaml, open_file
from bag.io.sim_data import load_sim_file
from bag.util.search import BinaryIterator, minimize_cost_golden_float
from bag.simulation.core import DesignManager

from verification.mos.query import MOSDBDiscrete


def design_amp(amp_specs, nch_db, pch_db):
    sim_env = amp_specs['sim_env']
    vdd = amp_specs['vdd']
    vtail = amp_specs['vtail']
    vgs_res = amp_specs['vgs_res']
    gain_min = amp_specs['gain_min']
    bw_min = amp_specs['bw_min']
    cload = amp_specs['cload']

    fun_ibiasn = nch_db.get_function('ibias', env=sim_env)
    fun_gmn = nch_db.get_function('gm', env=sim_env)
    fun_gdsn = nch_db.get_function('gds', env=sim_env)
    fun_cdn = nch_db.get_function('cdb', env=sim_env) + nch_db.get_function('cds', env=sim_env)
    fun_cgsn = nch_db.get_function('cgs', env=sim_env)

    fun_ibiasp = pch_db.get_function('ibias', env=sim_env)
    fun_gdsp = pch_db.get_function('gds', env=sim_env)
    fun_cdp = pch_db.get_function('cdd', env=sim_env)

    vgsn_idx = nch_db.get_fun_arg_index('vgs')
    vgsn_min, vgsn_max = fun_ibiasn.get_input_range(vgsn_idx)
    num_pts = int(math.ceil((vgsn_max - vgsn_min) / vgs_res))
    vgs_list = np.linspace(vgsn_min, vgsn_max, num_pts + 1).tolist()

    vgsp_idx = pch_db.get_fun_arg_index('vgs')
    vgsp_min, vgsp_max = fun_ibiasp.get_input_range(vgsp_idx)
    # sweep vgs, find best point
    performance = None
    for vgsn_cur in vgs_list:
        vout = vgsn_cur + vtail

        narg = nch_db.get_fun_arg(vgs=vgsn_cur, vds=vgsn_cur, vbs=vtail)
        ibiasn_unit = fun_ibiasn(narg)
        gmn_unit = fun_gmn(narg)
        gdsn_unit = fun_gdsn(narg)
        cdn_unit = fun_cdn(narg)
        cgsn_unit = fun_cgsn(narg)

        # find max gain
        def gain_fun1(vgsp_test):
            parg_test = pch_db.get_fun_arg(vgs=vgsp_test, vds=vout - vdd, vbs=0)
            ibiasp_unit_test = fun_ibiasp(parg_test)
            gdsp_unit_test = fun_gdsp(parg_test)
            return gmn_unit / ibiasn_unit / (gdsn_unit / ibiasn_unit + gdsp_unit_test / ibiasp_unit_test)

        result = minimize_cost_golden_float(gain_fun1, gain_min, vgsp_min, vgsp_max, tol=vgs_res / 10)
        opt_vgsp = result.x
        if opt_vgsp is None:
            print('vgsn = %.4g, max gain: %.4g' % (vgsn_cur, result.vmax))
            break

        # get number of input fingers needed to achieve gain_max with minimum number of load fingers
        seg_in_init = fun_ibiasp(pch_db.get_fun_arg(vgs=opt_vgsp, vds=vout - vdd, vbs=0)) * 2 / ibiasn_unit
        seg_in_init = int(round(seg_in_init / 2)) * 2
        # sweep gm size
        seg_in_iter = BinaryIterator(2, None, step=2)
        seg_in_iter.set_current(seg_in_init)
        while seg_in_iter.has_next():
            seg_in = seg_in_iter.get_next()
            ibiasn = seg_in * ibiasn_unit
            gmn = seg_in * gmn_unit
            gdsn = seg_in * gdsn_unit

            # sweep load size
            seg_load_iter = BinaryIterator(2, None, step=2)
            while seg_load_iter.has_next():
                seg_load = seg_load_iter.get_next()
                vbp, sgn = find_load_bias(pch_db, vdd, vout, vgsp_min, vgsp_max, ibiasn, seg_load, fun_ibiasp)

                if vbp is None:
                    if sgn > 0:
                        seg_load_iter.up()
                    else:
                        seg_load_iter.down()
                else:
                    parg = pch_db.get_fun_arg(vgs=vbp - vdd, vds=vout - vdd, vbs=0)
                    gdsp = seg_load * fun_gdsp(parg)
                    if gmn / (gdsp + gdsn) >= gain_min:
                        seg_load_iter.save_info((vbp, parg))
                        seg_load_iter.down()
                    else:
                        seg_load_iter.up()

            seg_load = seg_load_iter.get_last_save()
            if seg_load is None:
                # no load solution -> cannot meet gain spec.
                break

            vbp, parg = seg_load_iter.get_last_save_info()
            gdsp = seg_load * fun_gdsp(parg)
            cdp = seg_load * fun_cdp(parg)

            cdn = seg_in * cdn_unit
            cgsn = seg_in * cgsn_unit

            ro_cur = 1 / (gdsp + gdsn)
            gain_cur = gmn * ro_cur
            cpar_cur = cdn + cdp + (1 + 1 / gain_cur) * cgsn

            # check intrinsic bandwidth good
            if 1 / (ro_cur * cpar_cur * 2 * np.pi) < bw_min:
                break

            cload_cur = cload + cpar_cur
            bw_cur = 1 / (ro_cur * cload_cur * 2 * np.pi)
            if bw_cur < bw_min:
                seg_in_iter.up()
            else:
                seg_in_iter.save_info((seg_load, vbp, ibiasn, gain_cur, bw_cur))
                seg_in_iter.down()

        if seg_in_iter.get_last_save() is None:
            continue

        seg_in = seg_in_iter.get_last_save()
        seg_load, vbp, ibiasn, gain_cur, bw_cur = seg_in_iter.get_last_save_info()
        if performance is None or performance[0] > ibiasn:
            performance = (ibiasn, gain_cur, bw_cur, seg_in, seg_load, vgsn_cur, vbp)

    if performance is None:
        return None

    ibias_opt, gain_opt, bw_opt, seg_in, seg_load, vgs_in, vload = performance
    vio = vtail + vgs_in
    seg_tail, vbias = find_tail_bias(fun_ibiasn, nch_db, vtail, vgsn_min, vgsn_max, seg_in, ibias_opt)

    return dict(
        ibias=2 * ibias_opt,
        gain=gain_opt,
        bw=bw_opt,
        seg_in=seg_in,
        seg_load=seg_load,
        seg_tail=seg_tail,
        vtail=vbias,
        vindc=vio,
        voutdc=vio,
        vload=vload,
        vgs_in=vgs_in,
    )


def find_tail_bias(fun_ibiasn, nch_db, vtail, vgs_min, vgs_max, seg_tail_min, itarg):
    seg_tail_iter = BinaryIterator(seg_tail_min, None, step=2)
    while seg_tail_iter.has_next():
        seg_tail = seg_tail_iter.get_next()

        def fun_zero(vgs):
            narg = nch_db.get_fun_arg(vgs=vgs, vds=vtail, vbs=0)
            return fun_ibiasn(narg) * seg_tail - itarg

        if fun_zero(vgs_min) > 0:
            # smallest possible current > itarg
            seg_tail_iter.down()
        if fun_zero(vgs_max) < 0:
            # largest possible current < itarg
            seg_tail_iter.up()
        else:
            vbias = sciopt.brentq(fun_zero, vgs_min, vgs_max)  # type: float
            seg_tail_iter.save_info(vbias)
            seg_tail_iter.down()

    seg_tail = seg_tail_iter.get_last_save()
    vbias = seg_tail_iter.get_last_save_info()

    return seg_tail, vbias


def find_load_bias(pch_db, vdd, vout, vgsp_min, vgsp_max, itarg, seg_load, fun_ibiasp):
    def fun_zero(vbias):
        parg = pch_db.get_fun_arg(vgs=vbias - vdd, vds=vout - vdd, vbs=0)
        return fun_ibiasp(parg) * seg_load - itarg

    vbias_min = vdd + vgsp_min
    vbias_max = vdd + vgsp_max

    if fun_zero(vbias_max) > 0:
        # smallest possible current > itarg
        return None, -1
    if fun_zero(vbias_min) < 0:
        # largest possible current < itarg
        return None, 1

    vbias_opt = sciopt.brentq(fun_zero, vbias_min, vbias_max)  # type: float
    return vbias_opt, 0


def design(amp_dsn_specs, amp_char_specs_fname, amp_char_specs_out_fname):
    nch_config = amp_dsn_specs['nch_config']
    pch_config = amp_dsn_specs['pch_config']

    print('create transistor database')
    nch_db = MOSDBDiscrete([nch_config])
    pch_db = MOSDBDiscrete([pch_config])

    nch_db.set_dsn_params(**amp_dsn_specs['nch'])
    pch_db.set_dsn_params(**amp_dsn_specs['pch'])

    result = design_amp(amp_dsn_specs, nch_db, pch_db)
    if result is None:
        raise ValueError('No solution.')

    pprint.pprint(result)

    # update characterization spec file
    amp_char_specs = read_yaml(amp_char_specs_fname)
    # update bias
    var_dict = amp_char_specs['measurements'][0]['testbenches']['ac']['sim_vars']
    for key in ('vtail', 'vindc', 'voutdc'):
        var_dict[key] = result[key]
    for key in ('vdd', 'cload'):
        var_dict[key] = amp_dsn_specs[key]
    # update segments
    seg_dict = amp_char_specs['layout_params']['seg_dict']
    for key in ('in', 'load', 'tail'):
        seg_dict[key] = result['seg_' + key]

    with open_file(amp_char_specs_out_fname, 'w') as f:
        yaml.dump(amp_char_specs, f)

    return result


def simulate(prj, specs_fname):
    # simulate and report result
    sim = DesignManager(prj, specs_fname)
    sim.characterize_designs(generate=True, measure=True, load_from_file=False)
    # sim.test_layout(gen_sch=False)

    dsn_name = list(sim.get_dsn_name_iter())[0]
    summary = sim.get_result(dsn_name)
    fname = summary['ac']['gain_w3db_file']
    result = load_sim_file(fname)
    gain = result['gain_vout']
    w3db = result['w3db_vout']
    print('%s gain = %.4g' % (dsn_name, gain))
    print('%s w3db = %.4g' % (dsn_name, w3db))

    return gain, w3db


def run_main(prj):
    amp_dsn_specs_fname = 'specs_design/diffamp_paper.yaml'
    amp_char_specs_fname = 'specs_char/diffamp_paper.yaml'
    amp_char_specs_out_fname = 'specs_char/diffamp_paper_mod.yaml'

    # simulate(prj, amp_char_specs_out_fname)
    # return

    amp_dsn_specs = read_yaml(amp_dsn_specs_fname)
    gain_min_orig = amp_dsn_specs['gain_min']
    bw_min_orig = amp_dsn_specs['bw_min']

    result = None
    done = False
    gain, w3db = 0, 0
    while not done:
        result = design(amp_dsn_specs, amp_char_specs_fname, amp_char_specs_out_fname)
        gain, w3db = simulate(prj, amp_char_specs_out_fname)

        if gain >= gain_min_orig and w3db >= bw_min_orig:
            done = True
        else:
            if gain < gain_min_orig:
                gain_expected = result['gain']
                gain_scale = gain / gain_expected
                amp_dsn_specs['gain_min'] = gain_min_orig / gain_scale
            if w3db < bw_min_orig:
                bw_expected = result['bw']
                bw_scale = w3db / bw_expected
                amp_dsn_specs['bw_min'] = bw_min_orig / bw_scale

    pprint.pprint(result)
    print('final gain = %.4g' % gain)
    print('final w3db = %.4g' % w3db)


if __name__ == '__main__':
    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    run_main(bprj)
