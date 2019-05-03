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

    w3db_min = 2 * np.pi * bw_min

    fun_ibiasn = nch_db.get_function('ibias', env=sim_env)
    fun_gmn = nch_db.get_function('gm', env=sim_env)
    fun_gdsn = nch_db.get_function('gds', env=sim_env)
    fun_cdn = nch_db.get_function('cdb', env=sim_env) + nch_db.get_function('cds', env=sim_env)

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

        # get nmos SS parameters
        narg = nch_db.get_fun_arg(vgs=vgsn_cur, vds=vgsn_cur, vbs=vtail)
        ibiasn_unit = fun_ibiasn(narg)
        gmn_unit = fun_gmn(narg)
        gdsn_unit = fun_gdsn(narg)
        cdn_unit = fun_cdn(narg)

        # find vgsp
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

        # get pmos SS parameters
        parg = pch_db.get_fun_arg(vgs=opt_vgsp, vds=vout - vdd, vbs=0)
        ibiasp_unit = fun_ibiasp(parg)
        kp = ibiasn_unit / ibiasp_unit
        gdsp_unit = fun_gdsp(parg) * kp
        cdp_unit = fun_cdp(parg) * kp

        bw_intrinsic = (gdsp_unit + gdsn_unit) / (2 * np.pi * (cdp_unit + cdn_unit))
        # skip if we can never meet bandwidth requirement.
        if bw_intrinsic < bw_min:
            continue

        # compute total scale factor and number of input/load fingers
        bw_cur = 0
        seg_load = 0
        vbp = 0
        while bw_cur < bw_min:
            k = w3db_min * cload / (gdsp_unit + gdsn_unit - w3db_min * (cdn_unit + cdp_unit))

            seg_in = int(math.ceil(k / 2)) * 2
            seg_load = max(2, int(math.ceil(kp * k / 2)) * 2)
            # update kp and pmos SS parameters
            vbp, _ = find_load_bias(pch_db, vdd, vout, vgsp_min, vgsp_max, seg_in * ibiasn_unit, seg_load, fun_ibiasp)
            while vbp is None:
                seg_load += 2
                # update kp and pmos SS parameters
                vbp, _ = find_load_bias(pch_db, vdd, vout, vgsp_min, vgsp_max, seg_in * ibiasn_unit, seg_load, fun_ibiasp)
            kp = seg_load / seg_in

            parg = pch_db.get_fun_arg(vgs=vbp - vdd, vds=vout - vdd, vbs=0)
            gdsp_unit = fun_gdsp(parg) * kp
            cdp_unit = fun_cdp(parg) * kp

            # recompute gain/bandwidth
            bw_cur = (gdsp_unit + gdsn_unit) * seg_in / (2 * np.pi * (seg_in * (cdp_unit + cdn_unit) + cload))

        gain_cur = gmn_unit / (gdsp_unit + gdsn_unit)
        ibias_cur = seg_in * ibiasn_unit

        if performance is None or performance[0] > ibias_cur:
            performance = (ibias_cur, gain_cur, bw_cur, seg_in, seg_load, vgsn_cur, vbp)

    if performance is None:
        return None

    ibias_opt, gain_cur, bw_cur, seg_in, seg_load, vgs_in, vload = performance
    vio = vtail + vgs_in
    seg_tail, vbias = find_tail_bias(fun_ibiasn, nch_db, vtail, vgsn_min, vgsn_max, seg_in, ibias_opt)

    return dict(
        ibias=2 * ibias_opt,
        gain=gain_cur,
        bw=bw_cur,
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
    amp_dsn_specs_fname = 'specs_design/diffamp_simple.yaml'
    amp_char_specs_fname = 'specs_char/diffamp_simple.yaml'
    amp_char_specs_out_fname = 'specs_char/diffamp_simple_mod.yaml'

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
