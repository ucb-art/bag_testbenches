# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as sciopt
import scipy.signal as scisig

import matplotlib.pyplot as plt

from bag.util.search import FloatBinaryIterator
from bag.data.lti import LTICircuit, get_stability_margins, get_w_crossings

from verification.mos.query import MOSDBDiscrete


def get_db(nch_dir, pch_dir, intent='standard', interp_method='spline', sim_env='tt'):
    env_list = [sim_env]

    nch_db = MOSDBDiscrete([nch_dir], interp_method=interp_method)
    pch_db = MOSDBDiscrete([pch_dir], interp_method=interp_method)

    nch_db.env_list = pch_db.env_list = env_list
    nch_db.set_dsn_params(intent=intent)
    pch_db.set_dsn_params(intent=intent)

    return nch_db, pch_db


def plot_tf(fvec, tf_list, lbl_list):
    wvec = 2 * np.pi * fvec
    plt.figure(1)
    plt.plot(fvec, [0] * len(fvec), '--k')
    for (num, den), lbl in zip(tf_list, lbl_list):
        _, mag, phase = scisig.bode((num, den), w=wvec)
        poles = np.sort(np.abs(np.poly1d(den).roots) / (2 * np.pi))
        print(poles)
        poles = poles[:2]
        mag_poles = np.interp(poles, fvec, mag)

        p = plt.semilogx(fvec, mag, label=lbl)
        color = p[0].get_color()
        plt.plot(poles, mag_poles, linestyle='', color=color, marker='o')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.show()


def opt_cfb(phase_margin, cir, cmin, cmax, cstep, ctol):
    bin_iter = FloatBinaryIterator(cmin, None, ctol, search_step=cstep)
    while bin_iter.has_next():
        cur_cf = bin_iter.get_next()
        cir.add_cap(cur_cf, 'x', 'out')
        num, den = cir.get_num_den('in', 'out')
        cur_pm, _ = get_stability_margins(num, den)
        if cur_pm < phase_margin:
            if cur_cf > cmax:
                # no way to make amplifier stable, just return
                return None
            bin_iter.up()
        else:
            bin_iter.save()
            bin_iter.down()
        cir.add_cap(-cur_cf, 'x', 'out')

    return bin_iter.get_last_save()


def tf_vs_cfb(op_in, op_load, op_tail, cload, fg=2):
    fvec = np.logspace(6, 11, 1000)
    cvec = np.logspace(np.log10(5e-16), np.log10(5e-14), 5).tolist()

    scale_load = op_in['ibias'] / op_load['ibias'] * fg

    cir = LTICircuit()
    cir.add_transistor(op_in, 'mid', 'in', 'gnd', 'gnd', fg=fg)
    cir.add_transistor(op_load, 'mid', 'gnd', 'gnd', 'gnd', fg=scale_load)
    cir.add_transistor(op_load, 'out', 'mid', 'gnd', 'gnd', fg=scale_load)
    cir.add_transistor(op_tail, 'out', 'gnd', 'gnd', 'gnd', fg=fg)
    cir.add_cap(cload, 'out', 'gnd')

    gfb = op_load['gm'] * scale_load
    cir.add_conductance(gfb, 'mid', 'x')

    print('fg_in = %d, fg_load=%.3g, rfb = %.4g' % (fg, scale_load, 1/gfb))

    tf_list, lbl_list = [], []
    for cval in cvec:
        cir.add_cap(cval, 'x', 'out')
        tf_list.append(cir.get_num_den('in', 'out'))
        cir.add_cap(-cval, 'x', 'out')
        lbl_list.append('$C_{f} = %.3g$f' % (cval * 1e15))

    plot_tf(fvec, tf_list, lbl_list)


def funity_vs_scale2(op_in, op_load, op_tail, cload, phase_margin, fg=2):
    s2min = 1
    s2max = 40
    num_s = 100
    cmin = 1e-16
    cmax = 1e-9
    ctol = 1e-17
    cstep = 1e-15

    scale_load = op_in['ibias'] / op_load['ibias'] * fg
    gfb = op_load['gm'] * scale_load
    s2vec = np.linspace(s2min, s2max, num_s).tolist()
    f0_list, pm0_list, f1_list, pm1_list, copt_list = [], [], [], [], []
    for s2 in s2vec:
        cir = LTICircuit()
        cir.add_transistor(op_in, 'mid', 'in', 'gnd', 'gnd', fg=fg)
        cir.add_transistor(op_load, 'mid', 'gnd', 'gnd', 'gnd', fg=scale_load)
        cir.add_transistor(op_load, 'out', 'mid', 'gnd', 'gnd', fg=scale_load * s2)
        cir.add_transistor(op_tail, 'out', 'gnd', 'gnd', 'gnd', fg=fg * s2)
        cir.add_cap(cload, 'out', 'gnd')

        num, den = cir.get_num_den('in', 'out')
        f0_list.append(get_w_crossings(num, den)[0] / (2 * np.pi))
        pm0_list.append(get_stability_margins(num, den)[0])

        cir.add_conductance(gfb * s2, 'mid', 'x')
        copt = opt_cfb(phase_margin, cir, cmin, cmax, cstep, ctol)
        if copt is None:
            raise ValueError('oops, Cfb is None')
        cir.add_cap(copt, 'x', 'out')
        num, den = cir.get_num_den('in', 'out')

        f1_list.append(get_w_crossings(num, den)[0] / (2 * np.pi))
        pm1_list.append(get_stability_margins(num, den)[0])
        copt_list.append(copt)

    f, (ax0, ax1, ax2) = plt.subplots(3, sharex='all')
    ax0.plot(s2vec, np.array(copt_list) * 1e15)
    ax0.set_ylabel('Cf (fF)')
    ax1.plot(s2vec, pm1_list, label='Cf')
    ax1.plot(s2vec, pm0_list, label='no Cf')
    ax1.legend()
    ax1.set_ylabel('$\phi_{PM}$ (deg)')
    ax2.plot(s2vec, np.array(f1_list) * 1e-9, label='Cf')
    ax2.plot(s2vec, np.array(f0_list) * 1e-9, label='no Cf')
    ax2.legend()
    ax2.set_ylabel('$f_{UG}$ (GHz)')
    ax2.set_xlabel('$I_2/I_1$')
    plt.show()


def run_main():
    nch_dir = 'data/nch_w4'
    pch_dir = 'data/pch_w4'
    intent = 'ulvt'

    vtail = 0.15
    vdd = 0.9
    vmid = vdd / 2

    cload = 10e-15
    phase_margin = 45

    nch_db, pch_db = get_db(nch_dir, pch_dir, intent=intent)

    op_in = nch_db.query(vbs=-vtail, vds=vmid-vtail, vgs=vmid-vtail)
    op_load = pch_db.query(vbs=0, vds=vmid-vdd, vgs=vmid-vdd)
    in_ibias = op_in['ibias']

    ibias_fun = nch_db.get_function('ibias')

    def fun_zero(vg):
        arg = nch_db.get_fun_arg(vgs=vg, vds=vtail, vbs=0)
        return (ibias_fun(arg) - in_ibias) * 1e6

    vbias = sciopt.brentq(fun_zero, 0, vdd)
    # noinspection PyTypeChecker
    op_tail = nch_db.query(vbs=0, vds=vtail, vgs=vbias)

    # tf_vs_cfb(op_in, op_load, op_tail, cload)
    funity_vs_scale2(op_in, op_load, op_tail, cload, phase_margin, fg=2)


if __name__ == '__main__':
    run_main()
