# -*- coding: utf-8 -*-

"""This module contains various design methods/classes for amplifier components."""

from typing import TYPE_CHECKING, List, Tuple, Optional, Dict, Any

import numpy as np
import scipy.optimize as sciopt

from bag.util.search import BinaryIterator
from bag.math.dfun import DiffFunction

if TYPE_CHECKING:
    from verification.mos.query import MOSDBDiscrete


class LoadDiodePFB(object):
    """A differential load consists of diode transistor and negative gm cell.

    This topology is designed to have a large differential mode resistance and a
    small common mode resistance, plus a well defined output common mode

    Parameters
    ----------
    mos_db : MOSDBDiscrete
        the transistor small signal parameters database.
    """
    def __init__(self, mos_db):
        # type: (MOSDBDiscrete) -> None
        self._db = mos_db
        self._dsn_params = mos_db.dsn_params
        if 'stack' not in self._dsn_params:
            raise ValueError('This class assumes transistor stack is swept.')

        self._stack_list = sorted(mos_db.get_dsn_param_values('stack'))
        self._intent_list = mos_db.get_dsn_param_values('intent')
        self._valid_widths = mos_db.width_list
        self._best_op = None

    def design(self, itarg_list, vds2_list, ft_min, stack_list=None):
        # type: (List[float], List[float], float, Optional[List[int]]) -> None
        """Design the diode load.

        Parameters
        ----------
        itarg_list : List[float]
            target single-ended bias current across simulation environments.
        vds2_list : List[float]
            list of op-amp stage 2 vds voltage across simulation environments.
        ft_min : float
            minimum transit frequency of the composit transistor.
        stack_list : Optional[List[int]]
            list of valid stack numbers.
        """
        if stack_list is None:
            stack_list = self._stack_list

        vgs_idx = self._db.get_fun_arg_index('vgs')

        num_stack = len(stack_list)

        self._best_op = None
        best_score = None
        for intent in self._intent_list:
            for w in self._valid_widths:
                for idx1 in range(num_stack):
                    stack1 = stack_list[idx1]
                    self._db.set_dsn_params(w=w, intent=intent, stack=stack1)
                    ib1 = self._db.get_function_list('ibias')
                    gm1 = self._db.get_function_list('gm')
                    gds1 = self._db.get_function_list('gds')
                    cd1 = self._db.get_function_list('cdd')
                    vgs1_min, vgs1_max = ib1[0].get_input_range(vgs_idx)

                    for idx2 in range(idx1, num_stack):
                        stack2 = stack_list[idx2]
                        self._db.set_dsn_params(stack=stack2)
                        ib2 = self._db.get_function_list('ibias')
                        gm2 = self._db.get_function_list('gm')
                        gds2 = self._db.get_function_list('gds')
                        cd2 = self._db.get_function_list('cdd')
                        vgs2_min, vgs2_max = ib2[0].get_input_range(vgs_idx)

                        vgs_min = max(vgs1_min, vgs2_min)
                        vgs_max = min(vgs1_max, vgs2_max)

                        seg1_iter = BinaryIterator(2, None, step=2)
                        while seg1_iter.has_next():
                            seg1 = seg1_iter.get_next()

                            all_neg = True
                            one_pass = False
                            seg2_iter = BinaryIterator(0, None, step=2)
                            while seg2_iter.has_next():
                                seg2 = seg2_iter.get_next()

                                vgs_list, err_code = self._solve_vgs(itarg_list, seg1, seg2, ib1,
                                                                     ib2, vgs_min, vgs_max)
                                if err_code < 0:
                                    # too few fingers
                                    seg2_iter.up()
                                elif err_code > 0:
                                    # too many fingers
                                    seg2_iter.down()
                                else:
                                    one_pass = True
                                    cur_score = self._compute_score(ft_min, seg1, seg2, gm1, gm2,
                                                                    gds1, gds2, cd1, cd2, vgs_list)

                                    if cur_score != -1:
                                        all_neg = False

                                    if cur_score < 0:
                                        seg2_iter.down()
                                    else:
                                        seg2_iter.save()
                                        seg2_iter.up()
                                        if best_score is None or cur_score > best_score:
                                            best_score = cur_score
                                            self._best_op = (intent, stack1, stack2, w, seg1, seg2,
                                                             vgs_list, vds2_list)

                            if seg2_iter.get_last_save() is None:
                                # no solution for seg2
                                if all_neg and one_pass:
                                    # all solutions encountered have negative resistance,
                                    # this means we have insufficent number of diode fingers.
                                    seg1_iter.up()
                                elif not one_pass:
                                    # exit immediately with no solutions at all; too many fingers
                                    seg1_iter.down()
                                else:
                                    # all positive resistance solution break V*_min specs.
                                    # this means we have too many number of fingers.
                                    seg1_iter.down()
                            else:
                                seg1_iter.save()
                                seg1_iter.up()

    def _solve_vgs(self,
                   itarg_list,  # type: List[float]
                   k1,  # type: float
                   k2,  # type: float
                   ib1,  # type: List[DiffFunction]
                   ib2,  # type: List[DiffFunction]
                   vgs_min,  # type: float
                   vgs_max,  # type: float
                   ):
        # type: (...) -> Tuple[List[float], int]

        vgs_list = []
        for itarg, ifun1, ifun2 in zip(itarg_list, ib1, ib2):
            def zero_fun(vgs):
                fun_arg = self._db.get_fun_arg(vbs=0, vds=vgs, vgs=vgs)
                return ifun1(fun_arg) * k1 + ifun2(fun_arg) * k2 - itarg

            itest0 = zero_fun(vgs_min)
            itest1 = zero_fun(vgs_max)
            if itest0 < 0 and itest1 < 0:
                # too few fingers
                return [], -1
            elif itest0 > 0 and itest1 > 0:
                # too many fingers
                return [], 1
            else:
                vgs_cur = sciopt.brentq(zero_fun, vgs_min, vgs_max)
                vgs_list.append(vgs_cur)

        return vgs_list, 0

    def _compute_score(self, ft_min, scale1, scale2, gm1, gm2, gds1, gds2,
                       cd1, cd2, vgs_list):
        score = float('inf')
        for fgm1, fgm2, fgds1, fgds2, fcd1, fcd2, vgs in \
                zip(gm1, gm2, gds1, gds2, cd1, cd2, vgs_list):
            arg = self._db.get_fun_arg(vbs=0, vds=vgs, vgs=vgs)
            cur_gm1 = scale1 * fgm1(arg)
            cur_gm2 = scale2 * fgm2(arg)
            cur_gds1 = scale1 * fgds1(arg)
            cur_gds2 = scale2 * fgds2(arg)
            cur_cd = scale1 * fcd1(arg) + scale2 * fcd2(arg)
            cur_ft = (cur_gm1 + cur_gm1) / (2 * cur_cd) / 2 / np.pi
            cur_gds_tot = cur_gds1 + cur_gds2 + cur_gm1 - cur_gm2
            if cur_gm1 <= cur_gm2:
                # negative resistance
                return -1
            if cur_ft < ft_min:
                # break minimum V* spec
                return -2

            score = min(score, (cur_gm1 + cur_gm2) / cur_gds_tot)

        return score

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        if self._best_op is None:
            return None

        intent, stack1, stack2, w, seg1, seg2, vgs_list, vds2_list = self._best_op
        self._db.set_dsn_params(w=w, intent=intent, stack=stack1)
        ib1 = self._db.get_function_list('ibias')
        gm1 = self._db.get_function_list('gm')
        gds1 = self._db.get_function_list('gds')
        cg1 = self._db.get_function_list('cgg')
        cd1 = self._db.get_function_list('cdd')
        self._db.set_dsn_params(w=w, intent=intent, stack=stack2)
        ib2 = self._db.get_function_list('ibias')
        gm2 = self._db.get_function_list('gm')
        gds2 = self._db.get_function_list('gds')
        cg2 = self._db.get_function_list('cgg')
        cd2 = self._db.get_function_list('cdd')

        ctot1_list, gds1_list = [], []
        vstar2_list, gm2_list, gds2_list, cgg2_list, cdd2_list, ft_list = [], [], [], [], [], []
        for ib1f, gm1f, gds1f, cg1f, cd1f, ib2f, gm2f, gds2f, cg2f, cd2f, vgs, vds2 in \
                zip(ib1, gm1, gds1, cg1, cd1, ib2, gm2, gds2, cg2, cd2, vgs_list, vds2_list):
            arg1 = self._db.get_fun_arg(vbs=0, vds=vgs, vgs=vgs)
            cur_gm1 = seg1 * gm1f(arg1)
            cur_gm2 = seg2 * gm2f(arg1)
            cur_gds1 = seg1 * gds1f(arg1)
            cur_gds2 = seg2 * gds2f(arg1)
            cur_cg1 = seg1 * cg1f(arg1)
            cur_cg2 = seg2 * cg2f(arg1)
            cur_cd1 = seg1 * cd1f(arg1)
            cur_cd2 = seg2 * cd2f(arg1)
            cur_ctot = cur_cg1 + cur_cg2 + cur_cd1 + cur_cd2
            ctot1_list.append(cur_ctot)
            ft_list.append((cur_gm1 + cur_gm2) / (2 * (cur_cd1 + cur_cd2)) / 2 / np.pi)
            gds1_list.append(cur_gds1 + cur_gds2 + cur_gm1 - cur_gm2)

            arg2 = self._db.get_fun_arg(vbs=0, vds=vds2, vgs=vgs)
            cur_ib1 = seg1 * ib1f(arg2)
            cur_ib2 = seg2 * ib2f(arg2)
            cur_gm1 = seg1 * gm1f(arg2)
            cur_gm2 = seg2 * gm2f(arg2)
            cur_gds1 = seg1 * gds1f(arg2)
            cur_gds2 = seg2 * gds2f(arg2)
            cur_cg1 = seg1 * cg1f(arg2)
            cur_cg2 = seg2 * cg2f(arg2)
            cur_cd1 = seg1 * cd1f(arg2)
            cur_cd2 = seg2 * cd2f(arg2)
            vstar2_list.append(2 * (cur_ib1 + cur_ib2) / (cur_gm1 + cur_gm2))
            gm2_list.append(cur_gm1 + cur_gm2)
            gds2_list.append(cur_gds1 + cur_gds2)
            cgg2_list.append(cur_cg1 + cur_cg2)
            cdd2_list.append(cur_cd1 + cur_cd2)

        return dict(
            vgs=vgs_list,
            ft=ft_list,
            ctot1=ctot1_list,
            gds1=gds1_list,
            vstar2=vstar2_list,
            gm2=gm2_list,
            gds2=gds2_list,
            cgg2=cgg2_list,
            cdd2=cdd2_list,
            intent=intent,
            stack_diode=stack1,
            stack_ngm=stack2,
            w=w,
            seg_diode=seg1,
            seg_ngm=seg2,
        )


class InputGm(object):
    """A simple differential input gm stage.

    This class maximizes the gain given V* constraint.
    """

    def __init__(self, mos_db):
        # type: (MOSDBDiscrete) -> None
        self._db = mos_db
        self._dsn_params = mos_db.dsn_params
        if 'stack' not in self._dsn_params:
            raise ValueError('This class assumes transistor stack is swept.')

        self._stack_list = sorted(mos_db.get_dsn_param_values('stack'))
        self._intent_list = mos_db.get_dsn_param_values('intent')
        self._valid_widths = mos_db.width_list
        self._best_op = None

    def design(self,
               itarg_list,  # type: List[float]
               vg_list,  # type: List[float]
               vd_list,  # type: List[float]
               gds_load_list,  # type: List[float]
               vb,  # type: float
               vstar_min,  # type: float
               vds_tail_min,  # type: float
               seg_min=2,  # type: int
               stack_list=None,  # type: Optional[List[int]]
               ):
        # type: (...) -> None
        """Design the input gm stage.

        Parameters
        ----------
        itarg_list : List[float]
            target single-ended bias current across simulation environments.
        vg_list : List[float]
            gate voltage across simulation environments.
        vd_list : List[float]
            drain voltage across simulation environments.
        gds_load_list : List[float]
            load conductance across simulation environments.
        vb : float
            body bias voltage.
        vstar_min : float
            minimum V* of the diode.
        vds_tail_min : float
            minimum absolute vds voltage of tail device.
        seg_min : int
            minimum number of segments.
        stack_list : Optional[List[str]]
            If given, we will only consider these stack values.
        """
        vgs_idx = self._db.get_fun_arg_index('vgs')
        vds_idx = self._db.get_fun_arg_index('vds')

        if stack_list is None:
            stack_list = self._stack_list

        best_score = None
        self._best_op = None
        for intent in self._intent_list:
            for w in self._valid_widths:
                for stack in stack_list:
                    self._db.set_dsn_params(w=w, intent=intent, stack=stack)
                    ib = self._db.get_function_list('ibias')
                    gm = self._db.get_function_list('gm')
                    gds = self._db.get_function_list('gds')

                    # get valid vs range across simulation environments.
                    vgs_min, vgs_max = ib[0].get_input_range(vgs_idx)
                    vds_min, vds_max = ib[0].get_input_range(vds_idx)
                    vs_bnds = [(max(vg - vgs_max, vd - vds_max), min(vg - vgs_min, vd - vds_min))
                               for vg, vd in zip(vg_list, vd_list)]

                    iunit_list = self._solve_iunit_from_vstar(vstar_min, vb, vg_list, vd_list,
                                                              vs_bnds, ib, gm)
                    if iunit_list is not None:
                        tot_seg = min((itarg / iunit for itarg, iunit in
                                       zip(itarg_list, iunit_list)))
                        # now get actual numbers
                        num_seg = max(seg_min, int(tot_seg // 2) * 2)
                        vs_list, score = self._solve_vs(itarg_list, vg_list, vd_list, vs_bnds, vb,
                                                        num_seg, ib, gm, gds, gds_load_list,
                                                        vds_tail_min, vstar_min)
                        if score is not None and (best_score is None or score > best_score):
                            best_score = score
                            self._best_op = (intent, stack, w, num_seg, vg_list, vd_list,
                                             vs_list, vb)

    def _solve_vs(self, itarg_list, vg_list, vd_list, vs_bnds, vb, scale, ib, gm, gds,
                  gds_load_list, vds_tail_min, vstar_min):
        vs_list = []
        score = None
        for itarg, ibf, gmf, gdsf, vg, vd, gds_load, (vs_min, vs_max) in \
                zip(itarg_list, ib, gm, gds, vg_list, vd_list, gds_load_list, vs_bnds):

            def zero_fun(vs):
                arg = self._db.get_fun_arg(vbs=vb - vs, vds=vd - vs, vgs=vg - vs)
                return scale * ibf(arg) - itarg

            v1 = zero_fun(vs_min)
            v2 = zero_fun(vs_max)
            if v1 < 0 and v2 < 0 or v1 > 0 and v2 > 0:
                # no solution
                return None, None

            vs_cur = sciopt.brentq(zero_fun, vs_min, vs_max)  # type: float
            if abs(vs_cur - vb) < vds_tail_min:
                return None, None
            cur_arg = self._db.get_fun_arg(vbs=vb - vs_cur, vds=vd - vs_cur, vgs=vg - vs_cur)
            gm_cur = gmf(cur_arg) * scale
            ib_cur = ibf(cur_arg) * scale
            if 2 * ib_cur / gm_cur < vstar_min:
                # check V* spec again
                return None, None
            gds_cur = gdsf(cur_arg) * scale
            score_cur = gm_cur / (gds_cur + gds_load)
            if score is None:
                score = score_cur
            else:
                score = min(score, score_cur)

            vs_list.append(vs_cur)

        return vs_list, score

    def _solve_iunit_from_vstar(self, vstar_min, vb, vg_list, vd_list, vs_bnds, ib, gm):
        iunit_list = []
        for ibf, gmf, vg, vd, (vs_min, vs_max) in zip(ib, gm, vg_list, vd_list, vs_bnds):

            def zero_fun(vs):
                arg = self._db.get_fun_arg(vbs=vb - vs, vds=vd - vs, vgs=vg - vs)
                return 2 * ibf(arg) / gmf(arg) - vstar_min

            v1 = zero_fun(vs_min)
            v2 = zero_fun(vs_max)
            if v1 < 0 and v2 < 0:
                # cannot meet vstar_min spec
                return None
            elif v1 > 0 and v2 > 0:
                # NOTE: for very small V*, it may be the case that V* versus vs is not monotonic.
                vs_sol = vs_min if v1 < v2 else vs_max
            else:
                vs_sol = sciopt.brentq(zero_fun, vs_min, vs_max)

            cur_arg = self._db.get_fun_arg(vbs=vb - vs_sol, vds=vd - vs_sol, vgs=vg - vs_sol)
            iunit_list.append(ibf(cur_arg))

        return iunit_list

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        if self._best_op is None:
            return None

        intent, stack, w, seg, vg_list, vd_list, vs_list, vb = self._best_op

        self._db.set_dsn_params(w=w, intent=intent, stack=stack)
        ib = self._db.get_function_list('ibias')
        gm = self._db.get_function_list('gm')
        gds = self._db.get_function_list('gds')
        cgg = self._db.get_function_list('cgg')
        cdd = self._db.get_function_list('cdd')

        vstar_list, gm_list, gds_list, cgg_list, cdd_list = [], [], [], [], []
        for ibf, gmf, gdsf, cggf, cddf, vg, vd, vs in \
                zip(ib, gm, gds, cgg, cdd, vg_list, vd_list, vs_list):
            arg = self._db.get_fun_arg(vbs=vb - vs, vds=vd - vs, vgs=vg - vs)
            cur_ib = seg * ibf(arg)
            cur_gm = seg * gmf(arg)
            cur_gds = seg * gdsf(arg)
            cur_cgg = seg * cggf(arg)
            cur_cdd = seg * cddf(arg)
            vstar_list.append(2 * cur_ib / cur_gm)
            gds_list.append(cur_gds)
            cgg_list.append(cur_cgg)
            cdd_list.append(cur_cdd)
            gm_list.append(cur_gm)

        return dict(
            vstar=vstar_list,
            gm=gm_list,
            gds=gds_list,
            cgg=cgg_list,
            cdd=cdd_list,
            intent=intent,
            stack=stack,
            w=w,
            seg=seg,
            vs=vs_list,
        )
