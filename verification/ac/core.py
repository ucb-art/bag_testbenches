# -*- coding: utf-8 -*-

"""This module defines the AC testbench class."""

from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Sequence, Optional

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt

from bag.simulation.core import TestbenchManager

if TYPE_CHECKING:
    from bag.core import Testbench


class ACTB(TestbenchManager):
    """This class sets up a generic AC analysis testbench.
    """
    def __init__(self,
                 data_fname,  # type: str
                 tb_name,  # type: str
                 impl_lib,  # type: str
                 specs,  # type: Dict[str, Any]
                 sim_view_list,  # type: Sequence[Tuple[str, str]]
                 env_list,  # type: Sequence[str]
                 ):
        # type: (...) -> None
        TestbenchManager.__init__(self, data_fname, tb_name, impl_lib, specs,
                                  sim_view_list, env_list)

    def setup_testbench(self, tb):
        # type: (Testbench) -> None
        fstart = self.specs['fstart']
        fstop = self.specs['fstop']
        fndec = self.specs['fndec']
        sim_vars = self.specs['sim_vars']
        sim_vars_env = self.specs.get('sim_vars_env', None)
        sim_outputs = self.specs.get('sim_outputs', None)

        tb.set_parameter('fstart', fstart)
        tb.set_parameter('fstop', fstop)
        tb.set_parameter('fndec', fndec)

        for key, val in sim_vars.items():
            if isinstance(val, int) or isinstance(val, float):
                tb.set_parameter(key, val)
            else:
                tb.set_sweep_parameter(key, values=val)

        if sim_vars_env is not None:
            for key, val in sim_vars_env.items():
                tb.set_env_parameter(key, val)

        if sim_outputs is not None:
            for key, val in sim_outputs.items():
                tb.add_output(key, val)

    def get_outputs(self):
        # type: () -> List[str]
        """Returns a list of output names."""
        sim_outputs = self.specs.get('sim_outputs', None)
        if sim_outputs is None:
            return []
        return list(sim_outputs.keys())

    @classmethod
    def get_gain_and_w3db(cls, data, output_list, output_dict=None):
        # type: (Dict[str, Any], List[str], Optional[Dict[str, Any]]) -> Dict[str, Any]
        """Returns a dictionary of gain and 3db bandwidth information.

        Parameters
        ----------
        data : Dict[str, Any]
            the simulation data dictionary.
        output_list : Sequence[str]
            list of output names to compute gain/bandwidth for.
        output_dict : Optional[Dict[str, Any]]
            If not None, append to the given output dictionary instead.

        Returns
        -------
        output_dict : Dict[str, Any]
            A BAG data dictionary containing the gain/bandwidth information.
        """
        if output_dict is None:
            output_dict = {}
        swp_info = data['sweep_params']
        f_vec = data['freq']
        for out_name in output_list:
            out_arr = data[out_name]
            swp_params = swp_info[out_name]
            freq_idx = swp_params.index('freq')
            new_swp_params = [par for par in swp_params if par != 'freq']
            gain_arr, w3db_arr = cls._compute_gain_and_w3db(f_vec, np.abs(out_arr), freq_idx)
            cls.record_array(output_dict, data, gain_arr, 'gain_' + out_name, new_swp_params)
            cls.record_array(output_dict, data, w3db_arr, 'w3db_' + out_name, new_swp_params)

        return output_dict

    @classmethod
    def get_ugb_and_pm(cls, data, output_list, output_dict=None):
        # type: (Dict[str, Any], List[str], Optional[Dict[str, Any]]) -> Dict[str, Any]
        """Returns a dictionary of unity-gain bandwidth and phase margin information.

        Parameters
        ----------
        data : Dict[str, Any]
            the simulation data dictionary.
        output_list : Sequence[str]
            list of output names to compute specs for.
        output_dict : Optional[Dict[str, Any]]
            If not None, append to the given output dictionary instead.

        Returns
        -------
        output_dict : Dict[str, Any]
            A BAG data dictionary containing the gain/bandwidth information.
        """
        if output_dict is None:
            output_dict = {}
        swp_info = data['sweep_params']
        f_vec = data['freq']
        for out_name in output_list:
            out_arr = data[out_name]
            swp_params = swp_info[out_name]
            freq_idx = swp_params.index('freq')
            new_swp_params = [par for par in swp_params if par != 'freq']
            funity_arr, pm_arr = cls._compute_ugb_and_pm(f_vec, out_arr, freq_idx)
            cls.record_array(output_dict, data, funity_arr, 'funity_' + out_name, new_swp_params)
            cls.record_array(output_dict, data, pm_arr, 'pm_' + out_name, new_swp_params)

        return output_dict

    @classmethod
    def _compute_gain_and_w3db(cls, f_vec, out_arr, freq_idx):
        # type: (np.ndarray, np.ndarray, int) -> Tuple[np.ndarray, np.ndarray]
        """Compute the DC gain and bandwidth of the amplifier given output array.

        Parmeters
        ---------
        f_vec : np.ndarray
            the frequency vector.  Must be sorted.
        out_arr : np.ndarray
            the amplifier output transfer function.  Could be multidimensional.
        freq_idx : int
            frequency axis index.

        Returns
        -------
        gain_arr : np.ndarray
            the DC gain array.
        w3db_arr : np.ndarray
            the 3db bandwidth array.  Contains NAN if the transfer function never
            intersect the gain.
        """
        # move frequency axis to last axis
        out_arr = np.moveaxis(out_arr, freq_idx, -1)
        gain_arr = out_arr[..., 0]

        # convert
        orig_shape = out_arr.shape
        num_pts = orig_shape[-1]
        out_log = 20 * np.log10(out_arr.reshape(-1, num_pts))
        gain_log_3db = 20 * np.log10(gain_arr.reshape(-1)) - 3

        # find first index at which gain goes below gain_log 3db
        diff_arr = out_log - gain_log_3db[:, np.newaxis]
        idx_arr = np.argmax(diff_arr < 0, axis=1)
        freq_log = np.log10(f_vec)
        freq_log_max = freq_log[idx_arr]

        num_swp = out_log.shape[0]
        w3db_list = []
        for idx in range(num_swp):
            fun = interp.interp1d(freq_log, diff_arr[idx, :], kind='cubic', copy=False,
                                  assume_sorted=True)
            w3db_list.append(10.0**(cls._get_intersect(fun, freq_log[0], freq_log_max[idx])))

        return gain_arr, np.array(w3db_list).reshape(gain_arr.shape)

    @classmethod
    def _compute_ugb_and_pm(cls, f_vec, out_arr, freq_idx):
        # type: (np.ndarray, np.ndarray, int) -> Tuple[np.ndarray, np.ndarray]
        """Compute the unity gain frequency and phase margin of the frequency response.

        Parmeters
        ---------
        f_vec : np.ndarray
            the frequency vector.  Must be sorted.
        out_arr : np.ndarray
            the amplifier output transfer function.  Could be multidimensional.
        freq_idx : int
            frequency axis index.

        Returns
        -------
        funity_arr : np.ndarray
            the unity gain frequency array.
        pm_arr : np.ndarray
            the phase margin array, in degrees.
        """
        # move frequency axis to last axis
        out_arr = np.moveaxis(out_arr, freq_idx, -1)
        out_mag = np.abs(out_arr)
        out_phase = np.angle(out_arr, deg=True)

        # convert
        num_pts = out_arr.shape[-1]
        out_log = 20 * np.log10(out_mag.reshape(-1, num_pts))
        out_phase = out_phase.reshape(-1, num_pts)

        # find first index at which gain goes below 0dB
        idx_arr = np.argmax(out_log < 0, axis=1)
        freq_log = np.log10(f_vec)
        freq_log_max = freq_log[idx_arr]

        num_swp = out_log.shape[0]
        funity_list, pm_list = [], []
        for idx in range(num_swp):
            fun = interp.interp1d(freq_log, out_log[idx, :], kind='cubic', copy=False,
                                  assume_sorted=True)
            funity_log = cls._get_intersect(fun, freq_log[0], freq_log_max[idx])
            funity = 10.0 ** funity_log
            funity_list.append(funity)
            if funity != np.NAN:
                pfun = interp.interp1d(freq_log, out_phase[idx, :], kind='cubic', copy=False,
                                       assume_sorted=True)
                pm = 180 - (out_phase[idx, 0] - pfun(funity_log))
            else:
                pm = np.NAN
            pm_list.append(pm)

        funity_arr = np.array(funity_list).reshape(out_arr.shape[:-1])
        pm_arr = np.array(pm_list).reshape(out_arr.shape[:-1])
        return funity_arr, pm_arr

    @classmethod
    def _get_intersect(cls, fun, xmin, xmax):
        try:
            return sciopt.brentq(fun, xmin, xmax)
        except ValueError:
            return np.NAN
