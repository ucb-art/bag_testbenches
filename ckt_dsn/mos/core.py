# -*- coding: utf-8 -*-

"""This module contains essential classes/methods for transistor characterization."""

import os
import math
from typing import List, Any, Tuple, Dict, Optional, Union, Sequence

import yaml
import numpy as np
import scipy.constants
import scipy.interpolate

from bag.io import read_yaml, open_file
from bag.core import Testbench, BagProject, create_tech_info
from bag.data import Waveform
from bag.data.mos import mos_y_to_ss
from bag.simulation.core import SimulationManager
from bag.math.dfun import VectorDiffFunction, DiffFunction
from bag.math.interpolate import LinearInterpolator, interpolate_grid


class MOSCharSS(SimulationManager):
    """A class that handles transistor small-signal parameters characterization.

    This class characterizes transistor given a range of current-per-finger
    specifications.  It uses AnalogBase to draw the transistor layout, and
    currently it sweeps vgs/vds (not vbs yet) and also characterizes noise
    for a given range of frequency.

    In addition to entries required by SimulationManager, The YAML specification
    file must have the following entries:

    root_dir :
        directory to save simulation files.
    vgs_file :
        file to save vgs sweep information.
        Given current-per-finger spec, this class will figure out
        the proper vgs range to sweep.  This range is saved to this file.
    sch_params :
        Dictionary of default schematic parameters.
    layout_params :
        Dictionary of default layout parameters.
    dsn_name_base :
        the generated transistor cellview base name.
    sim_envs :
        List of simulation environment names.
    view_name :
        Extracted cell view name.
    impl_lib :
        library to put all generated cellviews.
    tb_ibias :
        bias current testbench parameters.  This testbench is used to
        find the proper range of vgs to characterize the transistor with.
        It should have the following entries:

        vgs_max :
            magnitude of maximum vgs value.
        vgs_num :
            number of vgs points.
    """

    def __init__(self, prj, spec_file):
        # type: (Optional[BagProject], str) -> None
        super(MOSCharSS, self).__init__(prj, spec_file)

    @property
    def width(self):
        """Returns the transistor width."""
        return self.specs['layout_params']['w']

    def get_default_dsn_value(self, name):
        # type: (str) -> Any
        """Returns default design parameter value."""
        return self.specs['layout_params'][name]

    def is_nmos(self, val_list):
        # type: (Tuple[Any, ...]) -> bool
        """Given current schematic parameter values, returns True if we're working with NMOS.."""
        try:
            # see if mos_type is one of the sweep.
            idx = self.swp_var_list.index('mos_type')
            return val_list[idx] == 'nch'
        except ValueError:
            # mos_type is not one of the sweep.
            return self.specs['layout_params']['mos_type'] == 'nch'

    def get_vgs_specs(self):
        # type: () -> Dict[str, Any]
        """laods VGS specifications from file and return it as dictionary."""
        vgs_file = os.path.join(self.specs['root_dir'], self.specs['vgs_file'])
        return read_yaml(vgs_file)

    def configure_tb(self, tb_type, tb, val_list):
        # type: (str, Testbench, Tuple[Any, ...]) -> None

        tb_specs = self.specs[tb_type]
        sim_envs = self.specs['sim_envs']
        view_name = self.specs['view_name']
        impl_lib = self.specs['impl_lib']
        dsn_name_base = self.specs['dsn_name_base']

        tb_params = tb_specs['tb_params']
        dsn_name = self.get_instance_name(dsn_name_base, val_list)
        is_nmos = self.is_nmos(val_list)

        tb.set_simulation_environments(sim_envs)
        tb.set_simulation_view(impl_lib, dsn_name, view_name)

        if tb_type == 'tb_ibias':
            tb.set_parameter('vgs_num', tb_params['vgs_num'])

            # handle VGS sign for nmos/pmos
            vgs_max = tb_params['vgs_max']
            if is_nmos:
                tb.set_parameter('vs', 0.0)
                tb.set_parameter('vgs_start', 0.0)
                tb.set_parameter('vgs_stop', vgs_max)
            else:
                tb.set_parameter('vs', vgs_max)
                tb.set_parameter('vgs_start', -vgs_max)
                tb.set_parameter('vgs_stop', 0.0)
        else:
            vgs_info = self.get_vgs_specs()
            vgs_start, vgs_stop = vgs_info[dsn_name]
            vbs_val = tb_params['vbs']

            # handle VBS sign and set parameters.
            if isinstance(vbs_val, list):
                if is_nmos:
                    vbs_val = sorted((-abs(v) for v in vbs_val))
                else:
                    vbs_val = sorted((abs(v) for v in vbs_val))
                tb.set_sweep_parameter('vbs', values=vbs_val)
            else:
                if is_nmos:
                    vbs_val = -abs(vbs_val)
                else:
                    vbs_val = abs(vbs_val)
                tb.set_parameter('vbs', vbs_val)

            if tb_type == 'tb_sp':
                tb.set_parameter('vgs_num', tb_params['vgs_num'])
                tb.set_parameter('sp_freq', tb_params['sp_freq'])

                vds_min = tb_params['vds_min']
                vds_max = tb_params['vds_max']
                vds_num = tb_params['vds_num']
                tb.set_parameter('vgs_start', vgs_start)
                tb.set_parameter('vgs_stop', vgs_stop)
                # handle VDS/VGS sign for nmos/pmos
                if is_nmos:
                    vds_vals = np.linspace(vds_min, vds_max, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_parameter('vb_dc', 0)
                else:
                    vds_vals = np.linspace(-vds_max, -vds_min, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_parameter('vb_dc', abs(vgs_start))
            elif tb_type == 'tb_noise':
                tb.set_parameter('freq_start', tb_params['freq_start'])
                tb.set_parameter('freq_stop', tb_params['freq_stop'])
                tb.set_parameter('num_per_dec', tb_params['num_per_dec'])

                vds_min = tb_params['vds_min']
                vds_max = tb_params['vds_max']
                vds_num = tb_params['vds_num']
                vgs_num = tb_params['vgs_num']
                vgs_vals = np.linspace(vgs_start, vgs_stop, vgs_num + 1)
                # handle VDS/VGS sign for nmos/pmos
                if is_nmos:
                    vds_vals = np.linspace(vds_min, vds_max, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_sweep_parameter('vgs', values=vgs_vals)
                    tb.set_parameter('vb_dc', 0)
                else:
                    vds_vals = np.linspace(-vds_max, -vds_min, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_sweep_parameter('vgs', values=vgs_vals)
                    tb.set_parameter('vb_dc', abs(vgs_start))
            else:
                raise ValueError('Unknown testbench type: %s' % tb_type)

    def process_ibias_data(self, write=True):
        # type: () -> None
        tb_type = 'tb_ibias'
        tb_specs = self.specs[tb_type]
        dsn_name_base = self.specs['dsn_name_base']
        root_dir = self.specs['root_dir']
        vgs_file = self.specs['vgs_file']
        layout_params = self.specs['layout_params']

        fg = layout_params['fg']
        ibias_min_fg = tb_specs['ibias_min_fg']
        ibias_max_fg = tb_specs['ibias_max_fg']
        vgs_res = tb_specs['vgs_resolution']

        ans = {}
        for val_list in self.get_combinations_iter():
            # invert PMOS ibias sign
            is_nmos = self.is_nmos(val_list)
            ibias_sgn = 1.0 if is_nmos else -1.0
            results = self.get_sim_results(tb_type, val_list)

            # assume first sweep parameter is corner, second sweep parameter is vgs
            corner_idx = results['sweep_params']['ibias'].index('corner')
            vgs = results['vgs']
            ibias = results['ibias'] * ibias_sgn  # type: np.ndarray

            wv_max = Waveform(vgs, np.amax(ibias, corner_idx), 1e-6, order=2)
            wv_min = Waveform(vgs, np.amin(ibias, corner_idx), 1e-6, order=2)
            vgs1 = wv_max.get_crossing(ibias_min_fg * fg)
            if vgs1 is None:
                vgs1 = vgs[0] if is_nmos else vgs[-1]
            vgs2 = wv_min.get_crossing(ibias_max_fg * fg)
            if vgs2 is None:
                vgs2 = vgs[-1] if is_nmos else vgs[0]

            if is_nmos:
                vgs_min, vgs_max = vgs1, vgs2
            else:
                vgs_min, vgs_max = vgs2, vgs1

            vgs_min = math.floor(vgs_min / vgs_res) * vgs_res
            vgs_max = math.ceil(vgs_max / vgs_res) * vgs_res

            dsn_name = self.get_instance_name(dsn_name_base, val_list)
            print('%s: vgs = [%.4g, %.4g]' % (dsn_name, vgs_min, vgs_max))
            ans[dsn_name] = [vgs_min, vgs_max]

        if write:
            vgs_file = os.path.join(root_dir, vgs_file)
            with open_file(vgs_file, 'w') as f:
                yaml.dump(ans, f)

    def _get_ss_params(self, method='linear', cfit_method='average'):
        # type: (str, str) -> Tuple[List[str], List[str], Dict[str, Dict[str, List[LinearInterpolator]]]]
        tb_type = 'tb_sp'
        tb_specs = self.specs[tb_type]
        layout_params = self.specs['layout_params']
        dsn_name_base = self.specs['dsn_name_base']

        fg = layout_params['fg']
        char_freq = tb_specs['tb_params']['sp_freq']

        axis_names = ['corner', 'vbs', 'vds', 'vgs']
        ss_swp_names = None  # type: List[str]
        corner_list = None
        corner_sort_arg = None  # type: Sequence[int]
        total_dict = {}
        for val_list in self.get_combinations_iter():
            dsn_name = self.get_instance_name(dsn_name_base, val_list)
            results = self.get_sim_results(tb_type, val_list)
            ibias = results['ibias']
            if not self.is_nmos(val_list):
                ibias *= -1
            ss_dict = mos_y_to_ss(results, char_freq, fg, ibias, cfit_method=cfit_method)

            if corner_list is None:
                ss_swp_names = [name for name in axis_names[1:] if name in results]
                corner_list = results['corner']
                corner_sort_arg = np.argsort(corner_list)  # type: Sequence[int]
                corner_list = corner_list[corner_sort_arg].tolist()

            cur_scales = []
            for name in ss_swp_names:
                cur_xvec = results[name]
                cur_scales.append((cur_xvec[0], cur_xvec[1] - cur_xvec[0]))

            # rearrange array axis
            sweep_params = results['sweep_params']
            swp_vars = sweep_params['ibias']
            order = [swp_vars.index(name) for name in axis_names if name in swp_vars]
            # just to be safe, we create a list copy to avoid modifying dictionary
            # while iterating over view.
            for key in list(ss_dict.keys()):
                new_data = np.transpose(ss_dict[key], axes=order)
                fun_list = []
                for idx in corner_sort_arg:
                    fun_list.append(interpolate_grid(cur_scales, new_data[idx, ...], method=method,
                                                     extrapolate=True, delta=1e-5))
                ss_dict[key] = fun_list

            # derived ss parameters
            self._add_derived_ss_params(ss_dict)

            total_dict[dsn_name] = ss_dict

        return corner_list, ss_swp_names, total_dict

    @classmethod
    def _add_derived_ss_params(cls, ss_dict):
        cgdl = ss_dict['cgd']
        cgsl = ss_dict['cgs']
        cgbl = ss_dict['cgb']
        cdsl = ss_dict['cds']
        cdbl = ss_dict['cdb']
        csbl = ss_dict['csb']

        ss_dict['cgg'] = [cgd + cgs + cgb for (cgd, cgs, cgb) in zip(cgdl, cgsl, cgbl)]
        ss_dict['cdd'] = [cgd + cds + cdb for (cgd, cds, cdb) in zip(cgdl, cdsl, cdbl)]
        ss_dict['css'] = [cgs + cds + csb for (cgs, cds, csb) in zip(cgsl, cdsl, csbl)]

    def _get_integrated_noise(self, fstart, fstop, scale=1.0):
        # type: (Optional[float], Optional[float], float) -> Tuple[List[str], Dict[str, List[LinearInterpolator]]]
        tb_type = 'tb_noise'
        layout_params = self.specs['layout_params']
        dsn_name_base = self.specs['dsn_name_base']

        fg = layout_params['fg']

        axis_names = ['corner', 'vbs', 'vds', 'vgs', 'freq']
        ss_swp_names = None  # type: List[str]
        delta_list = None
        corner_list = log_freq = None
        corner_sort_arg = None  # type: Sequence[int]
        output_dict = {}
        fstart_log = np.log(fstart)
        fstop_log = np.log(fstop)
        for val_list in self.get_combinations_iter():
            dsn_name = self.get_instance_name(dsn_name_base, val_list)
            results = self.get_sim_results(tb_type, val_list)
            out = np.log(scale / fg * results['idn'] ** 2)

            if corner_list is None:
                ss_swp_names = [name for name in axis_names[1:] if name in results]
                delta_list = [1e-6] * len(ss_swp_names)
                delta_list[-1] = 1e-3
                corner_list = results['corner']
                corner_sort_arg = np.argsort(corner_list)  # type: Sequence[int]
                corner_list = corner_list[corner_sort_arg].tolist()
                log_freq = np.log(results['freq'])

            cur_points = [results[name] for name in ss_swp_names]
            cur_points[-1] = log_freq

            # rearrange array axis
            sweep_params = results['sweep_params']
            swp_vars = sweep_params['idn']
            order = [swp_vars.index(name) for name in axis_names if name in swp_vars]
            fun_list = []
            out_trans = np.transpose(out, axes=order)
            for idx in corner_sort_arg:
                noise_fun = LinearInterpolator(cur_points, out_trans[idx, ...], delta_list, extrapolate=True)
                integ_noise = noise_fun.integrate(fstart_log, fstop_log, axis=-1, logx=True, logy=True)
                fun_list.append(integ_noise)
            output_dict[dsn_name] = fun_list

        return corner_list, output_dict

    def get_ss_info(self,
                    fstart=None,  # type: Optional[float]
                    fstop=None,  # type: Optional[float]
                    scale=1.0,  # type: float
                    temp=300,  # type: float
                    method='linear',  # type: str
                    cfit_method='average',  # type: str
                    ):
        # type: (...) -> Tuple[List[str], List[str], Dict[str, Dict[str, List[LinearInterpolator]]]]
        corner_list, ss_swp_names, tot_dict = self._get_ss_params(method=method, cfit_method=cfit_method)
        if fstart is not None and fstop is not None:
            _, noise_dict = self._get_integrated_noise(fstart, fstop, scale=scale)

            k = scale * (fstop - fstart) * 4 * scipy.constants.Boltzmann * temp
            for key, val in tot_dict.items():
                gm = val['gm']
                noise_var = noise_dict[key]
                val['gamma'] = [nf / gmf / k for nf, gmf in zip(noise_var, gm)]

        return corner_list, ss_swp_names, tot_dict


class MOSDBDiscrete(object):
    """Transistor small signal parameters database with discrete width choices.

    This class provides useful query/optimization methods and ways to store/retrieve
    data.

    Parameters
    ----------
    spec_list : List[str]
        list of specification file locations corresponding to widths.
    noise_fstart : Optional[float]
        noise integration frequency lower bound.  None to disable noise.
    noise_fstop : Optional[float]
        noise integration frequency upper bound.  None to disable noise.
    noise_scale : float
        noise integration scaling factor.
    noise_temp : float
        noise temperature.
    method : str
        interpolation method.
    cfit_method : str
        method used to fit capacitance to Y parameters.
    bag_config_path : Optional[str]
        BAG configuration file path.
    """

    def __init__(self,
                 spec_list,  # type: List[str]
                 noise_fstart=None,  # type: Optional[float]
                 noise_fstop=None,  # type: Optional[float]
                 noise_scale=1.0,  # type: float
                 noise_temp=300,  # type: float
                 method='linear',  # type: str
                 cfit_method='average',  # type: str
                 bag_config_path=None,  # type: Optional[str]
                 ):
        # type: (...) -> None
        # error checking

        tech_info = create_tech_info(bag_config_path=bag_config_path)

        self._width_res = tech_info.tech_params['mos']['width_resolution']
        self._sim_envs = None
        self._ss_swp_names = None
        self._sim_list = []
        self._ss_list = []
        self._ss_outputs = None
        self._width_list = []
        for spec in spec_list:
            sim = MOSCharSS(None, spec)
            self._width_list.append(int(round(sim.width / self._width_res)))
            # error checking
            if 'w' in sim.swp_var_list:
                raise ValueError('MOSDBDiscrete assumes transistor width is not swept.')

            corners, ss_swp_names, ss_dict = sim.get_ss_info(noise_fstart, noise_fstop,
                                                             scale=noise_scale, temp=noise_temp,
                                                             method=method, cfit_method=cfit_method)
            if self._sim_envs is None:
                self._ss_swp_names = ss_swp_names
                self._sim_envs = corners
                test_dict = next(iter(ss_dict.values()))
                self._ss_outputs = sorted(test_dict.keys())
            elif self._sim_envs != corners:
                raise ValueError('Simulation environments mismatch between given specs.')
            elif self._ss_swp_names != ss_swp_names:
                raise ValueError('signal-signal parameter sweep names mismatch.')

            self._sim_list.append(sim)
            self._ss_list.append(ss_dict)

        self._env_list = self._sim_envs
        self._cur_idx = 0
        self._dsn_params = dict(w=self._width_list[0])

    @property
    def width_list(self):
        # type: () -> List[Union[float, int]]
        """Returns the list of widths in this database."""
        return [w * self._width_res for w in self._width_list]

    @property
    def env_list(self):
        # type: () -> List[str]
        """The list of simulation environments to consider."""
        return self._env_list

    @env_list.setter
    def env_list(self, new_env_list):
        # type: (List[str]) -> None
        """Sets the list of simulation environments to consider."""
        self._env_list = new_env_list

    @property
    def dsn_params(self):
        # type: () -> Tuple[str, ...]
        """List of design parameters."""
        return self._sim_list[self._cur_idx].swp_var_list

    def get_default_dsn_value(self, var):
        # type: (str) -> Any
        """Returns the default design parameter values."""
        return self._sim_list[self._cur_idx].get_default_dsn_value(var)

    def get_dsn_param_values(self, var):
        # type: (str) -> List[Any]
        """Returns a list of valid design parameter values."""
        return self._sim_list[self._cur_idx].get_swp_var_values(var)

    def set_dsn_params(self, **kwargs):
        # type: (**kwargs) -> None
        """Set the design parameters for which this database will query for."""
        self._dsn_params.update(kwargs)
        w_unit = int(round(self._dsn_params['w'] / self._width_res))
        self._cur_idx = self._width_list.index(w_unit)

    def _get_dsn_name(self, **kwargs):
        # type: (**kwargs) -> str
        if kwargs:
            self.set_dsn_params(**kwargs)
        dsn_name = self._sim_list[self._cur_idx].get_design_name(self._dsn_params)

        if dsn_name not in self._ss_list[self._cur_idx]:
            raise ValueError('Unknown design name: %s.  Did you set design parameters?' % dsn_name)

        return dsn_name

    def get_function_list(self, name, **kwargs):
        # type: (str, **kwargs) -> List[DiffFunction]
        """Returns a list of functions, one for each simulation environment, for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        **kwargs :
            design parameter values.

        Returns
        -------
        output : Union[RegGridInterpVectorFunction, RegGridInterpFunction]
            the output vector function.
        """
        dsn_name = self._get_dsn_name(**kwargs)
        cur_dict = self._ss_list[self._cur_idx][dsn_name]
        fun_list = []
        for env in self.env_list:
            try:
                env_idx = self._sim_envs.index(env)
            except ValueError:
                raise ValueError('environment %s not found.' % env)

            fun_list.append(cur_dict[name][env_idx])
        return fun_list

    def get_function(self, name, env='', **kwargs):
        # type: (str, str, **kwargs) -> Union[VectorDiffFunction, DiffFunction]
        """Returns a function for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        env : str
            if not empty, we will return function for just the given simulation environment.
        **kwargs :
            design parameter values.

        Returns
        -------
        output : Union[RegGridInterpVectorFunction, RegGridInterpFunction]
            the output vector function.
        """
        if not env and len(self.env_list) == 1:
            env = self.env_list[0]

        if not env:
            return VectorDiffFunction(self.get_function_list(name, **kwargs))
        else:
            dsn_name = self._get_dsn_name(**kwargs)
            cur_dict = self._ss_list[self._cur_idx][dsn_name]
            try:
                env_idx = self._sim_envs.index(env)
            except ValueError:
                raise ValueError('environment %s not found.' % env)

            return cur_dict[name][env_idx]

    def get_fun_sweep_params(self, **kwargs):
        # type: (**kwargs) -> Tuple[List[str], List[Tuple[float, float]]]
        """Returns interpolation function sweep parameter names and values.

        Parameters
        ----------
        **kwargs :
            design parameter values.

        Returns
        -------
        sweep_params : List[str]
            list of parameter names.
        sweep_range : List[Tuple[float, float]]
            list of parameter range
        """
        dsn_name = self._get_dsn_name(**kwargs)
        sample_fun = self._ss_list[self._cur_idx][dsn_name]['gm'][0]

        return self._ss_swp_names, sample_fun.input_ranges

    def get_fun_arg(self, **kwargs):
        # type: (**kwargs) -> np.multiarray.ndarray
        """Convert keyword arguments to function argument."""
        return np.array([kwargs[key] for key in self._ss_swp_names])

    def get_fun_arg_index(self, name):
        # type: (str) -> int
        """Returns the function input argument index for the given variable"""
        return self._ss_swp_names.index(name)

    def query(self, **kwargs):
        # type: (**kwargs) -> Dict[str, np.multiarray.ndarray]
        """Query the database for the values associated with the given parameters.

        All parameters must be specified.

        Parameters
        ----------
        **kwargs :
            parameter values.

        Returns
        -------
        results : Dict[str, np.ndarray]
            the characterization results.
        """
        fun_arg = self.get_fun_arg(**kwargs)
        results = {name: self.get_function(name, **kwargs)(fun_arg) for name in self._ss_outputs}

        for key in self._ss_swp_names:
            results[key] = kwargs[key]

        return results
