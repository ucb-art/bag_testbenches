# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional, List

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'mos_tb_noise.yaml'))


# noinspection PyPep8Naming
class bag_testbenches__mos_tb_noise(Module):
    """Transistor noise characterization testbench.

    This testbench is used to characterize the transistor noise.
    """

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            dut_lib="Transistor DUT library name.",
            dut_cell='Transistor DUT cell name.',
            vbias_dict='Additional bias voltage dictionary.',
            ibias_dict='Additional bias current dictionary.',
            dut_conns='Transistor DUT connection dictionary.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            vbias_dict=None,
            ibias_dict=None,
            dut_conns=None,
        )

    def design(self,  # type: Module
               dut_lib,  # type: str
               dut_cell,  # type: str
               vbias_dict,  # type: Optional[Dict[str, List[str]]]]
               ibias_dict,  # type: Optional[Dict[str, List[str]]]]
               dut_conns,  # type: Optional[Dict[str, str]]
               ):
        # type: (...) -> None
        """Design this testbench.
        """
        if vbias_dict is None:
            vbias_dict = {}
        if ibias_dict is None:
            ibias_dict = {}
        if dut_conns is None:
            dut_conns = {}

        # setup bias sources
        self.design_dc_bias_sources(vbias_dict, ibias_dict, 'VBIAS', 'IBIAS', define_vdd=False)

        # setup DUT
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)
        for term_name, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', term_name, net_name)
