# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import pkg_resources
from typing import Tuple, Sequence, Dict, Union, Any

from bag import float_to_si_string
from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'dut_wrapper_dm.yaml'))


# noinspection PyPep8Naming
class bag_testbenches__dut_wrapper_dm(Module):
    """A class that wraps a differential DUT to single-ended.
    """

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            dut_lib='DUT library name.',
            dut_cell='DUT cell name.',
            balun_list='list of baluns to create.',
            pin_list='list of input/output pins.',
            dut_conns='DUT connection dictionary.',
            cap_list='list of load capacitances.',
            vcvs_list='list of voltage-controlled voltage sources.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            cap_list=None,
            vcvs_list=None,
        )

    def design(self,  # type: bag_testbenches__dut_wrapper_dm
               dut_lib='',  # type: str
               dut_cell='',  # type: str
               balun_list=None,  # type: Sequence[Tuple[str, str, str, str]]
               cap_list=None,  # type: Sequence[Tuple[str, str, Union[float, str]]]
               pin_list=None,  # type: Sequence[Tuple[str, str]]
               dut_conns=None,  # type: Dict[str, str]
               vcvs_list=None,  # type: Sequence[Tuple[str, str, str, str, Dict[str, Any]]]
               ):
        # type: (...) -> None
        """Design this wrapper schematic.

        This cell converts a variable number of differential pins to single-ended pins or
        vice-versa, by using ideal_baluns.  It can also create extra pins to be connected
        to the device.  You can also optionally instantiate capacitive loads or
        voltage-controlled voltage sources.

        VDD and VSS pins will always be there for primary supplies.  Additional supplies
        can be added as inputOutput pins using the pin_list parameters.  If you don't need
        supply pins, they will be left unconnected.

        NOTE: the schematic template contains pins 'inac', 'indc', 'outac', and 'outdc' by
        default.  However, if they are not specified in pin_list, they will be deleted.
        In this way designer has full control over how they want the inputs/outputs to be
        named.

        Parameters
        ----------
        dut_lib : str
            DUT library name.
        dut_cell : str
            DUT cell name.
        balun_list: Sequence[Tuple[str, str, str, str]]
            list of baluns to instantiate, represented as a list of
            (diff, comm, pos, neg) tuples.
        cap_list : Sequence[Tuple[str, str, Union[float, str]]]
            list of load capacitors to create.  Represented as a list of (pos, neg, cap_val) tuples.
            cap_val can be either capacitance value in Farads or a variable name/expression.
        pin_list : Sequence[Tuple[str, str]]
            list of pins of this schematic, represented as a list of (name, purpose) tuples.
            purpose can be 'input', 'output', or 'inputOutput'.
        dut_conns : Dict[str, str]
            a dictionary from DUT pin name to the net name.  All connections should
            be specified, including VDD and VSS.
        vcvs_list : Sequence[Tuple[str, str, str, str, Dict[str, Any]]]
            list of voltage-controlled voltage sources to create.  Represented as a list of
            (pos, neg, ctrl-pos, ctrl-neg, params) tuples.
        """
        # error checking
        if not balun_list:
            raise ValueError('balun_list cannot be None or empty.')
        if not pin_list:
            raise ValueError('pin_list cannot be None or empty.')
        if not dut_conns:
            raise ValueError('dut_conns cannot be None or empty.')

        # delete default input/output pins
        for pin_name in ('inac', 'indc', 'outac', 'outdc'):
            self.remove_pin(pin_name)

        # add pins
        for pin_name, pin_type in pin_list:
            self.add_pin(pin_name, pin_type)

        # replace DUT
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)

        # connect DUT
        for dut_pin, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', dut_pin, net_name)

        # add baluns and connect them
        inst_name = 'XBAL'
        num_inst = len(balun_list)
        name_list = ['%s%d' % (inst_name, idx) for idx in range(num_inst)]
        self.array_instance(inst_name, name_list)
        for idx, (diff, comm, pos, neg) in enumerate(balun_list):
            self.reconnect_instance_terminal(inst_name, 'd', diff, index=idx)
            self.reconnect_instance_terminal(inst_name, 'c', comm, index=idx)
            self.reconnect_instance_terminal(inst_name, 'p', pos, index=idx)
            self.reconnect_instance_terminal(inst_name, 'n', neg, index=idx)

        # configure load capacitors
        inst_name = 'CLOAD'
        if cap_list:
            num_inst = len(cap_list)
            name_list = ['%s%d' % (inst_name, idx) for idx in range(num_inst)]
            self.array_instance(inst_name, name_list)
            for idx, (pos, neg, val) in enumerate(cap_list):
                self.reconnect_instance_terminal(inst_name, 'PLUS', pos, index=idx)
                self.reconnect_instance_terminal(inst_name, 'MINUS', neg, index=idx)
                if isinstance(val, str):
                    pass
                elif isinstance(val, float) or isinstance(val, int):
                    val = float_to_si_string(val)
                else:
                    raise ValueError('Unknown schematic instance parameter: %s' % val)
                self.instances[inst_name][idx].parameters['c'] = val
        else:
            self.delete_instance(inst_name)

        # configure vcvs
        inst_name = 'ECTRL'
        if vcvs_list:
            num_inst = len(vcvs_list)
            name_list = ['%s%d' % (inst_name, idx) for idx in range(num_inst)]
            self.array_instance(inst_name, name_list)
            for idx, (pos, neg, cpos, cneg, params) in enumerate(vcvs_list):
                self.reconnect_instance_terminal(inst_name, 'PLUS', pos, index=idx)
                self.reconnect_instance_terminal(inst_name, 'MINUS', neg, index=idx)
                self.reconnect_instance_terminal(inst_name, 'NC+', cpos, index=idx)
                self.reconnect_instance_terminal(inst_name, 'NC-', cneg, index=idx)
                for key, val in params.items():
                    if isinstance(val, str):
                        pass
                    elif isinstance(val, float) or isinstance(val, int):
                        val = float_to_si_string(val)
                    else:
                        raise ValueError('Unknown schematic instance parameter: %s' % val)
                    self.instances[inst_name][idx].parameters[key] = val
        else:
            self.delete_instance(inst_name)
