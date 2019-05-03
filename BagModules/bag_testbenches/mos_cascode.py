# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, List, Tuple

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'mos_cascode.yaml'))


# noinspection PyPep8Naming
class bag_testbenches__mos_cascode(Module):
    """A cascode transistor generator.

    This class is used primarily for transistor characterization purposes.
    """

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            mos_type="Transistor type.  Either 'pch' or 'nch'.",
            lch='Transistor length in meters.',
            intentb='Bottom transistor threshold flavor.',
            intentc='Cascode transistor threshold flavor.',
            wb='Bottom transistor width in meters or number of fins.',
            wc='Cascode transistor width in meters or number of fins.',
            fgb='Bottom transistor number of segments.',
            fgc='Cascode transistor number of segments.',
            stackb='Number of stacked bottom transistors in a segment.',
            stackc='Number of stacked bottom transistors in a segment.',
            dum_info='Dummy information data structure.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            intentb='standard',
            intentc='standard',
            stackb=1,
            stackc=1,
            dum_info=None,
        )

    def design(self,  # type: Module
               mos_type,  # type: str
               lch,  # type: float
               intentb,  # type: str
               intentc,  # type: str
               wb,  # type: Union[float, int]
               wc,  # type: Union[float, int]
               fgb,  # type: int
               fgc,  # type: int
               stackb,  # type: int
               stackc,  # type: int
               dum_info,  # type: List[Tuple[Any]]
               ):
        """Design this cascode transistor.
        """
        if fgb == 1 or fgc == 1:
            raise ValueError('Cannot make 1 finger transistor.')
        # select the correct transistor type
        if mos_type == 'pch':
            self.replace_instance_master('XB', 'BAG_prim', 'pmos4_standard')
            self.replace_instance_master('XC', 'BAG_prim', 'pmos4_standard')

        for inst_name, w, stack, fg, intent in (('XB', wb, stackb, fgb, intentb),
                                                ('XC', wc, stackc, fgc, intentc)):
            if stack > 1:
                # array instances
                name_list = []
                term_list = []
                # add stack transistors
                for idx in range(stack):
                    name_list.append('%s%d<%d:0>' % (inst_name, idx, fg - 1))
                    cur_term = {}
                    if idx != stack - 1:
                        cur_term['S'] = 'mid%d<%d:0>' % (idx, fg - 1)
                    if idx != 0:
                        cur_term['D'] = 'mid%d<%d:0>' % (idx - 1, fg - 1)
                    term_list.append(cur_term)

                # design transistors
                self.array_instance(inst_name, name_list, term_list=term_list)
                for idx in range(stack):
                    self.instances[inst_name][idx].design(w=w, l=lch, nf=1, intent=intent)
            else:
                self.instances[inst_name].design(w=w, l=lch, nf=fg, intent=intent)

        # handle dummy transistors
        self.design_dummy_transistors(dum_info, 'XD', 'b', 'b')
