# -*- coding: utf-8 -*-

"""This module defines measurement classes for amplifiers."""

from typing import TYPE_CHECKING, Tuple, Dict, Any, Sequence

import os

from bag.io.sim_data import save_sim_results
from bag.simulation.core import MeasurementManager

if TYPE_CHECKING:
    from .core import ACTB


class AmpCharAC(MeasurementManager):
    """This class measures AC transfer function of an "amplifier".

    Parameters
    ----------
    data_dir : str
        Simulation data directory.
    meas_name : str
        measurement setup name.
    impl_lib : str
        implementation library name.
    specs : Dict[str, Any]
        the measurement specification dictionary.
    wrapper_lookup : Dict[str, str]
        the DUT wrapper cell name lookup table.
    sim_view_list : Sequence[Tuple[str, str]]
        simulation view list
    env_list : Sequence[str]
        simulation environments list.
    """

    def __init__(self, data_dir, meas_name, impl_lib, specs, wrapper_lookup, sim_view_list, env_list):
        # type: (str, str, str, Dict[str, Any], Dict[str, str], Sequence[Tuple[str, str]], Sequence[str]) -> None
        MeasurementManager.__init__(self, data_dir, meas_name, impl_lib, specs, wrapper_lookup, sim_view_list, env_list)

    def get_initial_state(self):
        # type: () -> str
        """Returns the initial FSM state."""
        return 'ac'

    def process_output(self, state, data, tb_manager):
        # type: (str, Dict[str, Any], ACTB) -> Tuple[bool, str, Dict[str, Any]]

        done = True
        next_state = ''

        gain_w3db_results = tb_manager.get_gain_and_w3db(data, tb_manager.get_outputs())
        file_name = os.path.join(self.data_dir, 'gain_w3db.hdf5')
        save_sim_results(gain_w3db_results, file_name)
        output = dict(gain_w3db_file=file_name)

        return done, next_state, output
