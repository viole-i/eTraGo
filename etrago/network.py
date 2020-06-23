#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:32:24 2020

@author: clara
"""
from pypsa.components import Network
import pandas as pd
import numpy as np
from egoio.tools import db
from sqlalchemy.orm import sessionmaker

from etrago.tools.io import NetworkScenario
from etrago.tools.plot import add_coordinates, plot_grid
from six import iteritems, itervalues, iterkeys
from etrago.tools.utilities import set_branch_capacity, convert_capital_costs, add_missing_components, set_random_noise, geolocation_buses_oop
from etrago.tools.extendable import (
            extendable,
            extension_preselection,
            print_expansion_costs)

from pypsa.io import (export_to_csv_folder, import_from_csv_folder,
                 export_to_hdf5, import_from_hdf5,
                 import_from_pypower_ppc, import_components_from_dataframe,
                 import_series_from_dataframe, import_from_pandapower_net)
import logging
logger = logging.getLogger(__name__)

class Etrago(Network):
    
    def __init__(self,
                 args,
                 csv_folder_name=None,
                 name="",
                 ignore_standard_types=False,
                 empty_network=False,
                 **kwargs):
        
        super().__init__(csv_folder_name, "", False, **kwargs)
        self.args=args

       
        if csv_folder_name==None and empty_network==False: 
            conn = db.connection(section=args['db'])
            Session = sessionmaker(bind=conn)
            self.session = Session()
            if self.args['gridversion'] is None:
                self.args['ormcls_prefix'] = 'EgoGridPfHv'
            else:
                self.args['ormcls_prefix'] = 'EgoPfHv'
            self._build_network_from_db()
            self._adjust_network()
            
        
        self.__renewable_carriers = ['wind_onshore', 'wind_offshore', 'solar',
                                     'biomass', 'run_of_river', 'reservoir']
        self.__renewable_gen = self.generators.index[
                self.generators.carrier.isin(self.__renewable_carriers)]


    def _build_network_from_db(self):
        self.scenario = NetworkScenario(self.session,
                                   version=self.args['gridversion'],
                                   prefix=self.args['ormcls_prefix'],
                                   method=self.args['method'],
                                   start_snapshot=self.args['start_snapshot'],
                                   end_snapshot=self.args['end_snapshot'],
                                   scn_name=self.args['scn_name'])

        self.scenario.build_network(network=self)
        
        logger.info('Imported network from db')

    def _adjust_network(self):
        add_coordinates(self)
        geolocation_buses_oop(self)
        add_missing_components(self)
        set_random_noise(self, 0.01)
        self.lines.v_nom=self.lines.bus0.map(self.buses.v_nom)
        self.links.v_nom=self.links.bus0.map(self.buses.v_nom)

        set_branch_capacity(self) # will be replaced when using new pypsa version

        if self.args['extendable'] != []:
            extendable(self, line_max=4)
            convert_capital_costs(self)


#### copied from pypsa
    def copy(self, with_time=True, ignore_standard_types=False):
        """
        Returns a deep copy of the Network object with all components and
        time-dependent data.

        Returns
        --------
        network : etrago.Network

        Parameters
        ----------
        with_time : boolean, default True
            Copy snapshots and time-varying network.component_names_t data too.
        ignore_standard_types : boolean, default False
            Ignore the PyPSA standard types.

        Examples
        --------
        >>> network_copy = network.copy()

        """

        network = self.__class__(args=self.args, empty_network=True, ignore_standard_types=ignore_standard_types)

        for component in self.iterate_components(["Bus", "Carrier"] + sorted(all_components - {"Bus","Carrier"})):
            df = component.df
            #drop the standard types to avoid them being read in twice
            if not ignore_standard_types and component.name in standard_types:
                df = component.df.drop(network.components[component.name]["standard_types"].index)

            import_components_from_dataframe(network, df, component.name)

        if with_time:
            network.set_snapshots(self.snapshots)
            for component in self.iterate_components():
                pnl = getattr(network, component.list_name+"_t")
                for k in iterkeys(component.pnl):
                    pnl[k] = component.pnl[k].copy()

        #catch all remaining attributes of network
        for attr in ["name", "srid"]:
            setattr(network,attr,getattr(self,attr))

        network.snapshot_weightings = self.snapshot_weightings.copy()
        network.lines.v_nom=network.lines.bus0.map(network.buses.v_nom)
        network.links.v_nom=network.links.bus0.map(network.buses.v_nom)
        return network
    
    # import plotting function
    plot_grid=plot_grid

standard_types = {"LineType", "TransformerType"}

passive_one_port_components = {"ShuntImpedance"}
controllable_one_port_components = {"Load", "Generator", "StorageUnit", "Store"}
one_port_components = passive_one_port_components|controllable_one_port_components

passive_branch_components = {"Line", "Transformer"}
controllable_branch_components = {"Link"}
branch_components = passive_branch_components|controllable_branch_components

#i.e. everything except "Network"
all_components = branch_components|one_port_components|standard_types|{"Bus", "SubNetwork", "Carrier", "GlobalConstraint"}
