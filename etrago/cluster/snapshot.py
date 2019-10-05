# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems


# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description for read-the-docs
""" This module contains functions for calculating representative days/weeks
based on a pyPSA network object. It is designed to be used for the `lopf`
method. Essentially the tsam package
( https://github.com/FZJ-IEK3-VSA/tsam ), which is developed by
Leander Kotzur is used.

Remaining questions/tasks:

- Does it makes sense to cluster normed values?
- Include scaling method for yearly sums
"""

import pandas as pd
import os
from etrago.tools.utilities import results_to_csv

if 'READTHEDOCS' not in os.environ:
    import pyomo.environ as po
    import tsam.timeseriesaggregation as tsam
    import datetime

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "Simon Hilpert"

write_results=True

def snapshot_clustering(network, args, how='daily'):
    """
    """
    resultspath = args['csv_export']
    clusters = args['snapshot_clustering']
    # original problem
    network, df_cluster=run(network=network.copy(), args=args, path=resultspath, write_results=write_results, n_clusters=None,
                  how=how, normed=False)
    
    network_original=network.copy()
    
    for c in clusters:
        path=os.path.join(resultspath, how)
        network, df_cluster=run(network=network_original.copy(), args=args, path=path, write_results=write_results, n_clusters=c, 
            how=how, normed=False)

    return network, df_cluster


def tsam_cluster(timeseries_df, typical_periods=10, how='daily'):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timeseries to cluster

    Returns
    -------
    timeseries : pd.DataFrame
        Clustered timeseries
    """

    if how == 'daily':
        hours = 24
    if how == 'weekly':
        hours = 168

    aggregation = tsam.TimeSeriesAggregation(
        timeseries_df,
        noTypicalPeriods=typical_periods,
        rescaleClusterPeriods=False,
        hoursPerPeriod=hours,
        clusterMethod='hierarchical')

    timeseries = aggregation.createTypicalPeriods()
    cluster_weights = aggregation.clusterPeriodNoOccur
    clusterOrder =aggregation.clusterOrder
    clusterCenterIndices= aggregation.clusterCenterIndices 
   
    # get all index for every hour of that day of the clusterCenterIndices
    start=[]  
    # get the first hour of the clusterCenterIndices (days start with 0)
    for i in clusterCenterIndices:
        start.append(i*hours)
        
    # get a list with all hours belonging to the clusterCenterIndices
    nrhours=[]  
    for j in start:
        nrhours.append(j)
        x = 1
        while x < hours: 
            j = j + 1
            nrhours.append(j)
            x = x + 1
                
    # get the origial Datetimeindex
    dates = timeseries_df.iloc[nrhours].index 
    
    #get list of representative days   
    representative_day=[]

    #cluster:medoid des jeweiligen Clusters
    dic_clusterCenterIndices = dict(enumerate(clusterCenterIndices)) 
    for i in clusterOrder: 
        representative_day.append(dic_clusterCenterIndices[i])

    #get list of last hour of representative days
    last_hour_datetime=[]
    for i in representative_day:
        last_hour = i * hours + hours - 1
        last_hour_datetime.append(timeseries_df.index[last_hour])
    #create a dataframe (index=nr. of day in a year/candidate)
    df_cluster =  pd.DataFrame({
                        'Cluster': clusterOrder, #Cluster of the day
                        'RepresentativeDay': representative_day, #representative day of the cluster
                        'last_hour_RepresentativeDay': last_hour_datetime}) #last hour of the cluster  
    df_cluster.index = df_cluster.index + 1
    df_cluster.index.name = 'Candidate'
    
    #create a dataframe each timeseries (h) and its candiddate day (i) df_i_h
    nr_day = []
    x = len(timeseries_df.index)/hours+1
    
    for i in range(1,int(x)):
        j=1
        while j <= hours: 
            nr_day.append(i)
            j=j+1     
    df_i_h = pd.DataFrame({'Timeseries': timeseries_df.index, 
                        'Candidate_day': nr_day}) 
    df_i_h.set_index('Timeseries',inplace=True)

    return df_cluster, cluster_weights, dates, hours, df_i_h

def disaggregate_soc_results(network):
    """
    Disaggregate snapshot clustered results. 
    Set soc_intra from cluster-days, soc_inter for each day and 
    soc as sum of soc_intra and soc_inter.
    """
    
    
    network.storage_units_t['state_of_charge_intra'] = pd.DataFrame(
                index = network.storage_units_t.state_of_charge.index, 
                columns = network.storage_units.index)
    network.storage_units_t['state_of_charge_inter'] = pd.DataFrame(
                index = network.storage_units_t.state_of_charge.index, 
                columns = network.storage_units.index)
        
    d = network.model.state_of_charge_intra.extract_values()
    for k in d.keys():
            network.storage_units_t['state_of_charge_intra'][k[0]][k[1]] = d[k]
    inter = network.model.state_of_charge_inter.extract_values() 
        
    for s in network.cluster.index: 
        snapshots = network.snapshots[
                network.snapshots.dayofyear-1  ==
                network.cluster['RepresentativeDay'][s]]

        day = pd.date_range(
                start = pd.to_datetime(s-1, 
                                       unit='d',
                                       origin=pd.Timestamp('2011-01-01')), 
                                       periods=24, freq = 'h')

        for su in network.storage_units.index:
            network.storage_units_t['state_of_charge_inter'][su][day] = \
                    inter[(su, s)]
            if not (day == snapshots).all():
                network.storage_units_t['state_of_charge_intra'][su][day]=\
                network.storage_units_t['state_of_charge_intra'][su][snapshots]
                    
                network.storage_units_t['state_of_charge_inter'][su][day] = \
                    inter[(su, s)]
                
                network.storage_units_t['state_of_charge'][su][day] = \
                    network.storage_units_t['state_of_charge_intra'][su][day] \
                    + network.storage_units_t['state_of_charge_inter'][su][day]

def run(network, args, path, write_results=False, n_clusters=None, how='daily',
        normed=False):
    """
    """
    
    if n_clusters is not None:
        path=os.path.join(path, str(n_clusters))

        # calculate clusters
        df_cluster, cluster_weights, dates, hours, df_i_h = tsam_cluster(
                prepare_pypsa_timeseries(network),
                typical_periods=n_clusters,
                how='daily')       
        network.cluster = df_cluster
        network.cluster_ts = df_i_h

        update_data_frames(network, cluster_weights, dates, hours)
        
        network.lopf(network.snapshots, 
                     extra_functionality=snapshot_cluster_constraints,
                     solver_name=args['solver'], 
                     solver_options=args['solver_options'], 
                     formulation = 'kirchhoff')
        
        # disaggregate soc results
        disaggregate_soc_results(network)
                    
    else:
        path=os.path.join(path, 'original')
        
        # network.cluster=False
        
        network.lopf(network.snapshots,
                     extra_functionality=None,
                     solver_name=args['solver'], 
                     solver_options=args['solver_options'],
                     formulation = 'kirchhoff')
        df_cluster=None
           
        
    
    if write_results:
        results_to_csv(network, args, path, pf_solution=None)
        write_lpfile(network, path=os.path.join(path, 'file.lp'))
    
    return network, df_cluster


def prepare_pypsa_timeseries(network, normed=False):
    """
    """

    if normed:
        normed_loads = network.loads_t.p_set / network.loads_t.p_set.max()
        normed_loads.columns = 'L' + normed_loads.columns
        normed_renewables = network.generators_t.p_max_pu
        normed_renewables.columns = 'G' + normed_renewables.columns

        df = pd.concat([normed_renewables,
                        normed_loads], axis=1)
    else:
        loads = network.loads_t.p_set.copy()
        loads.columns = 'L' + loads.columns
        renewables = network.generators_t.p_set.copy()
        renewables.columns = 'G' + renewables.columns
        df = pd.concat([renewables, loads], axis=1)

    return df


def update_data_frames(network, cluster_weights, dates, hours):
    """ Updates the snapshots, snapshots weights and the dataframes based on
    the original data in the network and the medoids created by clustering
    these original data.

    Parameters
    -----------
    network : pyPSA network object
    cluster_weights: dictionary
    dates: Datetimeindex


    Returns
    -------
    network

    """
    
    network.snapshot_weightings = network.snapshot_weightings.loc[dates]
    network.snapshots = network.snapshot_weightings.index

    # set new snapshot weights from cluster_weights
    snapshot_weightings = []
    for i in cluster_weights.values():
        x = 0
        while x < hours:
            snapshot_weightings.append(i)
            x += 1
    for i in range(len(network.snapshot_weightings)):
        network.snapshot_weightings[i] = snapshot_weightings[i]   
    
    #put the snapshot in the right order
    network.snapshots=network.snapshots.sort_values()
    network.snapshot_weightings=network.snapshot_weightings.sort_index()

     
    return network

def snapshot_cluster_constraints(network, snapshots):
    """  
    Sets snapshot cluster constraints for storage units according to :
    L. Kotzur et al: 'Time series aggregation for energy system design: 
    Modeling seasonal storage', 2018.
    
    Parameters
    -----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    snapshots: pd.DateTimeSeries
        List of snapshots 
    """
    sus = network.storage_units
    # take every first hour of the clustered days
    network.model.period_starts = network.snapshot_weightings.index[0::24]

    network.model.storages = sus.index
    
    if True:
        candidates = network.cluster.index.get_values()

        # create set for inter-temp constraints and variables
        network.model.candidates = po.Set(initialize=candidates,
                                          ordered=True)

        # create intra soc variable for each storage and each hour
        network.model.state_of_charge_intra = po.Var(
            sus.index, network.snapshots)

        def intra_soc_rule(m, s, h):
            """
            Sets soc_inter of first hour of every day to 0. Other hours are set
            by total_soc_contraint and pypsa's state_of_charge_constraint
            
            According to:
            L. Kotzur et al: 'Time series aggregation for energy system design: 
            Modeling seasonal storage', 2018, equation no. 18
            """
            
            if h.hour ==  0:
                expr = (m.state_of_charge_intra[s, h] == 0)
            else:
                expr = po.Constraint.Skip
                expr = (
                    m.state_of_charge_intra[s, h ] ==
                    m.state_of_charge_intra[s, h-pd.DateOffset(hours=1)] 
                    * (1 - network.storage_units.at[s, 'standing_loss'])
                    -(m.storage_p_dispatch[s,h-pd.DateOffset(hours=1)]/
                        network.storage_units.at[s, 'efficiency_dispatch'] -
                        network.storage_units.at[s, 'efficiency_store'] * 
                        m.storage_p_store[s,h-pd.DateOffset(hours=1)]))
            return expr

        network.model.soc_intra_all = po.Constraint(
            network.model.storages, network.snapshots, rule = intra_soc_rule)       
        
        # create inter soc variable for each storage and each candidate
        network.model.state_of_charge_inter = po.Var(
            sus.index, network.model.candidates, 
            within=po.NonNegativeReals)

        def inter_storage_soc_rule(m, s, i):
            """
            Define the state_of_charge_inter as the state_of_charge_inter of
            the day before minus the storage losses plus the state_of_charge_intra
            of one hour after the last hour of the representative day.
            For the last reperesentive day, the soc_inter is the same as 
            the first day due to cyclic soc condition 
            
            According to:
            L. Kotzur et al: 'Time series aggregation for energy system design: 
            Modeling seasonal storage', 2018, equation no. 19
            """
            last_hour = network.cluster["last_hour_RepresentativeDay"][i]

            if i == network.model.candidates[-1]:
               # print(last_hour)
               # expr = po.Constraint.Skip
                expr = (
                m.state_of_charge_inter[s, network.model.candidates[1] ] ==
               m.state_of_charge_inter[s, i] 
                * (1 - network.storage_units.at[s, 'standing_loss'])**24
                + m.state_of_charge_intra[s, last_hour]\
                        * (1 - network.storage_units.at[s, 'standing_loss'])
                        -(m.storage_p_dispatch[s, last_hour]/\
                        network.storage_units.at[s, 'efficiency_dispatch'] -
                        network.storage_units.at[s, 'efficiency_store'] * 
                        m.storage_p_store[s,last_hour]))

            else:
                expr = (
                m.state_of_charge_inter[s, i+1 ] ==
                m.state_of_charge_inter[s, i] 
                * (1 - network.storage_units.at[s, 'standing_loss'])**24
                + m.state_of_charge_intra[s, last_hour]\
                        * (1 - network.storage_units.at[s, 'standing_loss'])\
                        -(m.storage_p_dispatch[s, last_hour]/\
                        network.storage_units.at[s, 'efficiency_dispatch'] -
                        network.storage_units.at[s, 'efficiency_store'] * 
                        m.storage_p_store[s,last_hour]))
        
            return expr

        network.model.inter_storage_soc_constraint = po.Constraint(
            sus.index, network.model.candidates,
            rule=inter_storage_soc_rule)

                #new definition of the state_of_charge used in pypsa
        network.model.del_component('state_of_charge_constraint')
        network.model.del_component('state_of_charge_constraint_index')
        network.model.del_component('state_of_charge_constraint_index_0')
        network.model.del_component('state_of_charge_constraint_index_1')
        
        def total_state_of_charge(m,s,h):
            """
            Define the state_of_charge as the sum of state_of_charge_inter 
            and state_of_charge_intra
            
            According to:
            L. Kotzur et al: 'Time series aggregation for energy system design: 
            Modeling seasonal storage', 2018
            """

            return(m.state_of_charge[s,h] ==
                   m.state_of_charge_intra[s,h] + m.state_of_charge_inter[
                           s,network.cluster_ts['Candidate_day'][h]])                

        network.model.total_storage_constraint = po.Constraint(
                sus.index, network.snapshots, rule = total_state_of_charge)

        def state_of_charge_lower(m,s,h):
            """
            Define the state_of_charge as the sum of state_of_charge_inter 
            and state_of_charge_intra
            
            According to:
            L. Kotzur et al: 'Time series aggregation for energy system design: 
            Modeling seasonal storage', 2018
            """
             
          #  import pdb; pdb.set_trace()
          # Choose datetime of representive day
            date = str(network.snapshots[
                network.snapshots.dayofyear -1 ==
                network.cluster['RepresentativeDay'][h.dayofyear]][0]).split(' ')[0]
            
            hour = str(h).split(' ')[1]
            
            intra_hour = pd.to_datetime(date + ' ' + hour)

            return(m.state_of_charge_intra[s,intra_hour] + 
                   m.state_of_charge_inter[s,network.cluster_ts['Candidate_day'][h]]
                   * (1 - network.storage_units.at[s, 'standing_loss'])**24
                   >= 0)                

        network.model.state_of_charge_lower = po.Constraint(
                sus.index, network.cluster_ts.index, rule = state_of_charge_lower)
        
        
        network.model.del_component('state_of_charge_upper')
        network.model.del_component('state_of_charge_upper_index')
        network.model.del_component('state_of_charge_upper_index_0')
        network.model.del_component('state_of_charge_upper_index_1')


        def state_of_charge_upper(m,s,h):
            date = str(network.snapshots[
                network.snapshots.dayofyear -1 ==
                network.cluster['RepresentativeDay'][h.dayofyear]][0]).split(' ')[0]
            
            
            hour = str(h).split(' ')[1]
            
            intra_hour = pd.to_datetime(date + ' ' + hour)
            
            if network.storage_units.p_nom_extendable[s]:
                p_nom = m.storage_p_nom[s]
            else:
                p_nom = network.storage_units.p_nom[s]

            return (m.state_of_charge_intra[s,intra_hour] + 
                    m.state_of_charge_inter[s,network.cluster_ts['Candidate_day'][h]] 
                    * (1 - network.storage_units.at[s, 'standing_loss'])**24
                    <= p_nom * network.storage_units.at[s,'max_hours']) 
              
         
        network.model.state_of_charge_upper = po.Constraint(
             sus.index, network.cluster_ts.index,
             rule = state_of_charge_upper)


        def cyclic_state_of_charge(m,s):
            """
            Defines cyclic condition like pypsas 'state_of_charge_contraint'.
            There are small differences to original results.
            """
            
            last_day = network.cluster.index[-1]
            
            last_calc_hour = network.cluster['last_hour_RepresentativeDay'][last_day]
            
            last_inter = m.state_of_charge_inter[s, last_day]
            
            last_intra = m.state_of_charge_intra[s, last_calc_hour]
            
            first_day =  network.cluster.index[0]
            
            first_calc_hour = network.cluster['last_hour_RepresentativeDay'][first_day] - pd.DateOffset(hours=23)
            
            first_inter = m.state_of_charge_inter[s, first_day]
            
            first_intra = m.state_of_charge_intra[s, first_calc_hour]

            return  (first_intra + first_inter == \
                   (last_intra + last_inter
                   -(m.storage_p_dispatch[s,last_calc_hour]/ 
                           network.storage_units.at[s, 'efficiency_dispatch']
                           -m.storage_p_store[s,last_calc_hour] * 
                           network.storage_units.at[s, 'efficiency_store'])))

       # network.model.cyclic_storage_constraint = po.Constraint(
          #      sus.index,  rule = cyclic_state_of_charge)
        
def daily_bounds(network, snapshots):
    """ This will bound the storage level to 0.5 max_level every 24th hour.
    """
    
    sus = network.storage_units
    # take every first hour of the clustered days
    network.model.period_starts = network.snapshot_weightings.index[0::24]

    network.model.storages = sus.index

    def day_rule(m, s, p):
        """
        Sets the soc of the every first hour to the soc of the last hour
        of the day (i.e. + 23 hours)
        """
        return (
            m.state_of_charge[s, p] ==
            m.state_of_charge[s, p + pd.Timedelta(hours=23)])

    network.model.period_bound = po.Constraint(
        network.model.storages, network.model.period_starts, rule=day_rule)

       
####################################
def manipulate_storage_invest(network, costs=None, wacc=0.05, lifetime=15):
    # default: 4500 € / MW, high 300 €/MW
    crf = (1 / wacc) - (wacc / ((1 + wacc) ** lifetime))
    network.storage_units.capital_cost = costs / crf

def write_lpfile(network=None, path=None):
    network.model.write(path,
                        io_options={'symbolic_solver_labels': True})


def fix_storage_capacity(network, resultspath, n_clusters):  # "network" added
    path = resultspath.strip('daily')
    values = pd.read_csv(path + 'storage_capacity.csv')[n_clusters].values
    network.storage_units.p_nom_max = values
    network.storage_units.p_nom_min = values
    resultspath = 'compare-' + resultspath
