#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:34:01 2019

@author: kathiesterl
"""

import os
from os import path, listdir
from matplotlib import pyplot as plt
import pandas as pd
from numpy import genfromtxt

kmean = [50] #genfromtxt('C:\eTraGo\etrago\k_mean_parameter.csv')

abs_err = {}
rel_err = {}
abs_time = {}
rel_time = {}
benchmark_time={}
benchmark_objective={}
ks=[]
abs_su_expansion={}
rel_su_expansion={}
abs_net_expansion= {}
rel_net_expansion = {}
#home = os.path.expanduser('/home/clara/000')

#receive information from the results of the calculation 
for i in kmean:
    i =int(i)
    
    resultspath = '/home/clara/Schreibtisch/snapshot_clustering_analysis/120days'
    clustered_path = path.join(resultspath, 'daily')
    original_path = path.join(resultspath, 'original')

    network = pd.read_csv(path.join(original_path, 'network.csv'))
    storage_units = pd.read_csv(path.join(original_path, 'storage_units.csv'))
    storage_expansion = storage_units.p_nom_opt[storage_units.p_nom_extendable].sum()
    lines = pd.read_csv(path.join(original_path, 'lines.csv'))
    links = pd.read_csv(path.join(original_path, 'links.csv'))
    
    network_expansion = (lines.s_nom_opt[lines.s_nom_extendable]-lines.s_nom_min[lines.s_nom_extendable]).sum()+\
        (links.p_nom_opt[links.p_nom_extendable]-links.p_nom_min[links.p_nom_extendable]).sum()
        
    for c in listdir(clustered_path): # go through the snapshot_parameters
        if c != 'Z.csv': 
            network_c = pd.read_csv(path.join(clustered_path, c, 'network.csv'))
            storage_units_c = pd.read_csv(path.join(clustered_path, c, 'storage_units.csv'))
            storage_expansion_c = storage_units_c.p_nom_opt[storage_units_c.p_nom_extendable].sum()

            lines_c = pd.read_csv(path.join(clustered_path, c, 'lines.csv'))
            links_c = pd.read_csv(path.join(clustered_path, c,'links.csv'))
    
            network_expansion_c = (lines_c.s_nom_opt[lines_c.s_nom_extendable]-lines_c.s_nom_min[lines_c.s_nom_extendable]).sum()+\
                (links_c.p_nom_opt[links_c.p_nom_extendable]-links_c.p_nom_min[links_c.p_nom_extendable]).sum()

            abs_err[str(c)] = network_c['objective'].values[0]
            rel_err[str(c)] = ((abs(network['objective'].values[0] -
                                network_c['objective'].values[0])) / \
                                network['objective'].values[0] * 100)
            abs_time[str(c)] = float(network_c['time'])
            rel_time[str(c)] = (float(network_c['time']) /
                                float(network['time']) * 100)
            benchmark_time[str(c)] = float(network['time'])
            benchmark_objective[str(c)] = network['objective'].values[0]
            abs_su_expansion[str(c)] = storage_expansion_c
            rel_su_expansion[str(c)] = ((abs(storage_expansion -
                                storage_expansion_c)) /
                                storage_expansion * 100)
            
            abs_net_expansion[str(c)] = network_expansion_c
            rel_net_expansion[str(c)] = ((abs(network_expansion -
                                network_expansion_c)) /
                                network_expansion * 100)
    #create a dataframe with the needed results for each kmean        
    results = pd.DataFrame({
                        '1_obj_abs': abs_err,
                        '2_obj_rel': rel_err,
                        '3_obj_benchmark': benchmark_objective,
                        '4_time_abs': abs_time,
                        '5_time_rel': rel_time,
                        '6_time_benchmark':benchmark_time,
                        '7_abs_storage_expansion': abs_su_expansion,
                        '8_rel_storage_expansion': rel_su_expansion,
                        '9_abs_network_expansion': abs_net_expansion,
                        '10_rel_network_expansion': rel_net_expansion
                        })
    results.index = [int(i) for i in results.index]
    results.sort_index(inplace=True)
    
    #save the dataframe for each kmean
    results.to_csv('kmean'+str(i)+'.csv')
            
#definition of 2 axis plots
def plot_2d(variable, name):    
    fig, ax = plt.subplots()
    #get data for the plots
    for i in kmean:
        data = results.from_csv('kmean'+str(int(i))+'.csv', parse_dates=False)
        data = data.iloc[0:len(data)][variable]
        ax.plot(data,'--*',label='kmean '+str(int(i)).format(i=i))
    
    ax.plot()
    ax.set_title("Clustering Analysis")
    ax.set_xlabel("snapshot parameters")
    ax.set_ylabel(name)
    plt.legend(loc='best')
    plt.show()
    fig.savefig(path.join(resultspath , 'Analysis_2d,'+ name + '.png'))

 #plotting time and objective function
"""plot_2d('4_time_abs',name='absolute time in s')
plot_2d('5_time_rel',name='relative time deviation in %')

plot_2d('1_obj_abs',name='absolute objective function')
plot_2d('2_obj_rel',name='relative objective function deviation in %')"""


plot_2d('7_abs_storage_expansion',name='absolute storage expansion in MW')
plot_2d('8_rel_storage_expansion',name='relative storage expansion deviation in %')

plot_2d('9_abs_network_expansion',name='absolute network expansion in MW')
plot_2d('10_rel_network_expansion',name='relative network expansion deviation in %')