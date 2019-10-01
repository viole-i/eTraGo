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

home = os.path.expanduser('/home/clara/000')

#receive information from the results of the calculation 
for i in kmean:
    i =int(i)
    
    resultspath = os.path.join(home, 'Beispielrechnung')
    clustered_path = path.join(resultspath, 'daily')
    original_path = path.join(resultspath, 'original')

    network = pd.read_csv(path.join(original_path, 'network.csv'))
    
    for c in listdir(clustered_path): # go through the snapshot_parameters
        if c != 'Z.csv': 
            network_c = pd.read_csv(path.join(clustered_path, c, 'network.csv'))
    
            abs_err[str(c)] = network_c['objective'].values[0]
            rel_err[str(c)] = ((abs(network['objective'].values[0] -
                                network_c['objective'].values[0])) / 
                                network['objective'].values[0] * 100)
            abs_time[str(c)] = float(network_c['time'])
            rel_time[str(c)] = (float(network_c['time']) /
                                float(network['time']) * 100)
            benchmark_time[str(c)] = float(network['time'])
            benchmark_objective[str(c)] = network['objective'].values[0]
    
    #create a dataframe with the needed results for each kmean        
    results = pd.DataFrame({
                        '1_obj_abs': abs_err,
                        '2_obj_rel': rel_err,
                        '3_obj_benchmark': benchmark_objective,
                        '4_time_abs': abs_time,
                        '5_time_rel': rel_time,
                        '6_time_benchmark':benchmark_time})
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
    fig.savefig(path.join(home, 'Analysis_2d,'+ name + '.png'))

 #plotting time and objective function
plot_2d('4_time_abs',name='absolute time in s')
plot_2d('5_time_rel',name='relative time deviation in %')

plot_2d('1_obj_abs',name='absolute objective function')
plot_2d('2_obj_rel',name='relative objective function deviation in %')