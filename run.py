#!/usr/bin/env python3

import json
import os
import subprocess

import joblib
import networkx as nx
import numpy as np
import pandas as pd

import src

filepath_inspections_raw = 'data/raw/inspections.csv'
filepath_portcalls_raw = 'data/raw/portcalls.csv'
filepath_flag_performance = 'data/raw/flag_performance.json'
filepath_inspections_cleaned = 'data/cleaned/inspections.pkl'
filepath_portcalls_cleaned = 'data/cleaned/portcalls.pkl'
filepath_un_locode_cleaned = 'data/cleaned/un-locode.pkl'
filepath_inspections_processed = 'data/processed/inspections.pkl'
filepath_portcalls_processed = 'data/processed/portcalls.pkl'
filepath_ships_network = 'models/ships_network.npy'
filepath_ships_classification = 'models/ships_classification.npy'
filepath_network = 'models/network.pkl'
filepath_folds = 'models/folds.pkl'
filepath_x = 'models/x.pkl'
filepath_y = 'models/y.pkl'
filepath_s = 'models/s.pkl'
filepath_outer_folds = 'models/outer_folds.pkl'
filepath_inner_folds = 'models/inner_folds.pkl'
filepath_performance_folds = 'models/performance_folds.pkl'

def run():
    src.logger.info('#0 Check installs')
    if not os.path.isfile('teexgraph/teexgraph'):
        subprocess.run(['make', 'install'])
    
    src.logger.info("#1 Import inspections.") 
    if not os.path.isfile(filepath_inspections_cleaned):
        inspections_cleaned = src.import_inspections(filepath_inspections_raw)
        inspections_cleaned.to_pickle(filepath_inspections_cleaned)
    
    src.logger.info('#2 Import portcalls.')
    if not os.path.isfile(filepath_portcalls_cleaned):
        with open(filepath_flag_performance) as file:
            flag_performance = json.load(file)
            
        portcalls_cleaned = src.import_portcalls(filepath_portcalls_raw, 
                                                 flag_performance)
        portcalls_cleaned.to_pickle(filepath_portcalls_cleaned)
    
    src.logger.info('#3 Import UN Locode.')
    if not os.path.isfile(filepath_un_locode_cleaned):
        un_locode_cleaned = src.import_un_locode(
            [f'data/raw/2021-1 UNLOCODE CodeListPart{i}.csv' for i in (1,2,3)])
        un_locode_cleaned.to_pickle(filepath_un_locode_cleaned)
    
    src.logger.info('#4 Process inspections.')
    if not os.path.isfile(filepath_inspections_processed):
        inspections_cleaned = pd.read_pickle(filepath_inspections_cleaned)
        inspections_processed = src.process_inspections(inspections_cleaned)
        inspections_processed.to_pickle(filepath_inspections_processed)     
    
    src.logger.info('#5 Process portcalls.')
    if not os.path.isfile(filepath_portcalls_processed):
        portcalls_cleaned = pd.read_pickle(filepath_portcalls_cleaned)
        portcalls_processed = src.process_portcalls(portcalls_cleaned)
        portcalls_processed.to_pickle(filepath_portcalls_processed)
    
    src.logger.info('#6 Divide ships into classification and network set.')
    ships_classification_exists = os.path.isfile(filepath_ships_classification)
    ships_network_exists = os.path.isfile(filepath_ships_network)
    if not (ships_classification_exists and ships_network_exists):
        portcalls_processed = pd.read_pickle(filepath_portcalls_processed)
        inspections_processed = pd.read_pickle(filepath_inspections_processed)
        ships_classification, ships_network = src.divide_ships(
            portcalls_processed, inspections_processed)
        np.save(filepath_ships_classification, ships_classification)
        np.save(filepath_ships_network, ships_network)
        
    src.logger.info('#7 Construct network.')
    if not os.path.isfile(filepath_network):
        ships_network = np.load(filepath_ships_network, allow_pickle=True)
        portcalls_processed = pd.read_pickle(filepath_portcalls_processed)
        portcalls_network = portcalls_processed.loc[
            lambda x: x['ship'].isin(ships_network)]
        network = src.construct_network(portcalls_network)
        nx.write_gpickle(network, filepath_network)        
    
    src.logger.info('#8 Get features.')
    if not os.path.isfile(filepath_x):
        portcalls_processed = pd.read_pickle(filepath_portcalls_processed)
        ships_network = np.load(filepath_ships_network, allow_pickle=True)
        portcalls_network = portcalls_processed.loc[
            lambda x: x['ship'].isin(ships_network)]
        ships_classification = np.load(filepath_ships_classification, 
                                       allow_pickle=True)
        portcalls_classification = portcalls_processed.loc[
            lambda x: x['ship'].isin(ships_classification)]
        network = nx.read_gpickle(filepath_network)
        X = src.get_features(portcalls_network, portcalls_classification, 
                             network)
        X.to_pickle(filepath_x)
        
    src.logger.info('#9 Get targets.')
    if not os.path.isfile(filepath_y):
        inspections_processed = pd.read_pickle(filepath_inspections_processed)
        ships_classification = np.load(filepath_ships_classification, 
                                       allow_pickle=True)
        inspections_classification = inspections_processed.loc[
            lambda x: x['IMO'].isin(ships_classification)]
        y = src.get_targets(inspections_classification, ships_classification)
        y.to_pickle(filepath_y)
        
    src.logger.info('#10 Get sensitive group.')
    if not os.path.isfile(filepath_s):
        portcalls_processed = pd.read_pickle(filepath_portcalls_processed)
        ships_classification = np.load(filepath_ships_classification, 
                                       allow_pickle=True)
        portcalls_classification = portcalls_processed.loc[
            lambda x: x['ship'].isin(ships_classification)]
        s = src.get_sensitive_group(portcalls_classification, 
                                    ships_classification)
        s.to_pickle(filepath_s)
        
    src.logger.info('#11 Get folds.')
    if not os.path.isfile(filepath_folds):
        x = pd.read_pickle(filepath_x)
        y = pd.read_pickle(filepath_y)
        s = pd.read_pickle(filepath_s)
        outer_folds, inner_folds = src.get_folds(x, y, s)
        joblib.dump(outer_folds, filepath_outer_folds)
        joblib.dump(inner_folds, filepath_inner_folds)
        
    src.logger.info('#12 Get performance over every fold.')
    X = pd.read_pickle(filepath_x)
    y = pd.read_pickle(filepath_y)
    s = pd.read_pickle(filepath_s)
    outer_folds = joblib.load(filepath_outer_folds)
    inner_folds = joblib.load(filepath_inner_folds)
    performance_folds = src.learn(X, y, s, outer_folds, inner_folds)
    performance_folds.to_pickle(filepath_performance_folds)
    
    
if __name__ == '__main__':
    run()