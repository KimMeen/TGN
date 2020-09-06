# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:35:26 2020

@author: Ming Jin
"""

import numpy as np
import random
import pandas as pd


class Data:
    
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data(dataset_name, different_new_nodes_between_val_and_test=False):
    """
    INPUTS: 
        dataset_name: Wikipedia or Reddit
        different_new_nodes_between_val_and_test: Val and test set will use different unseen nodes (to test inductiveness)
        
    OUTPUTS: 
        node_features: Array of shape [n_nodes, node_feat_dim], node_feat_dim is fixed to 172
        edge_features: Array of shape [n_interactions, edge_feat_dim]
        full_data: Data instance; It contains interactions of the whole temporal graph (i.e. acrossing the entire timespan)
        train_data: Data instance; It contains interactions happening before the validation time which do not involve any new node used for inductiveness
        val_data:  Data instance; It contains interactions after training time but before the testing time, this setting may contain nodes in train_data (transductive setting)  
        test_data: Similar to val_data, this setting may contain nodes in train_data (transductive setting)
        new_node_val_data: Inductive val_data with edges that at least have one unseen node (inductive setting)
        new_node_test_data: Inductive test_data with edges that at least have one unseen node (inductive setting)
        
    P.S. 70%-15%-15% data split ratio applied
    """
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
    node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))
    
    # val and test splite timestamp: 70%-15%-15%
    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))  # return two float numbers
    
    # list of length n_interactions, which may contain duplicate nodes
    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values
    
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    
    random.seed(2020)
    
    node_set = set(sources) | set(destinations)  # set of all nodes (no duplications)
    n_total_unique_nodes = len(node_set)  # notice: set() will remove duplications
    
    # Compute nodes which appear at val & test time
    test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
    
    # Sample (10% * n_nodes) nodes from val & test nodes to be unseen nodes
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))
    
    # Mask saying for each source and destination whether they are unseen nodes
    # Two lists of length n_interactions where True for element (i.e. interaction) if src_node or dest_node belongs to new_test_node_set (i.e. unseen nodes)
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    
    # A list of length n_interaction where True for element (i.e. interaction) if both src_node and dest_node are not unseen nodes
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
    
    # For train we keep edges happening before the validation time && do not involve any unseen node
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask], edge_idxs[train_mask], labels[train_mask])
    
    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    
    # define the new nodes sets for testing inductiveness of the model
    new_node_set = node_set - train_node_set
    
    # # TODO: Their relationships
    # print(len(node_set))
    # print(len(test_node_set))
    # print(len(new_test_node_set))
    # print(len(train_node_set))
    # print(len(new_node_set))

    # exit()
    
    # val and test mask where we don't consider unseen nodes issue
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time
    
    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask], edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask], edge_idxs[test_mask], labels[test_mask])
    
    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])
        # one of each (src or dest) contains unseen nodes then True
        edge_contains_new_val_node_mask = np.array([(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array([(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    
    else:
        # one of each (src or dest) contains unseen nodes then True
        edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)
    
    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask], 
                             timestamps[new_node_val_mask], edge_idxs[new_node_val_mask], 
                             labels[new_node_val_mask])
    
    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])
    
    print("The dataset has {} interactions, involving {} different unique nodes".format(full_data.n_interactions, full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different unique nodes".format(train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different unique nodes".format(val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different unique nodes".format(test_data.n_interactions, test_data.n_unique_nodes))
    print("The inductive validation dataset has {} interactions, involving {} different unique nodes".format(new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("The inductive test dataset has {} interactions, involving {} different unique nodes".format(new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))
    
    return node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def compute_time_statistics(sources, destinations, timestamps):
    
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
      source_id = sources[k]
      dest_id = destinations[k]
      c_timestamp = timestamps[k]
      if source_id not in last_timestamp_sources.keys():
        last_timestamp_sources[source_id] = 0
      if dest_id not in last_timestamp_dst.keys():
        last_timestamp_dst[dest_id] = 0
      all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
      all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
      last_timestamp_sources[source_id] = c_timestamp
      last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)
    
    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst