# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:30:25 2020

@author: Ming Jin
"""

import numpy as np
import torch
import math
from sklearn.metrics import average_precision_score, roc_auc_score


############################## Neighbor Finder ###############################
class NeighborFinder:
  """
  INIT INPUTS:
     adj_list: A list of shape [max_node_idx, 1] in this format: [[src_node/dest_node, edge_idx to dest_node/src_node, timestamp]]
     uniform: Bool, if Ture then we randomly sample n_neighbors before the cut_time
     seed: random seed for
  """
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []  # neighbor ids
    self.node_to_edge_idxs = []  # corresponding edge idx
    self.node_to_edge_timestamps = []  # corresponding timestamp

    for neighbors in adj_list:
      # neighbors is a tuple: (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for src_idx in the overall interaction graph. 
    The returned interactions are sorted by time.
    3 lists will be returned: 
        node_to_neighbors: List of length [temporal_neighbors_before_cut_time]
        node_to_edge_idxs: List of length [temporal_neighbors_before_cut_time]
        node_to_edge_timestamps: List of length [temporal_neighbors_before_cut_time]
    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list source_nodes and correspond cut times (i.e. their current timestamps),
    this method extracts a list of sampled temporal neighbors of each node in source_nodes.
    
    INPUTS:
        source_nodes: A list (int) of nodes which temporal neighbors need to be extracted
        timestamps: A list (float) of timestamps for nodes in source_nodes
        n_neighbors: Extract this number of neighbors between time range [0, timestamps]
        
    OUTPUTS:
        neighbors: Arrary of shape [source_nodes, n_neighbors]
        edge_idxs: Arrary of shape [source_nodes, n_neighbors]
        edge_times: Arrary of shape [source_nodes, n_neighbors]
        
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    
    # ALL interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      # extracts all neighbors, interactions (i.e. edges) indexes and timestamps of ALL interactions of source_nodes happening before their corresponding cut_time (i.e. timestamps)
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node, timestamp)

      if len(source_neighbors) > 0 and n_neighbors > 0:
        
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
        
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)  # random sample n_neighbors temporal neighbors

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time cuz provided source_nodes are not sorted yet
          # so that neighbors, edge_times, and edge_idxs are all sorted by time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
          
        else:
            
          # Take most recent n_neighbors interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times


def get_neighbor_finder(data, uniform, max_node_idx=None):
    
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


############################ MLP Score function ##############################
class MergeLayer(torch.nn.Module):
    """
    Compute probability on an edge given two node embeddings
    
    INIT INPUTS:
        dim1 = dim2 = dim3 = node_feat_dim, emb_dim = node_feat_dim
        dim4 = 1
    
    INPUTS:
        x1: torch.cat([source_node_embedding, source_node_embedding], dim=0) with shape [batch_size * 2, emb_dim]
        x2: torch.cat([destination_node_embedding, negative_node_embedding]) with shape [batch_size * 2, emb_dim]
    """
    def __init__(self, dim1, dim2, dim3, dim4):
      super().__init__()
      self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
      self.fc2 = torch.nn.Linear(dim3, dim4)
      self.act = torch.nn.ReLU()
    
      torch.nn.init.xavier_normal_(self.fc1.weight)
      torch.nn.init.xavier_normal_(self.fc2.weight)
    
    def forward(self, x1, x2):
      x = torch.cat([x1, x2], dim=1)  # [batch_size * 2, emb_dim * 2]
      h = self.act(self.fc1(x))  # [batch_size * 2, emb_dim]
      return self.fc2(h)  # [batch_size * 2, 1]
  
############################ Negative Simpler ##############################
class RandEdgeSampler(object):
    """
    Negative simpler to randomly simple negatives from provided src and dest list
    
    INIT INPUTS:
        src_list: List of node ids
        dst_list: List of node ids
        
    INPUTS:
        size: How many negatives to sample
    """
    def __init__(self, src_list, dst_list, seed=None):
      self.seed = None
      self.src_list = np.unique(src_list)
      self.dst_list = np.unique(dst_list)
    
      if seed is not None:
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
    
    def sample(self, size):
      if self.seed is None:
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
      else:
        src_index = self.random_state.randint(0, len(self.src_list), size)
        dst_index = self.random_state.randint(0, len(self.dst_list), size)
      return self.src_list[src_index], self.dst_list[dst_index]
    
    def reset_random_state(self):
      self.random_state = np.random.RandomState(self.seed)

############################ EarlyStopMonitor ##############################
class EarlyStopMonitor(object):
    
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
      
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
      
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round

########################## Evaluate on val & test ##########################
def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)