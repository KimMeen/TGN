# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 22:58:18 2020

@author: Ming Jin
"""

import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import GRUMemoryUpdater
from modules.embedding_module import get_embedding_module
from modules.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  """
  TGN model
  
  INIT INPUTS:
      neighbor_finder: NeighborFinder instance, it could be 'train_ngh_finder' or 'full_ngh_finder' on different graph scoope
      node_features: Nodes raw features of shape [n_nodes, node_feat_dim]
      edge_features: Edges raw features of shape [n_interactinon, edge_feat_dim]
      n_layers: L in the paper, corresponding to L-hops as well
      n_heads: Number of attention heads
      dropout: Not used in codes
      use_memory: Bool variable, whether to augment the model with a node memory
      memory_update_at_start: Bool variable, whether to update memory at the start of the batch
      message_dimension: Node message dimension for m_i(t), default 100
      memory_dimension: Node memory dimension for s_i(t), default 172
      embedding_module_type: How to calculate embedding, default 'graph_attention'
      message_function: How to calculate node message, default 'mlp'
      mean_time_shift_src: 
      std_time_shift_src:
      mean_time_shift_dst:
      std_time_shift_dst:
      n_neighbors: How many temporal neighbos to be extracted
      aggregator_type: How to aggregate messages, default 'last'
  """
    
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=True, memory_update_at_start=True, 
               message_dimension=100, memory_dimension=172, embedding_module_type="graph_attention",
               message_function="mlp", mean_time_shift_src=0, std_time_shift_src=1, 
               mean_time_shift_dst=0, std_time_shift_dst=1, n_neighbors=None, aggregator_type="last"):
      
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)  # node features to tensor
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)  # edge features to tensor

    self.n_node_features = self.node_raw_features.shape[1]  # node_feat_dim
    self.n_nodes = self.node_raw_features.shape[0]          # n_nodes
    self.n_edge_features = self.edge_raw_features.shape[1]  # edge_feat_dim
    self.embedding_dimension = self.n_node_features         # emb_dim = node_feat_dim
    self.n_neighbors = n_neighbors

    self.use_memory = use_memory

    self.time_encoder = TimeEncode(dimension=self.n_node_features)  # encodes time to shape [node_feat_dim]
    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
        
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + self.time_encoder.dimension  # raw message dim
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension      # message dim
      
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           # message_dimension=message_dimension,
                           device=device)
      
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)                 # message function
      
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type, device=device)  # message aggregator

      self.memory_updater = GRUMemoryUpdater(memory=self.memory,
                                             message_dimension=message_dimension,
                                             memory_dimension=self.memory_dimension, device=device)     # memory updator

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                  node_features=self.node_raw_features,
                                                  edge_features=self.edge_raw_features,
                                                  neighbor_finder=self.neighbor_finder,
                                                  time_encoder=self.time_encoder,
                                                  n_layers=self.n_layers,
                                                  n_node_features=self.n_node_features,
                                                  n_edge_features=self.n_edge_features,
                                                  n_time_features=self.n_node_features,
                                                  embedding_dimension=self.embedding_dimension,
                                                  device=self.device,
                                                  n_heads=n_heads, dropout=dropout,
                                                  use_memory=use_memory,
                                                  n_neighbors=self.n_neighbors)                        # embedding module

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features, self.n_node_features, 1)
    
    
  def get_updated_memory(self, nodes, messages):
    """
    Get (but not persist) updated nodes' memory by using messages (AGG-->MSG-->MEM, while in paper the order is MSG-->AGG-->MEM)
    
    INPUTS:
        nodes: A list of length n_nodes; Node ids
        message: A dictionary {node_id:[([message_1], timestamp_1), ([message_2], timestamp_2), ...]}; Messages in previous batch
        
    OUTPUTS:
        updated_memory: A tensor of shape [unique_nodes, memory_dimension]
        updated_last_update: A tensor of shape [unique_nodes]    
    """
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)
    
    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)
    
    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update


  def update_memory(self, nodes, messages):
    """
    Updated nodes' memory by using messages (AGG-->MSG-->MEM, while in paper the order is MSG-->AGG-->MEM)
    
    INPUTS:
        nodes: A list of length len(nodes); Node ids
        message: A dictionary {node_id:[([message_1], timestamp_1), ([message_2], timestamp_2), ...]}; Messages in previous batch
    """
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update nodes' memory with the aggregated messages
    # Notice: update_memory() updates with no returns
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)
 
    
  def get_raw_messages(self, source_nodes, destination_nodes, edge_times, edge_idxs):
    """
    Get source_nodes' raw messages m_raw(t) = {[S(t-1), e(t)], t}
    
    INPUTS:
       source_nodes: Array of shape [batch_size]; Nodes' raw message to be calculated
       destination_nodes: Array of shape [batch_size];
       edge_times: Array of shape [batch_size]; Timestamps of interactions (i.e. Current timestamps) for source_nodes
       edge_idxs: Array of shape [batch_size]; Index of interactions (at edge_times) for source_nodes
           
    OUTPUTS:
       unique_sources: Array of shape [unique source nodes]
       messages: A dictionary {node_id:[([message_1], timestamp_1), ([message_2], timestamp_2), ...]}
                 where [message_x] is [S_i(t-1), S_j(t-1), e_ij(t), Phi(t-(t-1))], timestamp_x is the timestamp for each message_x
    """  
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]  # e_ij(t), or e(t)
    source_memory = self.memory.get_memory(source_nodes)  # S_i(t-1)
    destination_memory = self.memory.get_memory(destination_nodes)  # S_j(t-1)
    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(source_nodes), -1)  # Phi(t-t^wave)
    
    source_message = torch.cat([source_memory, destination_memory, edge_features, source_time_delta_encoding], dim=1)
    
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages


  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.
    Corresponding to algorithm 1 and 2 in the paper.
    
    INPUTS:
        source_nodes: Array of shape [batch_size]; Source node ids.
        destination_nodes: Array of shape [batch_size]; Destination node ids
        negative_nodes: Array of shape [batch_size]; Ids of negative sampled destination
        edge_times: Array of shape [batch_size]; Timestamps of interactions (i.e. Current timestamps) for those nodes (i.e. src, dest, neg)
        edge_idxs: Array of shape [batch_size]; Index of interactions
        n_neighbors: A number of temporal neighbor to consider in each layer (i.e. Each hop)
        
    OUTPUTS: Temporal embeddings for sources, destinations and negatives
        source_node_embedding: A tensor of shape [source_nodes, emb_dim]
        destination_node_embedding: A tensor of shape [destination_nodes, emb_dim]
        negative_node_embedding: A tensor of shape [negative_nodes, emb_dim] 
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])  # all nodes
    positives = np.concatenate([source_nodes, destination_nodes])              # positive samples
    timestamps = np.concatenate([edge_times, edge_times, edge_times])          # (Current) Timestamps for those nodes (i.e. V_2(t_1) and V_2(t_2))

    memory = None
    time_diffs = None
    
    if self.use_memory:

        ### Line 5-7 in Algorithm 2: Update memory first with previous batch messages, and then calculate embeddings
        if self.memory_update_at_start:
          # update memory for ALL nodes with messages stored in previous batches
          memory, last_update = self.get_updated_memory(list(range(self.n_nodes)), self.memory.messages)
        ### Line 3.5 in Algorithm 1: Use previous batch memory and calculate embeddings
        else:
          memory = self.memory.get_memory(list(range(self.n_nodes)))
          last_update = self.memory.last_update
        
        # Compute differences between the time the memory of a node was last updated,
        # and the time for which we want to compute the embedding of a node
        source_time_diffs = torch.FloatTensor(edge_times).to(self.device) - last_update[source_nodes].float()
        source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
        destination_time_diffs = torch.FloatTensor(edge_times).to(self.device) - last_update[destination_nodes].float()
        destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
        negative_time_diffs = torch.FloatTensor(edge_times).to(self.device) - last_update[negative_nodes].float()
        negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
        
        # time_diffs, i.e. Delta t, is for TimeEmbedding method
        time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs], dim=0)

    # Compute the embeddings for [source_nodes, destination_nodes, negative_nodes]
    # If memory_update_at_start is True: Line 8 in algorithm 2; The procedure is same as Figure 2 (right) in the paper
    # If memory_update_at_start is False: Line 4 in algorithm 1; The procedure is same as Figure 2 (left) in the paper 
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if self.use_memory:
        
        ### Line 12 in algorithm 2: If memory_update_at_start, we persist the update to memory (i.e. S(t-1)) here
        if self.memory_update_at_start:
          # Persist the updates to the memory only for sources and destinations
          self.update_memory(positives, self.memory.messages)
          # Remove messages for the positives, we have already updated the memory using positives old message
          self.memory.clear_messages(positives)
        
        ### Line 7 in algorithm 1
        ### Line 11 in algorithm 2
        # get raw message on source nodes
        unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                      destination_nodes,
                                                                      edge_times, edge_idxs)
        # get raw message on destination nodes
        unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                                source_nodes,
                                                                                edge_times, edge_idxs)
        
        ### Line 11 in Algorithm 2: If memory_update_at_start, we then store the new raw message
        if self.memory_update_at_start:
           self.memory.store_raw_messages(unique_sources, source_id_to_messages)
           self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
        ### Line 7-9 in Algorithm 1: If not memory_update_at_start, we update memory here with new raw message 
        else:
          self.update_memory(unique_sources, source_id_to_messages)
          self.update_memory(unique_destinations, destination_id_to_messages)

    return source_node_embedding, destination_node_embedding, negative_node_embedding


  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20):
    """
    Line 5 in algorithm 1; Line 9 in algorithm 2
    
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    
    INPUTS:
        source_nodes: Array of shape [batch_size]; Source node ids.
        destination_nodes: Array of shape [batch_size]; Destination node ids.
        negative_nodes: Array of shape [batch_size]; Negative node ids.
        edge_times: Array of shape [batch_size]; Timestamps of interactions (i.e. Current timestamps) for those nodes (i.e. src, dest, neg)
        edge_idxs: Array of shape [batch_size]; Index of interactions
        n_neighbors: A number of temporal neighbor to consider in each layer (i.e. Each hop)
    
    OUTPUTS:
    Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
    
    # get node embeddings for all nodes first
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
    
    # then calculate the P_pos and P_neg
    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding, negative_node_embedding])).squeeze(dim=0)
    
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()


  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder