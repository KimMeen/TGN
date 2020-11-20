# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:01:32 2020

@author: Ming Jin

There are four types of embedding calculation methods: Identity, Time projection, Temporal graph attention, Temporal graph sum

Providing a batch of nodes (i.e. source nodes), return the embeddings for those nodes
"""

import torch
from torch import nn
import numpy as np
import math

from modules.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
  """
  Abstract class for the embedding calculation
  
  INIT:
      node_features: Nodes raw features of shape [n_nodes, node_feat_dim]
      edge_features: Edges raw features of shape [n_interactinon, edge_feat_dim]
      neighbor_finder: NeighborFinder instance
      time_encoder: TimeEncoder instance encodes t to a vector of shape [n_time_features]
      n_layers: L in the paper, corresponding to L-hops as well
      n_node_features: Nodes raw feature dimension equals to node_feat_dim
      n_edge_features: Edges raw feature dimension euqals to edge_feat_dim
      n_time_features: Time encoding dimension equals to node_feat_dim
      embedding_dimension: Embedding dim for nodes, which equals to node_feat_dim
      device: 
      dropout: Not in codes
          
  INPUTS:
      memory: A Tensor of shape [n_nodes, mem_dim]; Memory.memory
      source_nodes: Array of shape [source_nodes]; Nodes in a batch in a certain time which embeddings to be calculated
      timestamps: Array of shape [source_nodes]; Timestamps of interactions (i.e. Current timestamps) for those nodes
      n_layers: A number; Indicating how many graph conv layers (i.e. How deep to aggregate neighbors' information)
      n_neighbors: A number; Indicating how many temporal neighbors to be considered in a certain hop
      time_diffs: A Tensor of shape [source_nodes]; Delta t, i.e. Differences between the time of a node was last updated (i.e. memory.last_update), 
                  and the time for which we want to compute the embedding of a node
      use_time_proj: Not in codes  
  """
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
      
    super(EmbeddingModule, self).__init__()
    
    self.node_features = node_features
    self.edge_features = edge_features
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    pass


class IdentityEmbedding(EmbeddingModule):
  """
  Identity embedding calculation: Z_i(t) = S_i(t)
  
  Node embedding of shape [source_nodes, mem_dim]
  """
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
  """
  Identity embedding calculation: Z_i(t) = S_i(t) * (1 + Linear(time_diff))
  
  Node embedding of shape [source_nodes, emb_dim]
  """    
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
      
    super(TimeEmbedding, self).__init__(node_features, edge_features,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)  # node_dim = 172 = mem_dim for wiki and reddit

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
      
      source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))
        
      return source_embeddings


class GraphEmbedding(EmbeddingModule):
  """
  THe second abstract class (Relationship: EmbeddingModule <-- GraphEmbedding <-- GraphSumEmbedding/GraphAttentionEmbedding)
  for the graph-based embedding calculation
  """
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
      
      
      super(GraphEmbedding, self).__init__(node_features, edge_features,
                                           neighbor_finder, time_encoder, n_layers,
                                           n_node_features, n_edge_features, n_time_features,
                                           embedding_dimension, device, dropout)

      self.use_memory = use_memory
      self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    """
    Recursive implementation of n_layers temporal graph embedding calculation.
    Finally we have h^n_layers (t), which is h^L (t) in paper.
    
    INPUTS: 
        same as EmbeddingModule
        
    OUTPUT:
        source_node_features (if n_layer==0): h^0 (t) of shape [source_nodes, node_feat_dim]
        OR
        source_embedding: h^1 (t) to h^n_layers (t), each of shape [source_nodes, emb_dim]
    """

    assert (n_layers >= 0)  # layers can't be negative

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)  # source node ids
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)  # timestamps of interactions for those nodes
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))  # Phi(0)
    source_node_features = self.node_features[source_nodes_torch, :]  # source node features of shape [source_nodes, node_feat_dim]

    # h^0 (t) = S(t) + V(t)
    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features  # node_dim = 172 = mem_dim for wiki and reddit

    if n_layers == 0:
      # h^0 (t)  
      return source_node_features
    else:
      # n_neighbors TEMPORAL neighbors of source_nodes, as well as the corrsponding edges and associated interaction timestamps
      # neighbors: Arrary of shape [source_nodes, n_neighbors]
      # edge_idxs: Arrary of shape [source_nodes, n_neighbors]
      # edge_times: Arrary of shape [source_nodes, n_neighbors]
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes, timestamps, n_neighbors=n_neighbors)

      # neighbors_torch, edge_idxs, edge_deltas_torch are what we need
      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
      edge_deltas = timestamps[:, np.newaxis] - edge_times  # This is t - t_N in the paper of shape [source_nodes, num_temp_neighbors]
      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      # Recrsively calculate L-hops neighbors' embeddings of source_nodes
      neighbors = neighbors.flatten()  # [source_nodes * num_temp_neighbors]
      # neighbor_embeddings of shape [len(neighbors), emb_dim]
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      # h^(l-1)_j (t) : neighbor_embeddings of shape [source_nodes, n_neighbors, emb_dim]
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      # Phi(t - t_j) : time_embeddings of shape [source_nodes, n_neighbors, n_time_features = node_feat_dim]
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)
      # e_ij : edge_features of shape [source_nodes, n_neighbors, n_edge_features = node_feat_dim]
      edge_features = self.edge_features[edge_idxs, :]
      
      # mask is a zeros tensor of shape [source_nodes, n_neighbors]
      # neighbors_torch will not be affected
      mask = neighbors_torch == 0
      
      # [source_nodes, emb_dim]
      source_embedding = self.aggregate(n_layers, source_node_features,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)
      # h^1 (t) to h^n_layers (t)
      return source_embedding

  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):
      """
      Sum or attention aggregation for the graph-based embedding calculation
      
      INPUTS:
          n_layers: L in the paper, i.e. L-hops
          source_node_features: h^0 (t) = S(t) + V(t)
          source_nodes_time_embedding: Phi(0) of shape [source_nodes, n_time_features]
          neighbor_embeddings: Neighbor embeddings of l-1 layer with shape [source_nodes, emb_dim]
          edge_time_embeddings: Phi(t - t_j) of shape [source_nodes, n_time_features]
          edge_features: e_ij (i.e. edge_features) of shape [source_nodes, n_neighbors, n_edge_features = node_feat_dim]
          mask: A zeros tensor of shape [source_nodes, n_neighbors]; Has been used by Graph Attn Module
      """
      return None


class GraphSumEmbedding(GraphEmbedding):
    
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
      
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    
    # mapping neighbors_features: [source_nodes, emb_dim + edge_feat_dim + time_feat_dim] --> [source_nodes, emb_dim]
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features + n_edge_features, 
                                                         embedding_dimension) for _ in range(n_layers)])
    
    self.linear_2 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_node_features + n_time_features, 
                                                         embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding, 
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):
    
    # neighbors_features: h_j ^ (l-1) (t) || e_ij || Phi(t - t_j)
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features], dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    # h^wave_i (t) in paper
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))
    
    # source_features: [source_nodes, n_node_features + n_time_features]
    source_features = torch.cat([source_node_features, source_nodes_time_embedding.squeeze()], dim=1)  # TODO: Why raw features?
    # [source_nodes, emd_dim + n_node_features + n_time_features]
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    # [source_nodes, emb_dim]
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
    
  def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
      
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(n_node_features=n_node_features,
                                                                        n_neighbors_features=n_node_features,
                                                                        n_edge_features=n_edge_features,
                                                                        time_dim=n_time_features,
                                                                        n_head=n_heads,
                                                                        dropout=dropout,
                                                                        output_dimension=n_node_features) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):
      
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    return source_embedding


'''
###############
REFERENCE ENTRY
###############
'''
def get_embedding_module(module_type, node_features, edge_features, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device, n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
    
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))