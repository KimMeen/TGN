# TGN
A PyTorch annotated replication of [twitter-research](https://github.com/twitter-research)/**[tgn](https://github.com/twitter-research/tgn)** 

Paper: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637)

## Requirements

Python >= 3.6

```
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

## Preprocess datasets

#### Download the public data

Download the sample datasets (eg. wikipedia and reddit) from [here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named `./data`

#### Preprocess the data

We use the dense `npy` format to save the features in binary format. If edge features or nodes features are absent, they will be replaced by a vector of zeros.

```
python utils/preprocess_data.py --data wikipedia
python utils/preprocess_data.py --data reddit
```

## Model training

Self-supervised learning using the link prediction task:

```
# TGN-attn: self-supervised learning on the wikipedia dataset
python link_prediction.py --data wikipedia --embedding_module graph_sum --use_memory --memory_update_at_start

# TGN-attn-reddit: self-supervised learning on the reddit dataset
python link_prediction.py --data reddit --embedding_module graph_sum --use_memory --memory_update_at_start
```

** Check more commands with `--help`

## TODOs

- Add code for training on the downstream node-classification task (semi-supervised setting)

## Cite the paper

```
@inproceedings{tgn_icml_grl2020,
    title={Temporal Graph Networks for Deep Learning on Dynamic Graphs},
    author={Emanuele Rossi and Ben Chamberlain and Fabrizio Frasca and Davide Eynard and Federico 
    Monti and Michael Bronstein},
    booktitle={ICML 2020 Workshop on Graph Representation Learning},
    year={2020}
}
```