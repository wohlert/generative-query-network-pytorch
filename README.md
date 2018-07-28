# Generative Query Network

This is a PyTorch implementation of the Generative Query Network (GQN)
described in the DeepMind paper "Neural scene representation and
rendering" by Eslami et al. For an introduction to the model and problem
described in the paper look at the article by [DeepMind](https://deepmind.com/blog/neural-scene-representation-and-rendering/).

The current implementation generalises to any of the datasets described
in the paper. However, currently, *only the Shepard-Metzler dataset* has
been implemented. To use this dataset you must download the [tf-records
from DeepMind](https://github.com/deepmind/gqn-datasets) and convert them to PyTorch tensors.

## Implementation

The implementation shown in this repository consists of the `tower`
representation architecture along with the generative model that is
similar to the one described in "Towards conceptual compression" by
Gregor et al.

## Results

Currently, the results are pending as the model is very computationally
costly to train for the datasets described in the paper.
