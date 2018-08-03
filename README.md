# Generative Query Network

This is a PyTorch implementation of the Generative Query Network (GQN)
described in the DeepMind paper "Neural scene representation and
rendering" by Eslami et al. For an introduction to the model and problem
described in the paper look at the article by [DeepMind](https://deepmind.com/blog/neural-scene-representation-and-rendering/).

![](https://storage.googleapis.com/deepmind-live-cms/documents/gif_2.gif)

The current implementation generalises to any of the datasets described
in the paper. However, currently, *only the Shepard-Metzler dataset* has
been implemented. To use this dataset you must download the [tf-records
from DeepMind](https://github.com/deepmind/gqn-datasets) and convert them to PyTorch tensors,
such as by using the [gqn_datasets_translator](https://github.com/l3robot/gqn_datasets_translator).

The model can be trained in full by in accordance to the paper by running the
script `run-gqn.py`.

## Implementation

The implementation shown in this repository consists of all of the
representation architectures described in the paper along with the
generative model that is similar to the one described in 
"Towards conceptual compression" by Gregor et al.

Additionally, this repository also contains implementations of the **DRAW
model and the ConvolutionalDRAW** model both described by Gregor et al.

## Contributing

The best way to contribute to this project is to train the model as described
in the paper (by running `run-gqn.py`) and submitting a pull request with the 
fully trained model.

Currently, the repository contains a model `model-final.pt` that has only
been trained on a subset of the data.