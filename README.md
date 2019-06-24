**Update 2019/06/24**: A model trained on 10% of the Shepard-Metzler dataset has been added, the following notebook explains the main features of this model: [nbviewer](https://nbviewer.jupyter.org/github/wohlert/generative-query-network-pytorch/blob/master/mental-rotation.ipynb)

# Generative Query Network

This is a PyTorch implementation of the Generative Query Network (GQN)
described in the DeepMind paper "Neural scene representation and
rendering" by Eslami et al. For an introduction to the model and problem
described in the paper look at the article by [DeepMind](https://deepmind.com/blog/neural-scene-representation-and-rendering/).

![](https://storage.googleapis.com/deepmind-live-cms/documents/gif_2.gif)

The current implementation generalises to any of the datasets described
in the paper. However, currently, *only the Shepard-Metzler dataset* has
been implemented. To use this dataset you can use the provided script in
```
sh scripts/data.sh data-dir batch-size
```

The model can be trained in full by in accordance to the paper by running the
file `run-gqn.py` or by using the provided training script
```
sh scripts/gpu.sh data-dir
```

## Implementation

The implementation shown in this repository consists of all of the
representation architectures described in the paper along with the
generative model that is similar to the one described in
"Towards conceptual compression" by Gregor et al.

Additionally, this repository also contains implementations of the **DRAW
model and the ConvolutionalDRAW** model both described by Gregor et al.

