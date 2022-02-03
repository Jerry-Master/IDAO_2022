# IDAO 2022

## Visualization

Here is presented the structures of one of the crystals. As you can see it is a perfect lattice except for some imperfections. It can have an atom missing of some atom is changed, in this case one atom has changed.

![crystal structure](https://github.com/Jerry-Master/IDAO_2022/blob/main/Visualizations/Crystal-structure.png?raw=true)

You can download the `embed.html` file in the Visualization folder to play with the interactive version.

## Environment

The environment should have the following packages:

* Numpy, Pandas, MatPlotLib, sklearn: The typical packages for data science.
* Tensorflow (CPU): The package for neural network.
* MegNet: The baseline model, it is a Graph Neural Network.
* NglView: The package used for visualizations.
* PyMatGen: The package used for loading the crystal structures.
* Keras Graph Neural Network: The library for creating graphs. More info below.

And the python version should be 3.9. In order to accomplish that a new environment can be created with `python3 -m venv crystal` and then by doing `pip3 install -r requirements.txt` the environment is set. You can also do it with `conda` and `virtualenv`. That is just how I did it.

In order to use the kernel inside jupyter you may have to install `ipykernel` for the environment. [Here](https://stackoverflow.com/questions/51934528/failed-to-start-the-kernel-on-jupyter-notebook) you can find how to do it. Or you can simply do `pip3 install ipykernel`. Also, in order to use nglview you may need to activate the related widget. It should suffice to do:

```
python3 -m ipykernel install --sys-prefix --name=crystal
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix nglview
```

The first line is to install the kernel into jupyter notebook and the others to enable the widget. If you have a previous version of crystal remove it by doing `jupyter kernelspec uninstall crystal` and install the new one.

## Density Functional Theory

The target of the dataset is computed following the [DFT theory](https://en.wikipedia.org/wiki/Density_functional_theory#Overview_of_method). In order to compute the band gap for new crystals one can use other [computational chemistry software](https://en.wikipedia.org/wiki/List_of_quantum_chemistry_and_solid-state_physics_software). Right now, [PySCF](https://pyscf.org/user/dft.html?highlight=band%20gap) seems to be the most versatile option to use with python.

## Graph Neural Networks
### Introduction

Since the crystal structure is represented as a graph, the natural way of working with this dataset is using GNNs. There are several types of GNNs, convolutional, recursive, and others. For the most part, the concepts are similar to normal neural networks but separating the input in nodes, edges and global states. Then the convolution nets consists basically in adding time-invariant shift transformations and the recurrent networks are simply using LSTM or GRU in addition to the graph representation of the input.

### Libraries

In order to code this models, and since we are using tensorflow, it seems logical to use the tensorflow graph neural network library. It was released very recently so the installation method is from source code only, it can be found [here](https://github.com/tensorflow/gnn). Apart from tensorflow, we can also use the keras library for GNNs. [Their github](https://github.com/aimat-lab/gcnn_keras) says that it can be installed through `pip install kgcnn`. And their documentation is [here](https://kgcnn.readthedocs.io/en/latest/implementation.html). It is not much but is what we have. 

The library from tensorflow and keras should be included in the crystal package in the drive. Right now, I was not able to include the `tensorflow_gnn` library due to some errors. I filed a GitHub issue to see if they solve them. But the `kgcnn` is included. You may need to install RDKit and OpenBabel. It can be done through Homebrew via `brew install rdkit` and `brew install open-babel`.

Several Graphs models have their [implementation](https://github.com/aimat-lab/gcnn_keras/tree/master/kgcnn/literature) in the [GitHub of the keras graph library](https://github.com/aimat-lab/gcnn_keras). Including another implementation of megnet which is the one we were given as baseline.

### Convolutional

A brief introduction to GCN can be found [here](https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b). The main takeaway from the article is how they normalize the adjacency matrix of the graph so that it can be used as a weight matrix. That concept is later used in the article explaining recurrent networks. 

If you want to dive into more details [here](https://github.com/Jerry-Master/IDAO_2022/blob/main/Papers/Convolutional%20GNN.pdf) is a paper explaining how they work and how they are related to the CNNs we all know. The article is more technical than the webpage. By looking at the formulas one can see the similarities between CNNs and GCNs. The key is to express the convolution as an operator including a time-invariant or shift operation. In 2D images it is a scalar product between filter and part of the image. Here is a scalar product between filter and the features of some nodes, weighted by some matrix resembling the adjacency matrix. 

### Recursive

The concept of Recurrent Graph Neural Network is well explained [in this arxiv-vanity page](https://www.arxiv-vanity.com/papers/1904.08035/). The update formulas for vanilla RNN, LSTM and GRU are presented to give background into the concept of recurrent networks. However, some formulas are badly parsed there, if some formula seems wrong check it [here](https://github.com/Jerry-Master/IDAO_2022/blob/main/Papers/Recurrent%20GNN%20(vs%20residual).pdf), for instance, formula (14) and (20) are wrongly written in the arxiv-vanity but are well written in the arxiv paper. The article also states that the use of recurrent units prevents overfitting better than residual connections and provides better results for deeper models.

If you prefer watching lectures to reading essays, [this page](https://gnn.seas.upenn.edu/lecture-11/) has a series of short videos explaining recurrent graph neural networks. One interesting video is that explaining the stability theorem for this type of networks. In normal GNNs there are several stability theorems, which are described [here](https://github.com/Jerry-Master/IDAO_2022/blob/main/Papers/Stability%20of%20GNN.pdf). It is good to have some similar results for GRNN.

For a more simplified version to try, there is [this](https://www.esann.org/sites/default/files/proceedings/2020/ES2020-107.pdf) article showing state-of-the-art results (in 2020) with even linear activation functions. Their code is publicly available in their [GitHub](https://github.com/lpasa/RecurrentDGNN). It uses the library PyTorch Geometric to work with graphs, and they tried the model with bioinformatics datasets.

Another R-GNN is used in a Reddit dataset of posts and comments to detect rumors. The article is in [Arxiv](https://arxiv.org/pdf/2108.03548.pdf) and the code on [GitHub](https://github.com/google-research/social_cascades). This model uses the RNN subclass from PyTorch to construct the model together with graph convolutions.

### Miscellaneous

While reading the above texts, I found yet another architecture of graphs neural networks: [Graph Attention Networks](https://www.arxiv-vanity.com/papers/1710.10903/). The concept is not so new. It consists of the well-known attention layer that is widely employed in multiple instance learning problems. Here the task of the attention layer is to create coefficients for each edge of each node that are used to weight the effect of the neighbouring nodes. The formula is the same as a normal attention layer, a fully connected layer and then a softmax. The key difference is that the coefficients are only computed for nodes that exists in a neighbourhood of the initial node. Thus taking into account the structure of the graph. Again, the code is publicly available in [their GitHub](https://github.com/PetarV-/GAT). This time it is implemented in tensorflow and the input is processed with the networkX library.

In many articles it is mentioned the Laplacian matrix. Taken from [its wikipedia page](https://en.wikipedia.org/wiki/Laplacian_matrix), the laplacian is defined as the degree matrix minus the adjacency matrix for simple graphs. For more complex analysis it can be normalised and it can take into consideration the weights of the edges.
