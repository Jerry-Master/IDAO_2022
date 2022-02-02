# IDAO 2022

## Visualization

Here is presented the structures of one of the crystals. As you can see it is a perfect lattice except for some imperfections. It can have an atom missing of some atom is changed, in this case one atom has changed.

![crystal structure](https://github.com/Jerry-Master/IDAO_2022/blob/main/Visualizations/Crystal-structure.png?raw=true)

You can download the `embed.html` file in the Visualization folder to play with the interactive version.

## Environment

There are installed the following packages, and all of their dependencies:

* Numpy, Pandas, MatPlotLib: The typical packages for data science.
* Tensorflow (CPU): The package for neural network.
* MegNet: The baseline model, it is a Graph Neural Network.
* NglView: The package used for visualizations.
* PyMatGen: The package used for loading the crystal structures.

The environment can be found [here](https://drive.google.com/file/d/1-SYXqPs_uHnLoopT8Awx703u5Je86Jl2/view?usp=sharing).

## Density Functional Theory

The target of the dataset is computed following the [DFT theory](https://en.wikipedia.org/wiki/Density_functional_theory#Overview_of_method). In order to compute the band gap for new crystals one can use other [computational chemistry software](https://en.wikipedia.org/wiki/List_of_quantum_chemistry_and_solid-state_physics_software). Right now, [PySCF](https://pyscf.org/user/dft.html?highlight=band%20gap) seems to be the most versatile option to use with python.

## Graph Neural Networks

Since the crystal structure is represented as a graph, the natural way of working with this dataset is using GNNs. There are several types of GNNs, convolutional, recursive, and others. A brief introduction to GCN can be found [here](https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b). If you want to dive into more details [here](https://github.com/Jerry-Master/IDAO_2022/blob/main/Papers/Convolutional%20GNN.pdf) is a paper explaining in more detail how they work and how they are related to the CNNs we all know.

In order to code this models, and since we are using tensorflow, it seems logical to use the tensorflow graph neural network library. It was released very recently so the installation method is from source code only, it can be found [here](https://github.com/tensorflow/gnn). Apart from tensorflow, we can also use the keras library for GNNs. [Their github](https://github.com/aimat-lab/gcnn_keras) says that it can be installed through `pip install kgcnn`. And their documentation is [here](https://kgcnn.readthedocs.io/en/latest/implementation.html). It is not much but is what we have. 
