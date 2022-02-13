# Data

The data is separated in several folders and some additional CSV. Below is the explanation of what is inside each folder and a brief description of each CSV.

## Structures

The data is originally given as JSON files which contains the information of the crystal in a format readable by the pymatgen library. It contains the global state (charge), the lattice (as 3 vectors), and information about each atom (location and atomic number). The main problem with using the data as it is, is that reading the data is slow. If we plan on doing cross-validation and extensive research on the subject we need a way of reading faster the data. 

## Images

This is the way of reading faster the data. An important feature of all crystals is that they all live in the same 8 by 8 by 3 isometric grid. Therefore, we can represent the data as 8 by 8 image with 3 channels. Where the pixel value is the atomic number except when there is no atom, which is a 0. There is no more information needed. This representation is a bijection between the pymatgen representation and 8 by 8 images. Because the charge is an irrelevant parameter (we need to check this), and the lattices are all the same. One could say that there is a piece of information that is lost: the edges or connections between atoms. But since they follow a clear pattern the edge can be reconstructed if needed easily. Remember, the purpose of this transformation is to read the data faster. Even if the edges are not represented, for the most part the atom locations are enough. Here is an example of an image:

![crystal image](https://github.com/Jerry-Master/IDAO_2022/blob/main/Data/img-example.png?raw=true)

## Distances

In order to apply K-nearest neighbour or to do some kind of multidimensional scaling one needs a distance matrix. Our first approach was to compute the distance between the images using techniques from video compression. In video compression consecutive fotograms are predicted from previous fotograms so that it is only needed to compress the difference.  The same approach can be modified to instead compute distances. One way to do so is to compute the size in bits of the difference. But that is overly complicated. Instead, we decided to modify the movement compensation module of some MPEG. We computed the movement vectors between each defect in the crystal and summed up their modulus. Since we had nearly 3000 points, that leaves several millions of distances to compute. In order to do it efficiently we parallelized the task using the 10 CPU kernels that kaggle allows and the library multiprocess to use the whole power of each CPU kernel. In the folder distances are the distances from each file to each training file.

## Coordinates

## CSV
