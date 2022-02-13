# Data

The data is separated in several folders and some additional CSV. Below is the explanation of what is inside each folder and a brief description of each CSV.

## Structures

The data is originally given as JSON files which contains the information of the crystal in a format readable by the pymatgen library. It contains the global state (charge), the lattice (as 3 vectors), and information about each atom (location and atomic number). The main problem with using the data as it is, is that reading the data is slow. If we plan on doing cross-validation and extensive research on the subject we need a way of reading faster the data. 

## Images

This is the way of reading faster the data. An important feature of all crystals is that they all live in the same 8 by 8 by 3 isometric grid. Therefore, we can represent the data as 8 by 8 image with 3 channels. Where the pixel value is the atomic number except when there is no atom, which is a 0. There is no more information needed. This representation is a bijection between the pymatgen representation and 8 by 8 images. Because the charge is an irrelevant parameter, and the lattices are all the same. One could say that there is a piece of information that is lost: the edges or connections between atoms. But since they follow a clear pattern the edge can be reconstructed if needed easily. Remember, the purpose of this transformation is to read the data faster. Even if the edges are not represented, for the most part the atom locations are enough. Furthermore, we save the images in npy format which is the fastest way to work with images. Here is an example of an image:

![crystal image](https://github.com/Jerry-Master/IDAO_2022/blob/main/Data/img-example.png?raw=true)

## Distances

In order to apply K-nearest neighbour or to do some kind of multidimensional scaling one needs a distance matrix. Our first approach was to compute the distance between the images using techniques from video compression. In video compression consecutive fotograms are predicted from previous fotograms so that it is only needed to compress the difference.  The same approach can be modified to instead compute distances. One way to do so is to compute the size in bits of the difference. But that is overly complicated. Instead, we decided to modify the movement compensation module of some MPEG. We computed the movement vectors between each defect in the crystal and summed up their modulus. Since we had nearly 3000 points, that leaves several millions of distances to compute. In order to do it efficiently we parallelized the task using the 10 CPU kernels that kaggle allows and the library multiprocess to use the whole power of each CPU kernel. In the folder distances are the distances from each file to each training file. In the private dataset the distances are also computed with respect to the training.

## Coordinates

Another remarkable feature of the crystal is more than 90% of the crystals has exactly 3 imperfections. That creates the possibility to reduce even more the dataset. Because now there is a bijection between each crystal and 3 points in space. Which means that the problem is effectively one of classifying triangles. In the folder coordinates are the coordinates of each crystal defect in npy format.

## CSV

Right now there are 4 CSV. One given by the organization: targets.csv. And three created by us. The one given are the targets that we need to predict for the public dataset. They are values of the band gap of each crystal. We made an interesting analysis of the target reaching the conclusion that there are three main types of crystals, and within group, little variations are what make the difference in the band gap. This is the target histogram:

![target histogram](https://github.com/Jerry-Master/IDAO_2022/blob/main/Data/target-hist.png?raw=true)

Each bin is separated from other in at least 0.02 eV, which means that for the metric we need to optimize there are effectively 86 bins / classes to predict. Also, they can be grouped in 3 easily as one can see. For the problem of classifying in 3, it is possible to do it with more than 95% accuracy. However, doing it for the 86 classes is more difficult. The distance approach has only given a maximum of 69% accuracy. Which is better than the baseline but not enough.

The other CSV has similar information as the folders, but are condensed in one single file:

* {Train, Test}\_distance\_matrix.csv: As the name says, these are the distance matrices between train and train, and between test and train.
* coordinates_train.csv: This file has a dataframe with the coordinates of each defect in the crystal for each crystal. Due to a problem in conversion, the data is as a `string` and not as `np.array` or a `list`.
* coordinates_train_angles.csv: Since each crystal has 3 vertices, that represents a triangle, which has 3 angles. Therefore, it can be represented as 3 coordinates in angles which add up to 180 degrees. This files contains those coordinates in radians.

