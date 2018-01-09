# Computational graph

This repo is a Python 3 implementation of a computational graph to do deep learning. I learned in that all big deep learning librairies used that principle so I wanted to understand it and to do that I implemented it.
## Getting Started

Simply clone this repo

### Prerequisites

Essential : Numpy

```
pip install numpy
```
Optional : Graphviz

### Example

Run fully, fully_mnist, conv or conv_mnist in the terminal

This is the supposed output of the fully file.

This image represent the dataset used, it is a 2d dataset with two classes in a XOR position with a little bit of noise so that perfect separation if impossible

![dataset](https://user-images.githubusercontent.com/6108674/34733441-aad5b384-f568-11e7-96e2-461ee43fec9d.PNG)

This image represent the graph built 
![graph](https://user-images.githubusercontent.com/6108674/34733442-aaf0e3fc-f568-11e7-88e1-7d5aa8fdaec1.PNG)

This image represent the exectution of the program in the terminal
![cmd](https://user-images.githubusercontent.com/6108674/34733440-aab8071c-f568-11e7-8bfc-10d1c08d9fd7.PNG)

This image represent the loss of the graph at each iteration
![loss](https://user-images.githubusercontent.com/6108674/34733444-ab06512e-f568-11e7-8f76-df068a4ce192.PNG)

This image represent the resulting decision boundary
![boudary](https://user-images.githubusercontent.com/6108674/34733439-aa9dffac-f568-11e7-8fba-9d855cc708b7.PNG)



## Built With

* [Numpy](http://www.numpy.org) - NumPy is the fundamental package for scientific computing with Python
* [Graphviz](https://graphviz.gitlab.io) - Graph Visualization Software


## Authors

* **Jean-Gabriel Simard** 


## Acknowledgments

* Jeremy Fix - CentraleSupélec Teacher
