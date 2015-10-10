# RoboticaUVic-PR-week1
UVic Robotics Master. Pattern Recognition Homework 1

## Instructions



## Preliminaries

This and the next three homework assignments have to be completed
using Python. Python is a very easy to learn and flexible programming
language, with many libraries for the most diverse tasks. There are
currently two main flavours of Python that are not fully compatible:
version 2 (currently 2.7) and version 3 (currently 3.5). In this
course we will be using version 2.7 as the version of OpenCV shipped
with Ubuntu does not support the latter version.

For the homework in this course, we will use the following libraries, mostly from the SciPy family:
- [SciPy](http://www.scipy.org/) is a collection of libraries for scientific computing in Python.
- [Numpy](http://www.numpy.org/) provides the backbone functionality for numerical processing.
- [Matplotlib](http://matplotlib.org/) is a powerful plotting library.
- [Scikit Learn](http://scikit-learn.org/stable/) Is a simple and complete machine learning library for Python.
- [OpenCV](http://opencv.org/) is the reference computer vision and image processing library, with comprehensive Python bindings.

Besides these libraries, [IPython](http://ipython.org/) is recommended as an extended and more friendly interactive Python interpreter. It is also necessary to view the course notebooks.

In Ubuntu 14.04, all the required software can be installed with the following commands:

```bash
$ sudo apt-get install python-scipy python-numpy python-matplotlib \
  python-opencv python-sklearn
$ sudo apt-get install ipython
```

Learning Python is extraordinarily easy, especially if other
programming languages are already known. There are a few tutorials to
get up to speed in a few minutes:
- *Learn Python in 10 minutes* [tutorial](http://www.stavros.io/tutorials/python/) for beginners
- A very concise and recommendable **Python+Numpy** [tutorial](https://cs231n.github.io/python-numpy-tutorial/) by Justin Johnson
- Official Python 2 [tutorial](https://docs.python.org/2/tutorial/)
- OpenCV-Python [tutorials](https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_tutorials.html)

For this assignment, it is recommended to complete at least the **Python+Numpy** tutorial.

## Linear Regression 

See the ipython notebook from the class for reference.

0. [Download](http://archive.ics.uci.edu/ml/datasets/Housing) the Housing Data Set from the UCI repository.
1. Load the data (using *numpy.loadtxt*) and separate the last column (target value, MEDV). Compute the average of the target value and the MSE obtained using it as a constant prediction.
2. Split the data in two parts (50-50) for training and testing (first for training, second for testing). Train a linear regressor model for each variable individually (plus a bias term) and compute the MSE on the training and the testing set. Which one is the most informative? which one generalizes better? and worse?
3. Now train a model with all the variables plus a bias term. What is the performance in the test set? Now remove the worst-performing variable you found in step 2, and run again the experiment. What happened?
4. We can give more capacity to a linear regression model by using *basis functions* (Bishop, sec. 3.1). In short we can apply non-linear transformations to the input variables to extend the feature vector. Here we will try a polynomial function:
![Alt](img/poly.png)


## Gradient Descent



## Model Selection