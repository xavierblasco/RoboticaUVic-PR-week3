# RoboticaUVic-PR-week1
UVic Robotics Master. Pattern Recognition Homework 1

## Instructions

Fork this repository and write code to complete the assignments. When
executed, your code must print the answers to the questions in each
section, alongside the results that led to these conclusions. Module
*textwrap* can be used to format long paragraphs of text and make them
look nicer. An IPython notebook is an acceptable alternative to a
plain python program.

First upload the completed assignment to the course Moodle for
grading; then, correct all the isses marked by the teacher and push it
to GitHub for the final evaluation.

This is a personal assignment, please complete it **individually**. 

## Preliminaries

This and the next three homework assignments have to be completed
using Python. Python is a very easy to learn and flexible programming
language, with many libraries for the most diverse tasks. There are
currently two main flavours of Python that are not fully compatible:
version 2 (currently 2.7) and version 3 (currently 3.5). In this
course we will be using version 2.7, as the version of OpenCV shipped
with Ubuntu does not support the latter version.

For the homework in this course, we will use the following libraries, mostly from the SciPy family:
- [SciPy](http://www.scipy.org/) is a collection of libraries for scientific computing in Python.
- [Numpy](http://www.numpy.org/) provides the backbone functionality for numerical processing.
- [Matplotlib](http://matplotlib.org/) is a powerful plotting library.
- [Scikit Learn](http://scikit-learn.org/stable/) Is a simple and complete machine learning library for Python.
- [OpenCV](http://opencv.org/) is the reference computer vision and image processing library, with comprehensive Python bindings.

Besides these libraries, [IPython](http://ipython.org/) is recommended as an extended and more friendly interactive Python interpreter. It is also necessary in order to view the course notebooks.

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

For this assignment, it is recommended to read at least the **Python+Numpy** tutorial.

## Linear Regression 

See the ipython notebook from the class for reference.

- **Q1**) [Download](http://archive.ics.uci.edu/ml/datasets/Housing) the
Housing Data Set from the UCI repository.

- **Q2**) Load the data (using *numpy.loadtxt*) and separate the last column
(target value, MEDV). Compute the average of the target value and the
MSE obtained using it as a constant prediction.

- **Q3**) Split the data in two parts (50-50) for training and testing (first
half for training, second half for testing). Train a linear regressor
model for each variable individually (plus a bias term) and compute
the MSE on the training and the testing set. Which variable is the
most informative? which one makes the model generalizes better? and
worse? Compute the coefficient of determination (R^2) measure for the
test set.

- **Q4**) Now train a model with all the variables plus a bias term. What is
the performance in the test set? Try removing the worst-performing
variable you found in step 2, and run again the experiment. What
happened?

- **Q5**) We can give more capacity to a linear regression model by using
*basis functions* (Bishop, sec. 3.1). In short, we can apply
non-linear transformations to the input variables to extend the
feature vector. Here we will try a polynomial function:  
 &nbsp;&nbsp;&nbsp;&nbsp; ![Polynomial basis expansion](img/poly.png)  
Repeat step 2 but adding, one by one, all
polynomials up to degree 4. What are the effects of adding more
capacity to the model?

## Gradient Descent

As we have seen, overfitting is a problem that arises when we try to
have more powerful methods, able to better adapt to the data. In order
to reduce overfitting, we can **regularize** our model, but then we do
not have a closed form solution and must resort to
optimization. Gradient descent is one of the most common optimization
techniques

```
1. Initialize theta at random
2. it = 0
2. While it<maxit:
3.     theta<sub>t+1</sub> = theta - nu * f'(theta)
4.     
```

- **Q6**) Implement code to train a regularized linear regression
model using gradient descent according to the previous pseudocode.
The objective function is the following:  
 &nbsp;&nbsp;&nbsp;&nbsp; ![Alt](img/RegLogReg.png)  
And its derivative is:  
  &nbsp;&nbsp;&nbsp;&nbsp; ![Alt](img/RegLogRegPrim.png)  
As an optional exercise, work the derivation of the objective function.

- **Q7**)

## Model Selection


## Extra

. Think of how you would use linear regression to improve or automate some aspect of your work or daily life (doesn't need to make a lot of sense), and describe the way in which you would approach the problem.