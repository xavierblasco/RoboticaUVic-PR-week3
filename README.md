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
programming languages are already known. There are some tutorials to
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

- **Q3**) Split the data in two parts (50-50) for training and testing
(first half for training, second half for testing). Train a linear
regressor model for each variable individually (plus a bias term) and
compute the MSE on the training and the testing set. Which variable is
the most informative? which one makes the model generalizes better?
and worse? Compute the coefficient of determination (R^2) for the test
set.

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
not have a closed form solution and must resort to optimization. First
we will use *Gradient Descent*, which is a very common optimization
algorithm. Here is a simple implementation in pseudo-code:

```
0. Function Gradient_Descent
1.   Initialize theta(0) at random
2.   t=0, maxit=100, step=0.01; epsilon=0.0001, loss=zeros(maxit)
3.   loss(0) = f(theta)
4.   do
5.      t=t+1
6.      theta(t) = theta(t-1) - step * f'(theta(t-1))
7.      loss(t) = f(theta)
8.   While t<maxit and (t>1 and loss(t-1)-loss(t)>epsilon):
9.   return theta
```

- **Q6**) The objective function *f* for Regularized Linera Regression 
is the following:  
 &nbsp;&nbsp;&nbsp;&nbsp; ![f=Regularized Linear Regression](img/RegLinReg.png)  
And its derivative *f'* is:  
  &nbsp;&nbsp;&nbsp;&nbsp; ![f-prime](img/RegLinRegPrim.png)  
Implement two functions in Python, one that computes *f* and another
that computes *f'*.  As an optional exercise, work the derivation of
the objective function.

- **Q7**) Implement code to train a regularized linear regression
model using gradient descent according to the previous pseudocode.
 > Some hints:  
 >  - Make sure your *f* and *f'* functions are correct. Here are some
 >  values for reference:  
 >   *f*(data_train_with_bias, theta_all_zeros) = 660.1083  
 >   *f'*(data_train_with_bias, theta_all_zeros) = [ -48.62, -20.40, -676.58, -422.27, -3.90, -24.70, -317.49, -3033.81, -206.06, -222.99, -15327.84, -857.69, -18476.12, -477.47]  
 >  - Start with only a few iterations, and check that your loss is decreassing, if it is doing a zig-zag, lower your learning step


## Model Selection


## Extra

. Think of how you would use linear regression to improve or automate some aspect of your work or daily life (doesn't need to make a lot of sense), and describe the way in which you would approach the problem.