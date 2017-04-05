rm(list = ls())

# reticulate is package to talk to python
library(reticulate)

# snakes is the virtualenv with tensorflow
use_condaenv("snakes")

library(tensorflow)

# The TensorFlow API is divided into modules. The top-level entry point to the API is tf
sess = tf$Session()

hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)

a <- tf$constant(10L)
b <- tf$constant(32L)
sess$run(a + b)


# --- sub modules
# Call the conv2d function within the nn sub-module
# tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')

# Create an optimizer from the train sub-module
# optimizer <- tf$train$GradientDescentOptimizer(0.5)

# --- numeric types
# The TensorFlow API is more strict about numeric types than is customary in R'
# Many TensorFlow function parameters require integers (e.g. for tensor dimensions) and in those cases it’s important to use an R integer literal (e.g. 1L)


# --- Numeric Lists
# there are a couple of special cases (mostly involving specifying the shapes of tensors) where you may need to create a numeric list with an embedded NULL or a numeric list with only a single item
# in order to force the argument to be treated as a list rather than a scalar, and to ensure that NULL elements are preserved

x <- tf$placeholder(tf$float32, list(NULL, 784L))
W <- tf$Variable(tf$zeros(list(784L, 10L)))
b <- tf$Variable(tf$zeros(list(10L)))

# --- Tensor Shapes
# This need to use list rather than c is very common for shape arguments 
# For these cases there is a shape function which you can use to make the calling syntax a bit more more clear

x <- tf$placeholder(tf$float32, shape(NULL, 784L))
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

# --- Tensor indexes
# Tensor indexes within the TensorFlow API are 0-based (rather than 1-based as R vectors are)
# This typically comes up when specifying the dimension of a tensor to operate on (e.g with a function like  tf$reduce_mean or tf$argmax). The first dimension of a tensor is specified as 0L, the second 1L, and so on. For example:
  
# call tf$reduce_mean on the second dimension of the specified tensor
cross_entropy <- tf$reduce_mean(
    -tf$reduce_sum(y_ * tf$log(y_conv), reduction_indices=1L)
  )

# call tf$argmax on the second dimension of the specified tensor
correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))







# Create 100 phony x, y data points, y = x * 0.1 + 0.3
x_data <- runif(100, min=0, max=1)
y_data <- x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W <- tf$Variable(tf$random_uniform(shape(1L), -1.0, 1.0))
b <- tf$Variable(tf$zeros(shape(1L)))
y <- W * x_data + b

# Minimize the mean squared errors.
loss <- tf$reduce_mean((y - y_data) ^ 2)
optimizer <- tf$train$GradientDescentOptimizer(0.5)
train <- optimizer$minimize(loss)

# Launch the graph and initialize the variables.
sess = tf$Session()
sess$run(tf$global_variables_initializer())

# Fit the line (Learns best fit is W: 0.1, b: 0.3)
for (step in 1:201) {
  sess$run(train)
  if (step %% 20 == 0)
    cat(step, "-", sess$run(W), sess$run(b), "\n")
}