rm(list = ls())

# first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channel
library(reticulate)
use_condaenv("snakes")

library(tensorflow)

#  ------ Load mnist data
# The MNIST data is split into three parts: 55,000 data points of training data (mnist$train), 10,000 points of test data (mnist$test), and 5,000 points of validation data (mnist$validation). 
#  we’re going to want our labels as “one-hot vectors”. A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. 
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)



sess <- tf$InteractiveSession()


x <- tf$placeholder(tf$float32, shape(NULL, 784L))
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))



#  A Variable is a modifiable tensor that lives in TensorFlow’s graph of interacting operations.
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

sess$run(tf$global_variables_initializer())

# First, we multiply x by W with the expression tf$matmul(x, W). This is flipped from when we multiplied them in our equation, where we had Wx, as a small trick to deal with x being a 2D tensor with multiple inputs. We then add b, and finally apply tf$nn$softmax
y <- tf$nn$softmax(tf$matmul(x, W) + b)

# --- Define loss and optimizer
# new placeholder for predictions


# define cost function
# Cross-entropy arises from thinking about information compressing codes in information theory
# H_y`(y) = - sum_i(y_i` * log (y_i))
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * log(y), reduction_indices=1L))

# it can automatically use the backpropagation algorithm
optimizer <- tf$train$GradientDescentOptimizer(0.5)
train_step <- optimizer$minimize(cross_entropy)

# we have to create an operation to initialize the variables we created. Note that this defines the operation but does not run it yet
init = tf$global_variables_initializer()

# Create session and initialize  variables
sess <- tf$Session()
sess$run(init)



# Train: we’ll run the training step 1000 times!
# Each step of the loop, we get a “batch” of one hundred random data points from our training set. We run train_step feeding in the batches data to replace the placeholders.

# Using small batches of random data is called stochastic training – in this case, stochastic gradient descent. 
# Ideally, we’d like to use all our data for every step of training because that would give us a better sense of what we should be doing, but that’s expensive. 
# So, instead, we use a different subset every time. Doing this is cheap and has much of the same benefit.
for (i in 1:1000) {
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  sess$run(train_step,
           feed_dict = dict(x = batch_xs, y_ = batch_ys))
}

# Test trained model
correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(accuracy,
         feed_dict = dict(x = mnist$test$images, y_ = mnist$test$labels))




#  initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients


weight_variable <- function(shape) {
  initial <- tf$truncated_normal(shape, stddev=0.1)
  tf$Variable(initial)
}

bias_variable <- function(shape) {
  initial <- tf$constant(0.1, shape=shape)
  tf$Variable(initial)
}

# Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input
# Our pooling is plain old max pooling over 2x2 blocks.

conv2d <- function(x, W) {
  tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')
}

max_pool_2x2 <- function(x) {
  tf$nn$max_pool(
    x, 
    ksize=c(1L, 2L, 2L, 1L),
    strides=c(1L, 2L, 2L, 1L), 
    padding='SAME')
}


# -- first conv layer
# It will consist of convolution, followed by max pooling. The convolutional will compute 32 features for each 5x5 patch
# Its weight tensor will have a shape of (5, 5, 1, 32). The first two dimensions are the patch size, the next is the number
# of input channels, and the last is the number of output channels. We will also have a bias vector with a component for each output channel.

W_conv1 <- weight_variable(shape(5L, 5L, 1L, 32L))
b_conv1 <- bias_variable(shape(32L))

# first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channel
x_image <- tf$reshape(x, shape(-1L, 28L, 28L, 1L))



h_conv1 <- tf$nn$relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 <- max_pool_2x2(h_conv1)



W_conv2 <- weight_variable(shape = shape(5L, 5L, 32L, 64L))
b_conv2 <- bias_variable(shape = shape(64L))

h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 <- max_pool_2x2(h_conv2)

W_fc1 <- weight_variable(shape(7L * 7L * 64L, 1024L))
b_fc1 <- bias_variable(shape(1024L))

h_pool2_flat <- tf$reshape(h_pool2, shape(-1L, 7L * 7L * 64L))
h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)

# --- Dropout
# To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron’s output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow’s tf$nn$dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.1

keep_prob <- tf$placeholder(tf$float32)
h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)

# --- Readout Layer
# we add a softmax layer, just like for the one layer softmax regression above.

W_fc2 <- weight_variable(shape(1024L, 10L))
b_fc2 <- bias_variable(shape(10L))

y_conv <- tf$nn$softmax(tf$matmul(h_fc1_drop, W_fc2) + b_fc2)

# --- Train and Evaluate the Model

# How well does this model do? To train and evaluate it we will use code that is nearly identical to that for the simple one layer SoftMax network above.
# The differences are that:
# We will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.
# We will include the additional parameter keep_prob in feed_dict to control the dropout rate.
# We will add logging to every 100th iteration in the training process.
# Feel free to go ahead and run this code, but it does 20,000 training iterations and may take a while (possibly up to half an hour), depending on your processor.

cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y_conv), reduction_indices=1L))
train_step <- tf$train$AdamOptimizer(1e-4)$minimize(cross_entropy)
correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

sess <- tf$InteractiveSession()
sess$run(tf$global_variables_initializer())


for (i in 1:10000) {
  batch <- mnist$train$next_batch(50L)
  if (i %% 100 == 0) {
  
      train_accuracy <- accuracy$eval(feed_dict = dict(
      x = batch[[1]], y_ = batch[[2]], keep_prob = 1.0))
    cat(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
  }
  train_step$run(feed_dict = dict(
    x = batch[[1]], y_ = batch[[2]], keep_prob = 0.5))
}

train_accuracy <- accuracy$eval(feed_dict = dict(
  x = mnist$test$images, y_ = mnist$test$labels, keep_prob = 1.0))
cat(sprintf("test accuracy %g", train_accuracy))
