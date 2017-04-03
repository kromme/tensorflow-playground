rm(list = ls())
library(reticulate)
use_condaenv("snakes")

library(tensorflow)

#  ------ Load mnist data
# The MNIST data is split into three parts: 55,000 data points of training data (mnist$train), 10,000 points of test data (mnist$test), and 5,000 points of validation data (mnist$validation). 
#  we’re going to want our labels as “one-hot vectors”. A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. 
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

trainset_df <- mnist$train$images
dim(trainset_df)

y_df <- mnist$train$labels
dim(y_df)

# visualize
x = 0:9
i = 1
image(matrix(trainset_df[i,], ncol = 28))
x[y_df[i,]==1]






# ------ Theory
# . If you want to assign probabilities to an object being one of several different things, softmax is the thing to do, 
# because softmax gives us a list of values between 0 and 1 that add up to 1. Even later on, when we train more sophisticated models, 
# the final step will be a layer of softmax.
# first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities.

# We also add some extra evidence called a bias. Basically, we want to be able to say that some things are more likely 
# independent of the input. The result is that the evidence for a class i given an input x is:
# evidence = sum_j(W_i,j * X_j + B_i), where W = weights, b = bias, for class i and j is an index for summing over the pixels.
# y = softmax(evidence) = normalize(exp(evidence)). softmax_i = (exp(X_i) / sum_j(exp(X_j)))



# ------ Create the model
# TensorFlow also does its heavy lifting outside R, but it takes things a step further to avoid this overhead. Instead of running a single expensive operation independently from R, TensorFlow lets us describe a graph of interacting operations that run entirely outside R (Approaches like this can be seen in a few machine learning libraries.)

# We describe these interacting operations by manipulating symbolic variables. Let’s create one (to access the TensorFlow API we reference the tf object that is exported by the tensorflow package)
# x isn’t a specific value. It’s a placeholder, a value that we’ll input when we ask TensorFlow to run a computation
# NULL means that it can be any size


x <- tf$placeholder(tf$float32, shape(NULL, 784L))

#  A Variable is a modifiable tensor that lives in TensorFlow’s graph of interacting operations.
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

# First, we multiply x by W with the expression tf$matmul(x, W). This is flipped from when we multiplied them in our equation, where we had Wx, as a small trick to deal with x being a 2D tensor with multiple inputs. We then add b, and finally apply tf$nn$softmax
y <- tf$nn$softmax(tf$matmul(x, W) + b)

# --- Define loss and optimizer
# new placeholder for predictions
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))


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




