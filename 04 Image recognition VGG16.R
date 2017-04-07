# randomthoughtsonr.blogspot.nl/2016/11/image-classification-in-r-using-trained.html

rm(list = ls())

library(reticulate)
use_condaenv("snakes")

library(tensorflow)

slim = tf$contrib$slim #Poor mans import tensorflow.contrib.slim as slim
tf$reset_default_graph() # Better to start from scratch

# We start with a placeholder tensor in which we later feed the images
# The first index of the tensor counts the image number and the second to 4th index is for the width, height, color. Since we want to allow for an arbitrary number of images of arbitrary size, we leave these dimensions open.
images = tf$placeholder(tf$float32, shape(NULL, NULL, NULL, 3))
imgs_scaled = tf$image$resize_images(images, shape(224,224))

# Definition of the network
library(magrittr) 
# The last layer is the fc8 Tensor holding the logits of the 1000 classes
fc8 = slim$conv2d(imgs_scaled, 64, shape(3,3), scope='vgg_16/conv1/conv1_1') %>% 
  slim$conv2d(64, shape(3,3), scope='vgg_16/conv1/conv1_2')  %>%
  slim$max_pool2d( shape(2, 2), scope='vgg_16/pool1')  %>%
  
  slim$conv2d(128, shape(3,3), scope='vgg_16/conv2/conv2_1')  %>%
  slim$conv2d(128, shape(3,3), scope='vgg_16/conv2/conv2_2')  %>%
  slim$max_pool2d( shape(2, 2), scope='vgg_16/pool2')  %>%
  
  slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_1')  %>%
  slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_2')  %>%
  slim$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_3')  %>%
  slim$max_pool2d(shape(2, 2), scope='vgg_16/pool3')  %>%
  
  slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_1')  %>%
  slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_2')  %>%
  slim$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_3')  %>%
  slim$max_pool2d(shape(2, 2), scope='vgg_16/pool4')  %>%
  
  slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_1')  %>%
  slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_2')  %>%
  slim$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_3')  %>%
  slim$max_pool2d(shape(2, 2), scope='vgg_16/pool5')  %>%
  
  slim$conv2d(4096, shape(7, 7), padding='VALID', scope='vgg_16/fc6')  %>%
  slim$conv2d(4096, shape(1, 1), scope='vgg_16/fc7') %>% 
  
  # Setting the activation_fn=NULL does not work, so we get a ReLU
  slim$conv2d(1000, shape(1, 1), scope='vgg_16/fc8')  %>%
  tf$squeeze(shape(1, 2), name='vgg_16/fc8/squeezed')


# visualise in tensorboard
dir.create('tensorboard')

tf$summary$FileWriter('tensorboard/vgg16', tf$get_default_graph())$close()

# Loading the weights
restorer = tf$train$Saver()
sess = tf$Session()
restorer$restore(sess, './vgg_16.ckpt')

# loading the images
library(jpeg)
img1 <- readJPEG('images/apple.jpg')
img1 <- readJPEG('images/football.jpg')
img1 <- readJPEG('images/tennis.jpg')
img1 <- readJPEG('images/ksnake.jpg')
d = dim(img1)
imgs = array(255*img1, dim = c(1, d[1], d[2], d[3]))

# Feeding and fetching the graph
# Now we have a graph in the session with the correct weights. We can do the predictions by feeding the placeholder tensor images with the value of the images stored in the array imgs. We fetch the fc8 tensor from the graph and store it in fc8_vals

fc8_vals = sess$run(fc8, dict(images = imgs))
fc8_vals[1:5]

# we are only interested in the positive values which we transfer to probabilities for the certain classes 
probs = exp(fc8_vals)/sum(exp(fc8_vals))

idx = sort.int(fc8_vals, index.return = TRUE, decreasing = TRUE)$ix[1:5]

# Reading the class names
library(readr)
names = read_delim("imagenet_classes.txt", "\t", escape_double = FALSE, col_names = FALSE)

# graph
library(grid)
g = rasterGrob(img1, interpolate=TRUE) 
text = ""
for (id in idx) {
  text = paste0(text, names[id,][[1]], " ", round(probs[id],5), "\n") 
}

library(ggplot2)
ggplot(data.frame(d=1:3)) + annotation_custom(g) + 
  annotate('text',x=0.05,y=0.05,label=text, size=7, hjust = 0, vjust=0, color='blue') + xlim(0,1) + ylim(0,1) 

