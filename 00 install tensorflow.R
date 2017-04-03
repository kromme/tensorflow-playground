# ---- Install tensorflow with anaconda



# install Tensorflow in Python
# install python3.5 in virtual env (snakes)
#   activate snakes

# install tensorflow
#   pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.0.1-cp35-cp35m-win_amd64.whl

# install latest Rcpp, which is not in MRO
install.packages('C:/Users/j.kromme/Documents/rcpp/Rcpp', repos = NULL, type="source")

# install reticulate (python binding), whichi is not in MRO
devtools::install_github("rstudio/reticulate")

# install tensorflow
devtools::install_github("rstudio/tensorflow")

# use conda virtual environment
library(reticulate)
use_condaenv("snakes")

# check
library(tensorflow)
