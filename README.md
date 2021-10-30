## implicit_representations

Implicit neural representations of images.

The repo is composed of four folders:

* data
* experiments
* demo
* utils

***
The **data** folder contains sub-folders, each one contains images of constant resolution. The content of the
sub-folders is exact, except the resolution of the images.
***
The **experiments** folder includes sub-folders of different experiments defined by their hyper-parameters. In those
sub-folders, are the saved models and figures generated from the training and the analysis of a given model. In case the
input hyperparameters lead to an existing experiment - the model will not be generated but will be loaded.
***
The **demo** folder includes four *.py files:
* *constants.py* includes some constants used by other functions
* *train.py* includes a function which is used for training or retraining a model. Since some models are provided in the
  repo, the user can skip this stage and use the existing models for the remaining analysis functions.
* *single_image_interpolation.py* is used to show the inherent image interpolation capability of the model. It plots (
  and saves) predicted images (of the training set) of different resolutions.
* *image_pair_interpolation.py* is used to visualize the similarity/dissimilarity of different images in the latent
  space and to demonstrate an interpolation between a pair of similar images with a sequenced *.gif.
* **Note:** all the above functions can run from command line with user provided arguments.

***
***
Amit Oved
amitoved@gmail.com