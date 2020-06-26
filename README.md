# Solution to the datascience challenge contest Put it on the map!

link to the contest: https://www.xstarter.io/challenge-details/put-it-on-the-map

<p align="center">
  <img src="outputs/random_geozones.jpg" width="600" title="hover text">
</p>

Goal: retrieve map lattitude and longitude coordinates

Database: 40k images with known coordinates for training and 10k images for testing

Example maps (different map-styles, resolutions and annotations):
<p align="center">
  <img src="map_examples/00ec31ca1e.png" width="200" title="hover text">
  <img src="map_examples/00bd4a1ea6.png" width="200" title="hover text">
  <img src="map_examples/00f6fe95ce.png" width="200" title="hover text">
</p>

Download the training and test data from the contest website

Workflow of the proposed solution:

* encoding images using transfert learning from pretrained ResNet model on a geographic zones classification task (Figure 1 shown above)
* binary hash images of training images in 512 feature space
* retrieving the closest image to the one to identify in that feature space using Locality Sensitive Hashing for fast approximate nearest neighbor

Run the notebook Find_similar_images_Training_submissionFinal.ipynb to access the solution
