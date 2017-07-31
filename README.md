# Traffic Sign Recognition

The steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[sample_image]: ./sample_image.png "Sample image"

[sign_hist]: ./sign_hist.png "Traffic Signs"

[mean_image]: ./sample_image_processed.png "Mean image"

[dl_1]: ./downloaded_sign_images/1.png "Speed limit (30km/h)"
[dl_2]: ./downloaded_sign_images/11.png "Right-of-way at the next intersection"
[dl_3]: ./downloaded_sign_images/17.png "No entry"
[dl_4]: ./downloaded_sign_images/18.png "General caution"
[dl_5]: ./downloaded_sign_images/21.png "Double curve"
[dl_6]: ./downloaded_sign_images/27.png "Pedestrian"
[dl_7]: ./downloaded_sign_images/28.png "Children crossing"
[dl_8]: ./downloaded_sign_images/31.png "Wild animals crossing"

[lenet_1]: ./graph_trunc_b256_e100_do_lenet.png "LeNet 1"

[vgg_1]: ./graph_he_b256_e100_r0001_vgg.png "VGG 1"
[vgg_2]: ./graph_he_do_b256_e150_r0001_vgg.png "VGG 2"

## Data Set Summary & Exploration

I used the python and the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Below is a sample traffic sign image from the dataset :

![alt_text][sample_image]

Below is a histogram showing the distribution of sign types in the training, validation and testing datasets.

![alt text][sign_hist]

## Data Pre-processing

For the data pre-processing step, I followed the recommendations in the [Stanford CS231n lecture and notes](https://cs231n.github.io/neural-networks-2/). Based on the notes and lecture, I decided to zero center each color channel of the training dataset. Below is an image showing the mean color of the training dataset using this method.

![alt_text][mean_image]

I decided not to normalize the images since "the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional preprocessing step." (CS231n notes)

## Model Architecture

### Model 1

My first attempt at improving validation accuracy was to modify LeNet by adding dropout regularization to the fully connected hidden layers and adjusting the hyperparameters. The model has the following architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x6	    | 1x1 stride, valid padding, outputs 10x10x16     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 140        									|
| RELU					|												|
| Dropout | keep probability 0.5 |
| Fully connected		| 84       									|
| RELU					|										|
| Dropout | keep probability 0.5 |
| Fully connected		|    43   									|
| Softmax				|   									|

After increasing the number of training epochs and lowering the learning rate, I thought that the model might be overfitting the training data set due to the 100% accuracy and an 11% difference with validation set accuracy. I added dropout regularization and continued to fine tune the hyperparameters until the model returned the result below. 

The network was trained using the TensorFlow AdamOptimizer to minimize the cross entropy loss function. The following hyperparameter values yielded the best results.

**_Weight initialization:_**
* mean = 0
* stddev = 0.05

**_Training:_**
* epochs = 100
* batch size = 256
* learning rate = 0.0002
* dropout keep prob. = 0.5

**_The final result for this model was:_**
* Training time (EC2 g2.2xlage): 258.320 seconds
* Training set accuracy: 0.999
* Validation set accuracy: 0.967
* Test set accuracy: 0.949
* Downloaded image set accuracy: 0.875

The following graphs show the accuracy and loss during training:

![alt text][lenet_1]

The graph on the left shows the validation and training set accuracy during training. The blue line is the training set accuracy and the green line is the validation set accuracy. The graph on the right shows the average loss for the last 10 epochs of training.

**Testing On New Images**

Here are the German traffic signs that I found on the web:

![alt text][dl_1] ![alt text][dl_2] ![alt text][dl_3] ![alt text][dl_4] 
![alt text][dl_5] ![alt text][dl_6] ![alt text][dl_7] ![alt text][dl_8]

The 30 km/h speed sign might be difficult to classify because it might be hard for a model to differentiate it from the 80 km/h signs. The model should not have any problem classifying the rest of the downloaded images. All of the images contain a straight-on perspective and are high contrast. Many of the images in the original data set have terrible contrast and are difficult to see what class they belong to.

**_Here are the results of the prediction:_**

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)   | Speed limit (30km/h)   |
| Right-of-way at the next intersection   									|   Right-of-way at the next intersection  |
| No entry    			| No entry 										|
| General caution 				| General caution 										|
| Double curve	      		| Right-of-way at the next intersection					 				|
| Pedestrians			| Pedestrians     							|
| Children crossing			| Children crossing     							|
| Wild animals crossing | Wild animals crossing |

The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%.

**_Here are the top-5 probabilities for each sign:_**
 
* Speed limit (30km/h):
  1. Speed limit (30km/h) -- 0.99994
  2. Speed limit (50km/h) -- 0.00005
  3. Speed limit (80km/h) -- 0.00001
  4. Speed limit (20km/h) -- 0.00000
  5. Speed limit (70km/h) -- 0.00000

* Right-of-way at the next intersection
  1. Right-of-way at the next intersection -- 1.00000

* No entry:
  1. No entry -- 1.00000

* General caution:
  1. General caution -- 1.00000

* Double curve:
  1. Right-of-way at the next intersection -- 0.99987
  2. Pedestrians -- 0.00010
  3. Children crossing -- 0.00002
  4. Beware of ice/snow -- 0.00002
  5. End of all speed and passing limits -- 0.00000

* Pedestrians:
  1. Pedestrians -- 0.99541
  2. General caution -- 0.00311
  3. Right-of-way at the next intersection -- 0.00145
  4. Road narrows on the right -- 0.00003
  5. Traffic signals -- 0.00000

* Children crossing:
  1. Children crossing -- 0.99951
  2. Slippery road -- 0.00048
  3. Road narrows on the right -- 0.00000
  4. Dangerous curve to the right -- 0.00000
  5. Bicycles crossing -- 0.00000

* Wild animals crossing:
  1. Wild animals crossing -- 1.00000

The model appears to be very certain of every prediction for this dataset. Perhaps if I had selected images at different angles and with low contrast this would not be the case.

### Model 2

My second attempt at improving validation accuracy was to try to use a deeper model with an architecture similar to VGGNet. The model has the following architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Convolution 3x3x64	    | 1x1 stride, same padding, outputs 32x32x64     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3x64	    | 1x1 stride, same padding, outputs 16x16x128     									|
| RELU					|												|
| Convolution 3x3x128	    | 1x1 stride, same padding, outputs 16x16x128     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128 				|
| Convolution 3x3x128	    | 1x1 stride, same padding, outputs 8x8x256     									|
| RELU					|												|
| Convolution 3x3x256	    | 1x1 stride, same padding, outputs 8x8x256     									|
| RELU					|												|
| Convolution 3x3x256	    | 1x1 stride, same padding, outputs 8x8x256     									|
| RELU					|				
| Max pooling	      	| 2x2 stride,  outputs 4x4x256 				|
| Fully connected		| 2048        									|
| RELU					|												|
| Dropout | keep probability 0.5 |
| Fully connected		| 2048       									|
| RELU					|										|
| Dropout | keep probability 0.5 |
| Fully connected		| 1024       									|
| RELU					|										|
| Dropout | keep probability 0.5 |
| Fully connected		|    43   									|
| Softmax				|   									|

I decided to base the model off of VGGNet because I wanted to experiment with regularization using a deeper model with wider layers.

The model was trained using the TensorFlow AdamOptimizer to reduce the cross entropy loss function. After some experimentation, the following hyperparameter values yielded the best results.

**_Weight initialization:_**
* As recommended in the CS231n lecture, I used a weight initialization that has been shown to perform better for deep CNNs (Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, He et al.). This was accomplished using the tf.contrib.layers.variance_scaling_initializer with factor=2.0 and mode='FAN_IN'.

**_Training:_**
* epochs = 150
* batch size = 256
* learning rate = 0.0001
* dropout keep prob. = 0.5

The final result for this model was:
* Training time (EC2 g2.2x): 10447.043 seconds
* Training set accuracy: 1.000
* Validation set accuracy: 0.960
* Test set accuracy: 0.952
* Downloaded image set accuracy: 0.75

The following graphs show the accuracy and loss during training without regularization:

![alt text][vgg_1]

The graph on the left shows the validation and training set accuracy during training. The blue line is the training set accuracy and the green line is the validation set accuracy. The graph on the right shows the average loss for the last 10 epochs of training. Clearly, this model is overfitting the training data set.

The following graphs show the accuracy and loss during training with dropout regularization on the hidden fully-connected layers:

![alt text][vgg_2]

While this model has marginally worse accuracy on the validation set, it takes ~41x as long to train.
