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

[lenet_1]: ./trunc_b256_e100_do_lenet.png "LeNet 1"

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

After some experimentation, the following hyperparameter values yielded the best results.

**Weight initialization:**
* mean = 0
* stddev = 0.05

**Training:**
* epochs = 100
* batch size = 256
* learning rate = 0.0002
* dropout keep prob. = 0.5

The final result for this model was:
* Training time (EC2 g2.2x): 258.320 seconds
* Training set accuracy: 0.999
* Validation set accuracy: 0.967
* Test set accuracy: 0.949
* Downloaded image set accuracy: 0.875

The following graphs show the accuracy and loss during training:

![alt text][lenet_1]

The graph on the left shows the validation and training set accuracy during training. The blue line is the training set accuracy and the green line is the validation set accuracy. The graph on the right shows the loss for the last 10 epochs of training.

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
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

## Testing On New Images

Here are the German traffic signs that I found on the web:

![alt text][dl_1] ![alt text][dl_2] ![alt text][dl_3] ![alt text][dl_4] 
![alt text][dl_5] ![alt text][dl_6] ![alt text][dl_7] ![alt text][dl_8] 

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
