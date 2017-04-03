# **Behavioral Cloning** 

## Project Report

### This write-up summarizes my overal strategy in completing the Behavioral Cloning project. This includes the architecture and training steps.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-images/activation_layers.png "Activation Layers"
[image2]: ./writeup-images/nvidia.png "NVIDIA"
[image3]: ./writeup-images/center.jpg "Center Driving"
[image4]: ./writeup-images/mse.png "Training History"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model was adapted from NVIDIA's _[End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)_ consisting of 4 convolutional layers, 8 activation layers, a max pooling layer, 5 fully connected layers and 6 regularizing layers (batch normalization and dropout) (model.py lines 60-105).

![alt text][image2]
The original NVIDIA model has about 27 million connections and 250 thousand parameters.

The model includes ELU layers to introduce nonlinearity (code lines 60-105), and the data is normalized in the model using a Keras lambda layer (code line 65). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropouts and batch normalization layers in order to reduce overfitting (model.py lines 90-103). 

The model was trained and validated on dataset provided by Udacity. For some reason the model was not able to generalize successfully with the dataset that I generated. My dataset consisted of driving around the track in this fashions: forward, reverse as well going off track. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer with the default parameters, so the learning rate was not tuned manually (model.py line 203). I used Mean Squared Error for the loss function.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road provided by Udacity to successfully train a model that is able to generalize well. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try out my good old friend LeNet first and then followed along David Silver and used his model from NVIDIA video and in the end used an iterative approach to adapt from NVIDIA's model as described on their paper

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I played with the hyperparameters and modified the model so that Dropouts and Batch Normalization would prevent the model from overfitting.

Then I Noticed that using any epochs higher than 10 would result in overfitting so after a couple of experiments, I used three epochs.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I switched from my own dataset to the dataset provided by Udacity.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 60-105) consisted of a convolution neural network with the following layers and layer sizes. Here is a visualization of the architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 160×320×3 RGB image, normalized               | 
| Cropping      	       | outputs 65×320×3                              |
| Convolution 5x5		|	2x2 stride, valid padding, outputs 5x5x24    |
| ELU	                 | Activation layer  |
| Convolution 5x5		|	2x2 stride, valid padding, outputs 5x5x36    |
| ELU	                 | Activation layer  |
| Convolution 5x5		|	2x2 stride, valid padding, outputs 5x5x48    |
| ELU	                 | Activation layer  |
| Convolution 3x3		|	1x1 stride, same padding, outputs 3x3x64    |
| ELU	                 | Activation layer  |
| Max pooling	         	| 2x2 pooling,  1x1 stride |
| Flattening	          |  Flattened feature map     |
| Fully connected		| 1164 output    |
| ELU	                 | Activation layer  |
| Batch Normalization   | Regularizer |
| Dropout	               | Regularizer |
| Fully connected		| 100 output    |
| ELU	                 | Activation layer  |
| Batch Normalization   | Regularizer |
| Dropout	             | Regularizer |       
| Fully connected		| 50 output    |
| ELU	                 | Activation layer  |
| Batch Normalization   | Regularizer | 
| Fully connected		| 10 output    |
| ELU	                 | Activation layer  |
| Batch Normalization   | Regularizer |        
| Fully connected		| 1 output    |

Thanks to the great feedback on my Traffic Sign Classifier, I switched from ReLU activation layers to ELU. I did not get around to experiementing with Parametric Rectified Linear Unit (PReLU) that NVIDIA was using in their paper but overall ELU seemed to be doing a good job. I came across this figure on Hackernoon that I f ound quite informative.

![alt text][image1]
> Source: [Hackernoon](https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover gracefully in case it goes wayward. Then I repeated this process driving backwards in order to get more data points.

To augment the data sat, I also flipped images and angles as per David Silver suggestions and this allowed me to generate even more data. After the collection process, I had 13599 training images . I then preprocessed this data by cropping 70 pixels from top and 25 pixels from the bottom to exclude redundant information from images that would slow down the training process and increase my model size.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

![alt text][image4]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the above figure I used an Adam optimizer so that manually training the learning rate wasn't necessary.
