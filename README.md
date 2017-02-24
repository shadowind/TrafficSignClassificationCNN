#**Project 2: Traffic Sign Recognition** 

##Writeup

---


**Build a Traffic Sign Recognition Project**


The goals / steps of this project are the following:  
* Load the data set (see below for links to the project data set)  
* Explore, summarize and visualize the data set  
* Design, train and test a model architecture  
* Use the model to make predictions on new images  
* Analyze the softmax probabilities of the new images  
* Summarize the results with a written report  


[//]: # (Image References)

[image1]: ./images/SignOccurance.png "SignOccurance"
[image2]: ./images/Overfitting.png "Overfitting"
[image3]: ./images/vis.png "Display image and transformation"
[image4]: ./OnlineTestData/4.png "Traffic Sign 1"
[image5]: ./OnlineTestData/13.png "Traffic Sign 2"
[image6]: ./OnlineTestData/14.png "Traffic Sign 3"
[image7]: ./OnlineTestData/34.png "Traffic Sign 4"
[image8]: ./OnlineTestData/35.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 4th code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

First I convert the traffic sign to gray scale and make some image transformation like [equalization](http://docs.opencv.org/3.2.0/d5/daf/tutorial_py_histogram_equalization.html)
![alt text][image3]
Here is an exploratory visualization of the data set. It is a bar chart showing how the data

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 9-11 code cell of the IPython notebook.

I used the following workflow (Cell 9):
1. Trim image to keep only traffic sign region
2. Resize image to (32,32,n)
3. Convert image to gray
4. Scale image

And later when I trim test images, there's error. See detail in Cell 10. So I added an error handling method to return original image if there's error. 


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is in Cell 12

I first shuffle the processed training images. And cut the first 70% to be training set. The rest to be validation set


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 13 cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5    	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	   	| 1x2 stride,  outputs 14x14x6|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 1x2 stride,  outputs 5x5x16 				|
| Flatten					|	400											|
| Fully connected		| input 400, output 200        									|
| RELU					|												|
| Dropout				|	0.8											|
| Fully connected		| input 200, output 84       									|
| RELU					|												|
| Dropout				|	0.8											|
| Fully connected		| input 84, output 43       									|
| Softmax	cross entropy			|         									|  



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 15-18 cell of the ipython notebook. 
optimizer: AdamOptimizer
batch size: 128
epochs: 15


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy: 0.975
* validation set accuracy: 0.965 
* test set accuracy of: 90.8

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First I used the default LeNet architecture. and build up more layers based on that model  
* What were some problems with the initial architecture?  
The inital architecture is quite simple, the major problem I observe is overfitting the training set. As I increase the Epochs, training accuarcy increases till 99%, however the validation didn't increase accordingly. Even had some decrease. 
![alt text][image2]
* How was the architecture adjusted and why was it adjusted?   
I added two dropout layers, and increase the fully connected layer 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web <http://benchmark.ini.rub.de/?section=gtsrb&subsection=news>:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The overall image quality are pretty good. 
The third image might be difficult to classify because the image is a bit distorted.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 27-28 cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (70km/h)  									| 
| Yield					| Yield											|
| Stop	      		| Stop					 				|
| Turn left ahead			| Turn left ahead      							|
| Ahead only    			| Ahead only 										|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook.

For the third image, the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999         			| Stop sign   									| 
| .00     				| Turn right ahead										|
| .00					| No entry											|
| .00	      			| Yield					 				|
| .00				    | Road work      							|

