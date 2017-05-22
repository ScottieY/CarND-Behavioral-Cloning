# Model Architecture and Training Strategy 

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths of 6 (model.py lines 55-57) 

The model includes RELU layers to introduce nonlinearity (code line 55-57), and the data is normalized in the model using a Keras lambda layer (code line 52). 

#### 2. Attempts to reduce overfitting in the model

The model contains max pooling layers in order to reduce overfitting (model.py lines 58). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 14). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 64).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and reverse laps.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive the car around the track safely.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because it worked for the traffic sign recognition project well

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to use a less complicated network

Then I also tried the nvidia net, however I found the network was to complicated for this project, it takes very long time to train and overfits data in the first epoch, so I decide to stick with LeNet and fine tune it.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as the bridge and tight corners, to improve the driving behavior in these cases, I did recovery lap and reverse lap to augment the data set to provide more detail on how the car should behave.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 52-62) consisted of a convolution neural network with the following layers and layer sizes ...

It first normalize the data on a scale between -0.5 - 0.5.

Then it crop out the bottom of the image which is the hood of the car.

Next it apply 6@(5,5) convolution layer followed by relu activation then a maxpooling.

Repeated the step above.

Then flatten the inputs and dense it to 120 outputs, 84 outputs and finally 1 output.



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:





![center_2017_05_05_02_35_07_489](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_07_489.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from a curb on the right side

![center_2017_05_05_02_35_09_123](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMGcenter_2017_05_05_02_35_09_123.jpg)

![center_2017_05_05_02_35_09_191](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_09_191.jpg)

![center_2017_05_05_02_35_09_267](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_09_267.jpg)

![center_2017_05_05_02_35_09_339](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_09_339.jpg)

![center_2017_05_05_02_35_09_406](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_09_406.jpg)

![center_2017_05_05_02_35_09_482](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_09_482.jpg)

![center_2017_05_05_02_35_09_550](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_09_550.jpg)

![center_2017_05_05_02_35_09_617](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_09_617.jpg)

![center_2017_05_05_02_35_09_691](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_09_691.jpg)

![center_2017_05_05_02_35_09_758](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_09_758.jpg)

![center_2017_05_05_02_35_09_861](https://github.com/ScottieY/CarND-Behavioral-Cloning/tree/master/data/IMG/center_2017_05_05_02_35_09_861.jpg)





I used both left and right camera with adding/subtracting a corretion angle to obtain more data points.

After the collection process, I had over 6000 number of data points. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the change in loss in approaching to 0. I used an adam optimizer so that manually training the learning rate wasn't necessary.
