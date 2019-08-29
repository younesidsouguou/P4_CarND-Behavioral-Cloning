# **Behavioral Cloning** 

---

### Project Description

The aim of this project is to showcase the power of deep neural networks into mimicing some specific behavior and reproducing it for a given task. In our case, it's driving a car into the road. The data provided in this project consist of images coming from cameras mounted on the car (front, front left, front right) and the steering angles of the vehicle. All of this, taking place into the Udacity's simulator.

My pipeline is described as below:

- Data preprocessing
- Data augmentation 
- Model design and architecture
- Model training and deployment

### Files included

`model.py` : Building and training the model .<br>
`drive.py` : Driving the car in the simulator after training.<br>
`model.h5` : Model weights.<br>
`output_video.mp4` : The output video of testing the model in the simulator.

### Data preprocessing

My training data consists of 8036 samples. Each sample contains 3 images (left, center, right) taken at the same time and their corresponding steering value ranging from -1 to 1. So I'm having a 8036x3 = 24108 images with repeating steering values for left, right and center images taken at the same time, which results a 24108 sample images.

### Data augmentation

In order to augment my data and give my neural network a better idea on what's going on. I flipped the images on the vertical axis and multiplied their corresponding steering values by -1 so that the model don't overfit on the left turn only but also on the right turns.

![](sample_datum.png)

### Model architecture

The architecture I built was inspired by the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) that have been used in their End-to-End Deep Learning for Self-Driving Cars project.

<center>
<img src="cnn_arch.png">
</center>
<br>
My architecture named DriveNet is described as follows:

|Layer					|Description									|Activation										| Shape				| 
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:| :----------------:| 
|Input					| RGB image										|-												| (160x320x3)		| 
|Lambda					| Normalization Layer to help the network converge fastly|-										| (160x320x3)		|
|Cropping				| Cropping out region of less interest (sky, ...)|-												| (65x320x3)		|
|Conv24					| 5x5 filter, 2x2 strides						|ReLU											| (31x158x24)		| 
|Conv36					| 5x5 filter, 2x2 strides						|ReLU											| (14x77x36)		|
|Conv48					| 5x5 filter, 2x2 strides						|ReLU											| (5x37x48)			|
|Conv64					| 3x3 filter, 1x1 strides						|ReLU											| (3x35x64)			|
|Conv64					| 3x3 filter, 1x1 strides						|ReLU											| (1x33x64)			|
|Flatten				| Flattening into one-dimensional array			|-												| (2112)			|
|Dropout				| 50% of dropout to avoid overfitting on the feature vector			|-							| (2112)			|
|Dense100				| Fully connected layer of 100 neurons			|Linear											| (100)				|
|Dropout				| 50% of dropout								|-												| (100)				|
|Dense10				| Fully connected layer of 10 neurons			|Linear											| (10)				|
|Output					| Output layer									|Linear											| (1)				|

<center>Trainable parameters: 348,219</center>
<br>

I included in my model some preprocessing like the normlization and cropping of the images. The ReLU activation was doing good introducting non-linearity into the model. The dropout was also helpfull making me avoid overfitting.

### Model training

I chose to optimize the mean square error loss function using `Adam` optimizer with a constant learning rate of 5e-3.  I split and shuffled  my data using 20% of it for validation. I trained my model only on the training data for 7 Epochs using a batch size of 64. 

### Model testing and deployment

I tested my model on the udacity simulator it was driving very good. So I wanted to push my testing further to evaluate model robustness. What I did is that I edited the `drive.py` file so I increased the speed of the PI throttle controller from 9 MPH to 30 MPH so if the model isn't stable enough certainly it will fail at some point in the circuit like by the bride or the turns right after it. But my model has resisted all of this and still worked fine. I'm very proud of this achievement.