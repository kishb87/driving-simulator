## Udacity Self Driving Car Nanodegree Project 3 - Behavioral Cloning

### Network Architechture

For this project, I was able to create a neural network that can steer a car with constant throttle in a simulated environment. I implemented the [Nvidia Paper on End-to-end learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) for my neural architechture which has demonstrated the ability to steer a car in the real world. The model takes in as an input 66x200X3 images. The model consists of 24 filters followed by 36 filters, 48 filters and two 64 filter segments. Each segment has 2x2 max pooling, a dropout layer, and a relu activation function. The final segment flattens the input data then consists of fully several fully conneted layers (dense layer of 1164 neurons, a dropout layer, dense layer of 100, dense layer of 50, dense layer of 10, and finally a dense layer of 1 to generate the output). It is important to note that it is not possible to distinguish which parts of the network do feature extraction and which parts of the network providing the steering prediction.

NVIDIA ConvNet Architechture for Steering Angle Prediction:  
![alt text](arch.png "NVIDIA Architechture")


### Training
An Ubuntu 16.04 machine with an Intel i5 processor and a Nvidia GTX 1070 GPU was used to train the network. Before the network could be trained, the images had to be preprocessed. The training images were resized, normalized, and transformed to YUV from RGB. In addition brightness, shadows were added at random to images. Images were also flipped at random. The purpose of these functions was to augment the data and thus provide the neural network with robust training examples to help it generalize. The functions were provided by [Vivek Yadev's post on data augmentation](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.hlx538tr0). 
