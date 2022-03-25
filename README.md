 ## Authors:
 * [Drishti Singh](ds6730@nyu.edu)
 
 * [Harshada Sinha](hs4703@nyu.edu)
 
 * [Vaishnavi Rajput](vr2229@nyu.edu)
 
 ## About the project
 
 As part of this mini project, a residual network architecture was implemented and trained with certain objectives. The size of the trainable parameters was required to be strictly under a definite value while experimenting with the various hyper-parameters of the ResNet model to maximize the accuracy on the CIFAR-10 dataset. The model was trained using different optimizers, data augmentation strategies, batch size, epochs, and regularization techniques. Changes in accuracy with respect to different network configurations were noted and compared based on which the final residual network model was selected. 
 
 
 ## Introduction
 
 Over the years, deep learning architectures have become deeper and deeper to solve more complex tasks of classification and recognition. However, as the number of layers in the neural network increases, the harder it becomes to train the model due to vanishing gradient problem that results in lower accuracies. 
The ResNet model is one of the most popular and successful deep learning models so far. The architecture allows training up to hundreds or even thousands of layers and still achieves compelling performance. It was introduced by Shaoqing Ren, Kaiming He, Jian Sun, and Xiangyu Zhang in their paper “Deep Residual Learning for Image Recognition” [1](https://arxiv.org/abs/1512.03385) in 2015. ResNet has the ability to tackle the problem of vanishing gradients by using residual blocks with shortcut connections. The shortcut connection simply performs identity mapping, and their outputs are added to the output of the stacked layer.
In our experiment, ResNet architecture was designed and trained for image classification on the CIFAR-10 benchmark dataset. The model was trained in a manner that the number of trainable parameters were < 5M and the accuracy > 80%.



## Objective 

The objective of this project:

1. Modify the provided resnet model and train it on Cifar-10 dataset to get a reasonable test accuracy on [CIFAR 10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

2. The number of trainable parameters should be less than 5M.

3. Try and test different techniques like augmentation, normalization etc to get better accurcay.

4. Try different hyperparameters and find which gave better results.

5. Reach to a final result by comparing different models and their performance.


## Steps to reproduce the results
 
The language for development is python version 3.9 and Numpy version is 1.20.3 as the code is on github  to download the files directly. Also it is recommended installing Anaconda and adding it to the path of Windows.

[Python](https://www.python.org/)

[Github](https://github.com/)

## Clone the repository

Steps to reproduce the result and or clone the repository:

1.  Open the command prompt(Windows)/ terminal(Mac) and change to the directory where you want to clone the repo
2.  Command to clone the repo
```
git clone https://github.com/vrjpt10/Mini-Project-1-Residual-Network-Design.git

```

3. Run the file miniproject1.py 

4. Then run the test.py file following the proper folder stucture.


## Results

We obtained a test accuracy of ~92% after running 50 epochs.

![image](https://user-images.githubusercontent.com/85714572/160190271-1aa02a06-4991-40d0-89ad-5d54ec704cbb.png)

## Final Model

![image](https://user-images.githubusercontent.com/85714572/160190635-dcc91216-e8c9-445e-8d51-2373ca448cbb.png)

### Acknowledgement

  We are grateful to Prof. Siddharth Garg, Prof. Arsalan Mosenia and the teaching assistants for their help and support. Their guidance helped us to learn and implement our learnings through this project.

