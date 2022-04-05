 ## Authors:
 * [Drishti Singh](ds6730@nyu.edu)
 
 * [Harshada Sinha](hs4703@nyu.edu)
 
 * [Vaishnavi Rajput](vr2229@nyu.edu)
 
 ## About the project
 
 As part of this mini project, a residual network architecture was implemented and trained with certain objectives. The size of the trainable parameters was required to be strictly under a definite value while experimenting with the various hyper-parameters of the ResNet model to maximize the accuracy on the CIFAR-10 dataset. The model was trained using different optimizers, data augmentation strategies, batch size, epochs, and regularization techniques. Changes in accuracy with respect to different network configurations were noted and compared based on which the final residual network model was selected. 
 
 ## Introduction
 
 Over the years, deep learning architectures have become deeper and deeper to solve more complex tasks of classification and recognition. However, as the number of layers in the neural network increases, the harder it becomes to train the model due to vanishing gradient problem that results in lower accuracies. The ResNet model is one of the most popular and successful deep learning models so far. The architecture allows training up to hundreds or even thousands of layers and still achieves compelling performance. It was introduced by Shaoqing Ren, Kaiming He, Jian Sun, and Xiangyu Zhang in their paper “Deep Residual Learning for Image Recognition” 1 in 2015. ResNet has the ability to tackle the problem of vanishing gradients by using residual blocks with shortcut connections. The shortcut connection simply performs identity mapping, and their outputs are added to the output of the stacked layer. In our experiment, ResNet architecture was designed and trained for image classification on the CIFAR-10 benchmark dataset. The model was trained in a manner that the number of trainable parameters were < 5M and the accuracy > 80%.
 
 ## Objective
 
  The objective of this project:

Modify the provided resnet model and train it on Cifar-10 dataset to get a reasonable test accuracy on CIFAR 10 dataset.

1. The number of trainable parameters should be less than 5M.

2. Try and test different techniques like augmentation, normalization etc to get better accurcay.

3. Try different hyperparameters and find which gave better results.

4. Reach to a final result by comparing different models and their performance.

## Steps to reproduce the results

The language for development is python version 3.9 and Numpy version is 1.20.3 as the code is on github to download the files directly. Also it is recommended installing Anaconda and adding it to the path of Windows.

Python(https://www.python.org/)

Github(https://github.com/)

### Clone the repository
Steps to reproduce the result and or clone the repository:

1. Open the command prompt(Windows)/ terminal(Mac) and change to the directory where you want to clone the repo
2. Command to clone the repo
 
   ```bash
  git clone https://github.com/vrjpt10/Mini-Project-1-Residual-Network-Design.git
  ```
 ### To run the model then test
 
Run the file miniproject1.py

Then run the test.py file following the proper folder stucture.

The miniproject1.py was basically executed on Google colab and for testing requires all the implicit libraries alraedy imported before the testing is done.
 
 ## Results
 
 (https://user-images.githubusercontent.com/85714572/160190271-1aa02a06-4991-40d0-89ad-5d54ec704cbb.png)
 
 ## Final Model
 
 (https://user-images.githubusercontent.com/85714572/160190635-dcc91216-e8c9-445e-8d51-2373ca448cbb.png)
 
 ### Acknowledgement
 
 We are grateful to Prof. Siddharth Garg, Prof. Arsalan Mosenia and the teaching assistants for their help and support. Their guidance helped us to learn and implement our learnings through this project.
 
 ## Resubmission
 
We resubmitted the project1_model.py and project1_model.pt as the file names were not the same in training and test environment. So we made the corrections and retrained the model with proper names and also we were using different normalization values so changed the test script accordingly and ran the self_eval.py test script to test the accuracy and getting 92.9%. We have also excluded any extra libraries from the project1_model.py file to avoid any further failure. 
![Test_accuracy](https://user-images.githubusercontent.com/85714572/161825824-37028b6e-dacb-46de-aa56-6208d4b7cc07.PNG)

 
 ## Resubmission
 
We resubmitted the project1_model.py and project1_model.pt as the file names were not the same in training and test environment. So we made the corrections and retrained the model with proper names and also we were using different normalization values so changed the test script accordingly and ran the self_eval.py test script to test the accuracy and getting 92.9%. We have also excluded any extra libraries from the project1_model.py file to avoid any further failure. 
![Test_accuracy](https://user-images.githubusercontent.com/85714572/161825824-37028b6e-dacb-46de-aa56-6208d4b7cc07.PNG)
