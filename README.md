 ## Authors:
 * [Drishti Singh](ds6730@nyu.edu)
 
 * [Harshada Sinha](hs4703@nyu.edu)
 
 * [Vaishnavi Rajput](vr2229@nyu.edu)
 
 ## About the project
 
 As part of this mini project, a residual network architecture was implemented and trained with certain objectives. The size of the trainable parameters was required to be strictly under a definite value while experimenting with the various hyper-parameters of the ResNet model to maximize the accuracy on the CIFAR-10 dataset. The model was trained using different optimizers, data augmentation strategies, batch size, epochs, and regularization techniques. Changes in accuracy with respect to different network configurations were noted and compared based on which the final residual network model was selected. 
 
 
 ## Resubmission
 
 
We resubmitted the project1_model.py and project1_model.pt as the file names were not the same in training and test environment. So we made the corrections and retrained the model with proper names and also we were using different normalization values so changed the test script accordingly and ran the self_eval.py test script to test the accuracy and getting 92.9%. We have also excluded any extra libraries from the project1_model.py file to avoid any further failure. 
![Test_accuracy](https://user-images.githubusercontent.com/85714572/161825824-37028b6e-dacb-46de-aa56-6208d4b7cc07.PNG)
