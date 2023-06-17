### Purpose: Deep dive into coding and applying different blocks in 7 steps.

## Based on MNIST dataset
### Create a simple Convolutional Neural Network model and predict

### Project Setup:
Clone the project as shown below:-

```bash
$ git clone git@github.com:pankaja0285/era_v1_session7_pankaja.git
$ cd era_v1_session7_pankaja
```
About the file structure</br>
|__era1_S7_0_BasicSetup.ipynb<br/>
|__era1_S7_1_BasicSkeleton.ipynb<br/>
|__era1_S7_2_Batch_Normalization.ipynb<br/>
|__era1_S7_3_Dropout.ipynb<br/>
|__era1_S7_4_Fully_Connected_layer.ipynb<br/>
|__era1_S7_5_Augmentation.ipynb<br/>
|__era1_S7_6_LRScheduler.ipynb<br/>
|__model.py<br/>
|__README.md<br/>
|__requirements.txt<br/>
|__utils.py<br/>

**NOTE:** List of libraries required: ***torch*** and ***torchsummary***, ***tqdm*** for progress bar, which are installed using requirements.txt<br/>

One of 2 ways to run any of the notebooks, for instance **era1_S7_0_BasicSetup.ipynb** notebook:<br/>
1. Using Anaconda prompt - Run as an **administrator** start jupyter notebook from the folder ***era_v1_session7_pankaja*** and run it off of your localhost<br/>
**NOTE:** Without Admin privileges, the installs will not be correct and further import libraries will fail. <br/>
```
jupyter notebook
```
2. Upload the notebook folder ***era_v1_session7_pankaja*** to google colab at [colab.google.com](https://colab.research.google.com/) and run it on colab<br/>

###
**NOTE:** Follow along the **S5.ipynb** - notebook cells and run sequentially to see the outputs.

### Step 0:
**File used: era1_S7_0_BasicSetup.ipynb**
 
### Step 1:
**File used: era1_S7_1_BasicSkeleton.ipynb**

### Step 2:
**File used: era1_S7_2_Batch_Normalization.ipynb**

### Step 3:
**File used: era1_S7_3_Dropout.ipynb**

### Step 4:
**File used: era1_S7_4_Fully_Connected_layer.ipynb**

### Step 5:
**File used: era1_S7_5_Augmentation.ipynb**

### Step 6:
**File used: era1_S7_6_LRScheduler.ipynb**

### Python script files - details:
**model.py** - This has Model_1, Model_2, Model_3, Model_4, Model_5, Model_6, Model_7 <br />
in all 7 models to achieve an train accuracy = 99.5 and test accuracy = 98.6

*The illustration below shows how many layers are in this Convolution Neural Network that we have based upon and its details:-*
![CNN diagram used](cnn_28_x_28.png)

**utils.py** - This file contains the following main functions
* get_device() - checks for device availability for cuda, if not gives back cpu as set device
* plot_sample_data() - plots a sample grid of random 12 images from the training data
* plot_metrics() - plots the metrics - train and test - losses and accuracies respectively
* show_summary() - displays the model summary with details of each layer
* download_train_data() - downloads train data from MNIST
* download_test_data() - downloads test data from MNIST
* create_data_loader() - common data loader function using which we create both train_loader and test_loader by appropriately passing required arguments
* train_and_predict() - trains the CNN model on the training data and uses the trained model to predict on the test data

### More resources:
Some useful resources on MNIST and ConvNet:

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Colah's blog](https://colah.github.io/posts/2014-10-Visualizing-MNIST/)
- [FloydHub Building your first ConvNet](https://blog.floydhub.com/building-your-first-convnet/)
- [How Convolutional Neural Networks work - Brandon Rohrer](https://youtu.be/FmpDIaiMIeA)
- [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
- [Stanford CS231n](https://cs231n.github.io/convolutional-networks/)
- [Stanford CS231n Winter 2016 - Karpathy](https://youtu.be/NfnWJUyUJYU)

### Contributing:
For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!
