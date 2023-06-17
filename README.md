### Purpose: To modularize the assignment notebook from ERA - Session 7 into script files and properly import the functions into a new notebook (S5.ipynb) and run to reproduce results.

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
|__era1_S7_4_ Fully Connected layer.ipynb<br/>
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

**NOTE:** Follow along the **S5.ipynb** - notebook cells and run sequentially to see the outputs.

### Notebook execution sequence of sections is as follows:
**File used: era1_S7_0_BasicSetup.ipynb**
 1. Under section **Install Libraries**, run the cell(s) - to install required libraries 
 2. Under section **Import Libraries** run the cell(s)
        - to import the script functions from model.py, utils.py
 3. Under section **Download train and test dataset** run the cell(s)
        - it will automatically dowload and process the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset for you, mounts the pytorch mnist dataset in the `../data` folder inside the container for our purposes, so that we do not need to download it at training time. 
 4. Under section **Plot random sample of train data**, run the cell(s)
        - to plot random sample in a 3 X 4 grid, to visualize the sample data
 5. Under section **Train the model on train data and predict on test data**, run the cell(s)
        - to train a simple Convolutional Neural Network model on the train data and predict for the test data 
 6. Under section **Plot metrics - training losses and accuracies, along with test losses and test accuracies**, run the cell(s)
        - to plot the losses and accuracies of training and testing datasets
 7. Under section **Display model summary**, run cell(s)
        - to display the model summary on a 28 X 28 with one gray channel on which the model is structured for.

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
