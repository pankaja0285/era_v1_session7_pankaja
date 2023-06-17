# Import libraries
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torchvision import datasets, transforms

from model import *


def get_device():
    """
    Check if cuda is available, if available then return the device name, else cpu
    :return: 'cuda' or 'cpu'
    """
    # CUDA?
    use_cuda = torch.cuda.is_available()
    print("CUDA Available?", use_cuda)
    use_device = torch.device("cuda" if use_cuda else "cpu")
    return use_device


def plot_sample_data(data_loader, sample_size=12):
    """
    Plot a sample plot of images from the input data_loader
    :param data_loader: current data loader containing images by batches
    :return: None
    """
    batch_data, batch_label = next(iter(data_loader))

    fig = plt.figure()

    for i in range(sample_size):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        # show the current image using cmap as gray - (grayscale)
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
    fig.show()


def plot_metrics(loss_train, accuracies_train, loss_test, accuracies_test):
    """
    Plot metrics
    :param loss_train: calculated train losses
    :param accuracies_train: accuracies on training
    :param loss_test: calculated test losses
    :param accuracies_test: accuracies on predicting on test data
    :return: None
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(loss_train)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(accuracies_train)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(loss_test)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(accuracies_test)
    axs[1, 1].set_title("Test Accuracy")
    fig.show()


def show_summary(nn_model, input_size):
    """
    Display the trainable and non-trainable parameters, in a user-friendly manner
    :param nn_model: input model for which we want to print the summary
    :param input_size: input image tensor size, for which the neural network is built for the
                       model to be trained on
    :return: None
    """
    summary(nn_model, input_size=input_size)
