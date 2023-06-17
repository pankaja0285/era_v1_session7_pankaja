import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR

from tqdm import tqdm

from utils import get_device


# Begin: model structure 1 - 6 ***************************************************************************
class Model_1(nn.Module):
    """
    Class for creating a neural network, to be used for creating a model
    and load on to the device and use it for training the data and predict on
    test data.
    """
    def __init__(self):
        super(Model_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14 > 14 | 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # 14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(256, 512, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(512, 1024, 3) # 5 > 3 | 32 | 3*3*1024 | 3x3x1024x10 | 
        self.conv7 = nn.Conv2d(1024, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

# Model_2
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()

        # define convolution block1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=0, bias=False),  # 28x28 output 28x28 RF : 3x3
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=0, bias=False), # 28x28 output 28x28 RF : 5x5
            nn.ReLU(),                    
        )

        # Transition Block (MaxPool + 1x1)
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            # 1x1 convolution
            nn.Conv2d(16, 8, 1, bias=False), # 26x26 output - 26x26 RF 14x14
            nn.ReLU(),
            # 26x26 output - 13x13 RF 14x14
        )

        # define convolution block2
        self.conv2 =  nn.Sequential(
            nn.Conv2d(8, 10, 3, padding=0, bias=False), # 13x13 output - 11x11 RF 16x16
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=0, bias=False),  # 11x11 output - 9x9 RF 18x18
            nn.ReLU(),
            nn.Conv2d(10, 16, 3, padding=0, bias=False), # 9x9 output - 7x7 RF 20x20
            nn.ReLU(),          
        )
        
        # final convolution output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )  
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool2d(x)
        x = x.view(-1,10)

        return F.log_softmax(x,dim=1)


class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()


        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3,padding=0,bias=False),  # 28x28 output 28x28 RF : 3x3
            nn.BatchNorm2d(8),
            nn.Dropout(0.10),
            nn.ReLU(),

            nn.Conv2d(8, 16, 3,padding=0,bias=False), # 28x28 output 28x28 RF : 5x5
            nn.BatchNorm2d(16),
            nn.Dropout(0.10),
            nn.ReLU(),

                    
        )

        # Transition Block 
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            # 1x1 convolution
            nn.Conv2d(16, 8, 1,bias=False), # 26x26 output - 26x26 RF 14x14
            nn.BatchNorm2d(8),
            nn.Dropout(0.10),
            nn.ReLU(),

              # 26x26 output - 13x13 RF 14x14

        )

        # Convolutional Block 2
        self.conv2 =  nn.Sequential(

            nn.Conv2d(8, 10, 3,padding=0, bias=False), # 13x13 output - 11x11 RF 16x16
            nn.BatchNorm2d(10),
            nn.Dropout(0.10),
            nn.ReLU(),

            nn.Conv2d(10, 16, 3,padding=0, bias=False),  # 11x11 output - 9x9 RF 18x18
            nn.BatchNorm2d(16),
            nn.Dropout(0.10),
            nn.ReLU(),

            nn.Conv2d(16, 10, 3,padding=0, bias=False), # 9x9 output - 7x7 RF 20x20
            nn.BatchNorm2d(10),
            nn.Dropout(0.10),
            nn.ReLU(),
        )
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)

        

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.avgpool2d(x)
        #x = self.conv3(x)
        x = x.view(-1,10)

        return F.log_softmax(x,dim=1)
    
class Model_4(nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()


        #Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3,padding=0,bias=False),  # 28x28 output 28x28 RF : 3x3
            nn.BatchNorm2d(8),
            nn.Dropout(0.10),
            nn.ReLU(),

            nn.Conv2d(8, 16, 3,padding=0,bias=False), # 28x28 output 28x28 RF : 5x5
            nn.BatchNorm2d(16),
            nn.Dropout(0.10),
            nn.ReLU(),                    
        )

        #Transition Block 
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            # 1x1 convolution
            nn.Conv2d(16, 8, 1,bias=False), # 26x26 output - 26x26 RF 14x14
            nn.BatchNorm2d(8),
            nn.Dropout(0.10),
            nn.ReLU(),

            # 26x26 output - 13x13 RF 14x14
        )

        # Convolution Block 2
        self.conv2 =  nn.Sequential(

            nn.Conv2d(8, 10, 3,padding=0, bias=False), # 13x13 output - 11x11 RF 16x16
            nn.BatchNorm2d(10),
            nn.Dropout(0.10),
            nn.ReLU(),

            nn.Conv2d(10, 16, 3,padding=0, bias=False),  # 11x11 output - 9x9 RF 18x18
            nn.BatchNorm2d(16),
            nn.Dropout(0.10),
            nn.ReLU(),

            nn.Conv2d(16, 10, 3,padding=0, bias=False), # 9x9 output - 7x7 RF 20x20
            nn.BatchNorm2d(10),
            nn.Dropout(0.10),
            nn.ReLU(),
        )
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)

        

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.avgpool2d(x)
        #x = self.conv3(x)
        x = x.view(-1,10)

        return F.log_softmax(x,dim=1)
    

class Model_5(nn.Module):
    def __init__(self, dropout_value=0.03):
        super(Model_4, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        ) 

        #Transition Block
        self.trans1 = nn.Sequential(
            
            nn.MaxPool2d(2, 2), # output_size = 12
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 24
        

        # CONVOLUTION BLOCK 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),

        ) 
        
        # OUTPUT BLOCK
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.avgpool2d(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

class Model_6(nn.Module):
    def __init__(self, dropout_value=0.03):
        super(Model_6, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        ) 

        # Transition Block
        self.trans1 = nn.Sequential(
            
            nn.MaxPool2d(2, 2), # output_size = 12
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 24        

        # Convolution Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        ) 
        
        # Output Block
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.avgpool2d(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

class Model_7(nn.Module):
    def __init__(self, dropout_value=0.03):
        super(Model_7, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), # Input 28x28 output 26x26 RF : 3x3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # Input 26x26 output 24x24 RF : 5x5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 

        #Transition Block
        self.trans1 = nn.Sequential(
            
            nn.MaxPool2d(2, 2), # output_size = 12
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)  # Input 12x12 output 12x12 RF : 6x6
        ) # output_size = 24
        

        # CONVOLUTION BLOCK 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),   # Input 12x12 output 10x10 RF : 6x6
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),  # Input 10x10 output 8x8 RF : 10x10
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=16, out_channels=18, kernel_size=(3, 3), padding=0, bias=False),  # Input 8x8 output 6x6 RF : 14x14
            nn.ReLU(),            
            nn.BatchNorm2d(18),
            nn.Dropout(dropout_value)

        ) 
        
        # OUTPUT BLOCK
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=16, kernel_size=(1, 1), padding=0, bias=False), # Input 6x6 output 6x6 RF : 18x18
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value))

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)  # Input 6x6 output 6x6 RF : 18x18


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.avgpool2d(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
# **************************************************************************************************    
# **********   Below can be moved to a dataset.py at a later point ****************
def download_model_data(mean=(0.1307,), std=(0.3081,),
                       download_path='../data'):
    """
    Download MNIST - test data
    :param mean: defaults to (0.1307, )
    :param std: defaults to (0.3081, )
    :param download_path: defaults to ../data, can be modified to any other path and even provide it as
                          parameter in a config yaml file for instance, same goes for all other configurations
    :return: test_dataset: test dataset
    """

    # Train Phase transformations
    train_transforms = transforms.Compose([
                                        # transforms.Resize((28, 28)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                        transforms.Normalize(mean, std)   
                                        ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                          #  transforms.Resize((28, 28)),                                          
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)
                                           ])
    
    # apply the train transformation and download the train dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    # apply the test transformation and download the test dataset
    test_dataset = datasets.MNIST(download_path, train=False, download=True, transform=test_transforms)
    return train_dataset, test_dataset


# End: model structure 1 - 9 ***************************************************************************

# **************   Train the model - this function can be moved to a train.py later, if we choose to *********
def get_correct_prediction_count(p_prediction, p_labels):
    # apply argmax and return the predictions
    return p_prediction.argmax(dim=1).eq(p_labels).sum().item()


def train(model, device, train_loader, optimizer,
          train_losses, train_acc, use_LRScheduler=False, scheduler=None):
    """
    Train the data
    :param model: model to train the data on
    :param device: device name
    :param train_loader: train loader
    :param optimizer: optimizer
    :param train_losses: training data losses, so that the current losses can be appended
    :param train_acc: training data accuracies, so that the current accuracies can be appended
    :return: current accumulated training data losses and accuracies
    """
    # test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0    
    
    # process the data in the training dataset in batches, loading to the device
    # and apply training processes
    for batch_idx, (data, target) in enumerate(pbar):
        # step: load data and target to devide
        data, target = data.to(device), target.to(device)
        # step: apply zero_grad - zero out the gradients,
        #       so that you do the parameter update correctly
        optimizer.zero_grad()

        # step: predict
        pred = model(data)

        # step: calculate loss
        loss = F.nll_loss(pred, target)
        train_loss += loss.item()

        # step: backpropagation
        loss.backward()
        # step: call the step function
        optimizer.step()
        if use_LRScheduler:
            scheduler.step()

        # step: get the prediction count
        correct += get_correct_prediction_count(pred, target)
        processed += len(data)

        # step: set the description for the tqdm progress bar
        pbar.set_description(
            desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

    # append the current accuracy to the train_acc array
    train_acc.append(100 * correct / processed)
    # append the current loss to the train_losses array
    train_losses.append(train_loss / len(train_loader))

    # return the list train losses and accuracies
    return train_losses, train_acc


def test(model, device, test_loader,
         test_losses, test_acc):
    """
    Load test data and predict using the model trained on training data
    :param model: trained model
    :param device: device name
    :param test_loader: test loader with test data
    :param test_losses: test losses, so that the current losses can be appended
    :param test_acc: test accuracies, so that the current accuracies can be appended
    :return: current accumulated test data losses and accuracies
    """
    model.eval()

    test_loss = 0
    correct = 0

    # proceed without applying any gradients, as we are predicting for test data
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # step: load data and target to device
            data, target = data.to(device), target.to(device)

            # step: predict
            output = model(data)
            # step: calculate the loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            # get the prediction count
            correct += get_correct_prediction_count(output, target)

    test_loss /= len(test_loader.dataset)
    # append the current accuracy to the test_acc array
    test_acc.append(100. * correct / len(test_loader.dataset))
    # append the current loss to the test_losses array
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return the list of test losses and accuracies
    return test_losses, test_acc


def create_data_loader(data, batch_size=512, shuffle=False, num_workers=2):
    """
    Create a data loader
    :param data: input data
    :param batch_size: defaults to 512
    :return: data_loader: data loader for the data passed as input
    """

    # assemble the data_loader parameters
    #   batch_size: how many images to use at a time to train/test
    #   shuffle: to obtain a random sample of images, so we reduce the bias
    #   num_workers: number of worker threads to use
    #   pin_memory: True, supposed to speed up the transfer of data from host to device
    #               - again this is sort of debatable depending on the version of pytorch used,
    #               as seen in this link:
    #               https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/19
    kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers, 'pin_memory': True}

    data_loader = torch.utils.data.DataLoader(data, **kwargs)
    return data_loader


def train_and_predict(nn_model, device, train_loader, test_loader, 
                      num_epochs=20, lr=0.01, max_lr=0.015, use_LRScheduler = True):
    """
    Train the model on training data and predict on test data
    :param device: device name
    :param train_loader: training data loader
    :param test_loader: test data loader
    :param num_epochs: number of epochs to run, default 20
    :return: nn_model, so we can print the summary
    """
    step = ""
    # initialize the losses and acc for train and test
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []
    scheduler = None

    try: 
        # optimizer
        optimizer = optim.SGD(nn_model.parameters(), lr=lr, momentum=0.9)
        if use_LRScheduler:
            scheduler = OneCycleLR(optimizer, max_lr=max_lr, epochs=15, steps_per_epoch=len(train_loader))
        # step: run for num_epochs + 1 starting from epoch 1
        for epoch in range(1, num_epochs + 1):
            msg = f'Epoch {epoch}'
            print(msg)
            # train with train_loader in the current epoch
            step = f"train {msg}"
            train_losses, train_acc = train(nn_model, device, train_loader, optimizer,
                                            train_losses, train_acc, 
                                            use_LRScheduler=use_LRScheduler, scheduler=scheduler
                                            )
            # using the trained model, test with test_loader in the current epoch
            step = f"test {msg}"
            test_losses, test_acc = test(nn_model, device, test_loader,
                                         test_losses, test_acc)
    except Exception as ex:
        print(f"Error occurred at step - {step},\n{ex}")
    return nn_model, train_losses, train_acc, test_losses, test_acc
    