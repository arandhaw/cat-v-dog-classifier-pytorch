#!/usr/bin/env python
# coding: utf-8
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import ray
import os
import subprocess
import logging

import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

output_dir = "/home/arand/cat-v-dog-classifier-pytorch/results"

# function to run bash commands
# print determines whether output is printed
def bash(command, show_result = True):
    try:
        print("Trying:", command)
        result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
        if show_result:
            print(result.stdout)
    except Exception as e:
        print("Error occured during:", command)
        print("Error message:\n")
        print(e)

# loads data on remote machine and returns directory
def load_data():
    if not os.path.exists("/cat_dog/training_data"):
        bash("sudo mkdir /cat_dog")
        bash("sudo chmod ugo+rwx /cat_dog")
        bash("sudo gsutil cp -r gs://yakoa-model-data/Cats_and_dogs/* /cat_dog")
        bash("sudo unzip /cat_dog/training_data.zip -d /cat_dog", show_result = False)
    if not os.path.exists("/cat_dog/training_data"):
        print("ERROR!!!!! The data directory doesn't exist")
    return "/cat_dog/training_data"

@ray.remote(num_gpus = 1)
def train_model():
    data_dir = load_data()
    print(os.listdir(data_dir))

    # Define transforms for the training data and testing data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    # We can load in a model such as [DenseNet](http://pytorch.org/docs/0.3.0/torchvision/models.html#id5). Let's print out the model architecture so we can see what's going on.
    model = models.densenet121(pretrained=True)
    print(model)
    # This model is built out of two main parts, the features and the classifier. The features part is a stack of convolutional layers and overall works as a feature detector that can be fed into a classifier. The classifier part is a single fully-connected layer `(classifier): Linear(in_features=1024, out_features=1000)`. This layer was trained on the ImageNet dataset, so it won't work for our specific problem. That means we need to replace the classifier, but the features will work perfectly on their own. In general, I think about pre-trained networks as amazingly good feature detectors that can be used as the input for simple feed-forward classifiers.

    # In[4]:


    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, 512)),
                            ('relu1', nn.ReLU()),
                            ('fc2', nn.Linear(512,256)),
                            ('relu2', nn.ReLU()),
                            ('fc3', nn.Linear(256, 2)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
    model.classifier = classifier


    # With our model built, we need to train the classifier. However, now we're using a **really deep** neural network. If you try to train this on a CPU like normal, it will take a long, long time. Instead, we're going to use the GPU to do the calculations. The linear algebra computations are done in parallel on the GPU leading to 100x increased training speeds. It's also possible to train on multiple GPUs, further decreasing training time.
    # 
    # PyTorch, along with pretty much every other deep learning framework, uses [CUDA](https://developer.nvidia.com/cuda-zone) to efficiently compute the forward and backwards passes on the GPU. In PyTorch, you move your model parameters and other tensors to the GPU memory using `model.to('cuda')`. You can move them back from the GPU with `model.to('cpu')` which you'll commonly do when you need to operate on the network output outside of PyTorch. As a demonstration of the increased speed, I'll compare how long it takes to perform a forward and backward pass with and without a GPU.

    # In[4]:

    # In[6]:


    # Try to replace with just ['cuda'] if you are using GPU 

    for device in ['cpu', 'cuda']:

        criterion = nn.NLLLoss()
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

        model.to(device)

        for ii, (inputs, labels) in enumerate(trainloader):

            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            start = time.time()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if ii==3:
                break
            
        print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")


    # You can write device agnostic code which will automatically use CUDA if it's enabled like so:
    # ```python
    # # at beginning of the script
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 
    # ...
    # 
    # # then whenever you get a new Tensor or Module
    # # this won't copy if they are already on the desired device
    # input = data.to(device)
    # model = MyModule(...).to(device)
    # ```
    # 
    # From here, I'll let you finish training the model. The process is the same as before except now your model is much more powerful. You should get better than 95% accuracy easily.
    # 
    # Train a pretrained models to classify the cat and dog images. Continue with the DenseNet model, or try ResNet, it's also a good model to try out first. Make sure you are only training the classifier and the parameters for the features part are frozen.

    # In[5]:


    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training using:", device)
    model = models.densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(256, 2),
                                    nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    model.to(device)


    # In[7]:


    traininglosses = []
    testinglosses = []
    testaccuracy = []
    totalsteps = []
    
    epochs = 1
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                traininglosses.append(running_loss/print_every)
                testinglosses.append(test_loss/len(testloader))
                testaccuracy.append(accuracy/len(testloader))
                totalsteps.append(steps)
                print(f"Device {device}.."
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Step {steps}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()   # switch back to train mode

    checkpoint = {
        'parameters' : model.parameters,
        'state_dict' : model.state_dict()
    }
    # modelref = ray.put(checkpoint)  # doesn't work
    torch.save(checkpoint, "/mnt/shared/catdogmodel.pth")
    # return stuff
    training_record = (traininglosses, testinglosses, testaccuracy, totalsteps)
    return training_record

training_record = ray.get(train_model.remote())
(traininglosses, testinglosses, testaccuracy, totalsteps) = training_record

plt.plot(totalsteps, traininglosses, label='Train Loss')
plt.plot(totalsteps, testinglosses, label='Test Loss')
plt.plot(totalsteps, testaccuracy, label='Test Accuracy')
plt.legend()
plt.grid()

savedir = '/home/arand/cat-v-dog-classifier-pytorch/results'
plt.savefig("/mnt/shared/training_progress.png")

bash(f"mv /mnt/shared/training_progress.png {savedir}")

bash(f"mv /mnt/shared/catdogmodel.pth {savedir}")