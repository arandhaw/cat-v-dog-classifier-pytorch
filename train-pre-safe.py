# built-in libs
import time
import os
import subprocess
import logging

# third party libs
import ray
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# convenience function for bash commands
def bash(command, show_result = False):
    try:
        if show_result == True:
            print("Running:", command)
        result = subprocess.run(command, shell = True, capture_output=True, text=True, check=True)
        if show_result:
            print(result.stdout)
    except Exception as e:
        print("Bash command failed:", command)
        print("Error message\n", e)

def setup_environment():
    # download data from bucket
    if not os.path.exists("/cat_dog/training_data"):
        bash("sudo mkdir /cat_dog")
        bash("sudo chmod ugo+rwx /cat_dog")
        bash("sudo gsutil cp -r gs://yakoa-model-data/Cats_and_dogs/* /cat_dog")
        bash("sudo unzip /cat_dog/training_data.zip -d /cat_dog", show_result = False)
    if not os.path.exists("/cat_dog/training_data"):
        print("ERROR!!!!! Could not get data from bucket")
    # setup shared directory
    if not os.path.exists("/shared/hello-world"):
        bash("sudo apt-get -y install nfs-common")
        bash("sudo mkdir -p /shared")
        bash("sudo mount 10.0.24.154:/vol1 /shared")
        bash("sudo chmod -R 777 /shared")
        # set custom permissions for shared directory
        bash("sudo apt-get install acl")
        bash("sudo chmod g+s /shared")
        bash("setfacl -d -m g::rwx /shared")
        bash("setfacl -d -m o::rwx /shared")
    
    if not os.path.exists("/cat_dog/training_data"):
        print("ERROR!!!!! The shared directory doesn't exist")

@ray.remote(num_gpus = 1)
def train_model():
    # get shared file system set up
    setup_environment()
    data_dir = "/cat_dog/training_data"
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
    # print(model)
    # This model is built out of two main parts, the features and the classifier. 
    # The features part is a stack of convolutional layers and overall works as a feature detector
    # that can be fed into a classifier. 
    # The classifier part is a single fully-connected layer 
    # This layer was trained on the ImageNet dataset, so it won't work for our specific problem. 
    # That means we need to replace the classifier, but the features will work perfectly on their own. 
    # In general, I think about pre-trained networks as amazingly good feature detectors that can 
    # be used as the input for simple feed-forward classifiers.

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

    # loss function
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training using:", device)
    model.to(device)

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
    torch.save(checkpoint, "/shared/catdogmodel.pth")
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

savedir = '/home/arand/cat-v-dog-classifier-pytorch/models'
plt.savefig('/shared/figure.png')

bash(f"sudo mv /shared/catdogmodel.pth {savedir}")
bash(f"sudo mv /shared/figure.png {savedir}")