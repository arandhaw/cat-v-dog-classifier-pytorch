# Premption safe example for cat dog

# first party libs
import os
import subprocess
import time
import tempfile

# third party libs
import ray
import ray.train as train
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# function to run bash commands
# print determines whether output is printed
def bash(command, show_result = False):
    try:
        if show_result == True:
            print("Running:", command)
        result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
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
        bash("chmod -R 777 /shared")
        # set custom permissions for shared directory
        bash("sudo apt-get install acl")
        bash("bash chmod g+s /shared")
        bash("setfacl -d -m g::rwx /shared")
        bash("setfacl -d -m o::rwx /shared")
    
    if not os.path.exists("/cat_dog/training_data"):
        print("ERROR!!!!! The shared directory doesn't exist")

# main training function - all logic for training must be contained here
def training_function(train_loop_config):
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

    # Modify batch_size since now each worker does a part of each batch
    batch_size = round(64 / ray.train.get_context().get_world_size())
    # Needed to load data
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Wrap data loader with ray 
    trainloader = ray.train.torch.prepare_data_loader(trainloader)
    testloader = ray.train.torch.prepare_data_loader(testloader)

    # download generic model
    model = models.densenet121(pretrained=True)
    # print(model)

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training using:", device)
    model.to(device)

    # wrap model in ray
    model = ray.train.torch.prepare_model(model)

    traininglosses = []
    testinglosses = []
    testaccuracy = []
    totalsteps = []

    running_loss = 0

    epochs = 1
    print_every = 5
    start_epoch = 0
    start_step = 0
    
    checkpoint = ray.train.get_checkpoint()  # returns most recent checkpoint, or None 
    # load checkpoint if it exists
    if checkpoint:
        print("Loading checkpoint")
        with checkpoint.as_directory() as checkpoint_dir:
            state_dict = torch.load(checkpoint_dir + "/checkpoint.pth")
            model.load_state_dict(state_dict['model_state_dict'])
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            start_epoch = state_dict['epoch']
            start_step = state_dict['step']

    for epoch in range(start_epoch, epochs):
        # this shuffles the data each epoch
        if ray.train.get_context().get_world_size() > 1:
            trainloader.sampler.set_epoch(epoch)
            testloader.sampler.set_epoch(epoch)

        for step, (inputs, labels) in enumerate(trainloader, start=start_step):
            if step > 100: 
                break
                
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (step + 1) % print_every == 0:
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
                totalsteps.append(step)
                print(f"Device {device}.."
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Step {step}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()

                # data to save
                metrics = {"training loss" : traininglosses[-1], 
                            "testing loss" : testinglosses[-1], 
                            "testing accuracy" : testaccuracy[-1], 
                            "epoch" : epoch,
                            "step" : step
                            }
                checkpoint = {
                    'parameters' : model.parameters,
                    'state_dict' : model.state_dict(),
                    'metrics' : metrics
                }
                # save to any directory
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    torch.save({
                        'metrics' : metrics,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, temp_checkpoint_dir + "/checkpoint.pth")
                    # create a ray report with metrics and the checkpoint path
                    ray.train.report(
                        metrics,
                        checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
                    )
    print("Finished training")

trainer = ray.train.torch.TorchTrainer(
    training_function,
    train_loop_config={},
    scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=True),
    # [5a] If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
    run_config=ray.train.RunConfig(storage_path="/shared", name = "new_world=pre",
                                failure_config=train.FailureConfig(max_failures=2))
)

result = trainer.fit()

result_path = result.checkpoint.path
print("Path of final checkpoint", result_path)

# savedir = '/home/arand/cat-v-dog-classifier-pytorch/results/'
# bash(f"sudo cp -r {result_path} {savedir}")