from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from dataset_generator.dataset_generator import generate_random_repartition,generate_random_subdataset_repetition
from distance.repartition import distance_repartition_dict

import json

transform = transforms.Compose(
[
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist= datasets.MNIST('../data',
                    transform=transform,
                    download=True
)



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch,log_interval):
    
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
    
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)


def generate_experiments(filename,train_node_number=100,test_node_number=50,dataset_size_min=1000,dataset_size_max=10000,dataset_size_step=10):

    label_list = list(set(mnist.targets.data.numpy()))

    filepath = "./experiments/performance_distance/"
    filepath = filepath+filename+".json"

    size_list = [dataset_size_min + k * (dataset_size_max-dataset_size_min) for k in range(dataset_size_step+1)]

    train_list = [ generate_random_repartition(label_list) for _ in range(train_node_number) ]
    test_list = [ generate_random_repartition(label_list) for _ in range(test_node_number) ]

    result = {'size_list':size_list,'train_list':train_list,'test_list':test_list}

    with open(filepath,"w") as file:

        json.dump(result,file)

def execute_experiments(filename,dataset=mnist):

    filepath = "./experiments/performance_distance/"
        
    with open(filepath+filename+".json",'r') as file:

        experiments = json.load(file)

    size_list = experiments['size_list']
    test_list = experiments['test_list']
    train_list = experiments['train_list']

    # Model initialisation
    batch_size = 64
    test_batch_size = 1000
    epochs = 30
    lr = 1.0
    gamma = 0.7
    no_cuda = False
    seed=1
    log_interval = 10

    use_cuda = not no_cuda and torch.cuda.is_available() 
    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_dictionnary = {}

    print("Step 1 : Dataset generation")
    # Test dataset generation
    # TODO : parralelize building
    for size in size_list:
        
        print("    Dataset size : "+str(size)+ "(started)")
        test_dictionnary[size] = []

        for repartition in test_list:
            
            data,label = generate_random_subdataset_repetition(mnist.data,mnist.targets,repartition,size)
            test_dataset = datasets.MNIST("./data",transform=transform)

            test_dataset.data = data
            test_dataset.targets= label

            test_dictionnary[size].append(test_dataset)

        print("    Dataset size : "+str(size)+ "(end)")

    analitics = []

    print("Step 2 : model training and testing")

    for train_repartition in train_list:
            
        model_analitics ={'repartition':train_repartition}
        model_analitics['sizes'] = []

        for size in size_list:

            size_analitic = {}

            data,label = generate_random_subdataset_repetition(mnist.data,mnist.targets,train_repartition,size)
            train_dataset = datasets.MNIST("./data",transform=transform,train=True)

            
            train_dataset.data = data
            train_dataset.targets= label

            data_test,label_test = generate_random_subdataset_repetition(mnist.data,mnist.targets,train_repartition,size)
            test_dataset = datasets.MNIST("./data",transform=transform)

            test_dataset.data = data_test
            test_dataset.targets= label_test


            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size, 
                shuffle=True, 
                **kwargs
            )

            print(train_loader)

            model = Net().to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=lr)

            scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

            for epoch in range(1, epochs + 1):

                train(model, device, train_loader, optimizer, epoch,log_interval)
                scheduler.step()
            
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size, 
                shuffle=True, 
                **kwargs
            )

            accuracy = test(model,device,test_loader)/100

            size_analitic['accuracy'] = accuracy 
            size_analitic['size'] = size
            size_analitic['result'] = [] 

            for dataset,repartition in zip(test_dictionnary[size],test_list):
                
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=test_batch_size, 
                    shuffle=True, 
                    **kwargs
                )

                accuracy = test(model,device,test_loader)/100
                distance = distance_repartition_dict(train_repartition,repartition)

                analytic = {'accuracy':accuracy,'distance':distance,'repartition':repartition}
            
                size_analitic['result'].append(analytic)
            
            model_analitics['sizes'].append(size_analitic)

        analitics.append(model_analitics)

    with  open(filepath+filename+"_result.json",'w') as file:

        json.dump(analitics,file)          


if __name__ == '__main__':

    if not os.path.isdir('./experiments'):

        os.mkdir('./experiments')

    if not os.path.isdir("./experiments/performance_distance"):

        os.mkdir("./experiments/performance_distance")


    generate_experiments("experiment_1")
    execute_experiments("experiment_1")

    

