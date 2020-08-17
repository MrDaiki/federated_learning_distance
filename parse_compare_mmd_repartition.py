import json
import os

import numpy as np

from torchvision import datasets,transforms

from dataset_generator.dataset_generator import generate_random_repartition
from distance.mmd_repartiton_compare import experiment

mnist = datasets.MNIST('./data',download=True)

def generate_random_experiment(filepath,dataset=mnist,number=100):

    label_list = list(set(dataset.targets.data.numpy()))

    experiment_list = []

    for _ in range(number):
        
        dataset_rep_a = generate_random_repartition(label_list)
        dataset_rep_b = generate_random_repartition(label_list)

        experiment = {'start':dataset_rep_a,'end':dataset_rep_b}
        experiment_list.append(experiment)


    with open(filepath,'w') as file:

        json.dump(experiment_list,file)


def execute_experiment(filepath,dataset=mnist):

    with open(filepath,"r") as file:

        experiments = json.load(file)
    
    
    results = []

    for expe in experiments:
        
        formated_experiment_start = {int(label):data for label,data in expe['start'].items()}
        formated_experiment_end = {int(label):data for label,data in expe['end'].items()}

        x,y = experiment(formated_experiment_start,formated_experiment_end,dataset)

        fx,fy = list(x),list(y)

        results.append({'distance_mmd':fy,'distance_repartition':fx})

    with open(filepath+"_result.json",'w') as file:

        json.dump(results,file)


def prompt_experiments(filepath):

    with open(filepath,"r") as file:

        values = json.load(file)

    for value in values:

        x = np.array(value['distance_repartition'])
        y = np.array(value['distance_mmd'])

        ax = plt.axes(xmin= np.min(x),xmax=np.max(x),ymin=np.min(y),ymax=np.max(y))


d1 = {
    0: 0.1,
    1: 0.1,
    2: 0.1,
    3: 0.1,
    4: 0.1,
    5: 0.1,
    6: 0.1,
    7: 0.1,
    8: 0.1,
    9: 0.1,
}

d2 = {
    0: 0.91,
    1: 0.01,
    2: 0.01,
    3: 0.01,
    4: 0.01,
    5: 0.01,
    6: 0.01,
    7: 0.01,
    8: 0.01,
    9: 0.01,
}

print(experiment(d1,d2,mnist))