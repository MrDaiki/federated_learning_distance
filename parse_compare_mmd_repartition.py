import json
import os

import numpy as np

from torchvision import datasets,transforms

from dataset_generator.dataset_generator import generate_random_repartition
from distance.mmd_repartiton_compare import experiment

import matplotlib.pyplot as plt

mnist = datasets.MNIST('./data',download=True)

def generate_random_experiment(filepath,filename,dataset=mnist,number=100):

    label_list = list(set(dataset.targets.data.numpy()))

    experiment_list = []

    for _ in range(number):
        
        dataset_rep_a = generate_random_repartition(label_list)
        dataset_rep_b = generate_random_repartition(label_list)

        experiment = {'start':dataset_rep_a,'end':dataset_rep_b}
        experiment_list.append(experiment)


    with open(filepath+filename+'.json','w') as file:

        json.dump(experiment_list,file)


def execute_experiment(filepath,filename,dataset=mnist):

    with open(filepath+filename+".json","r") as file:

        experiments = json.load(file)
    
    
    results = []

    i = 1
    for expe in experiments:

        print("=================== Experiment "+str(i)+" ====================")
        print(" ")
        formated_experiment_start = {int(label):data for label,data in expe['start'].items()}
        formated_experiment_end = {int(label):data for label,data in expe['end'].items()}

        x,y = experiment(formated_experiment_start,formated_experiment_end,dataset)

        fx,fy = list(x),list(y)

        print("Step 4 : Saving results")
        results.append({'distance_mmd':fy,'distance_repartition':fx})
        print(" ")

    with open(filepath+filename+"_result.json",'w') as file:

        json.dump(results,file)


if __name__ == '__main__':


    #setup of experiments files
    filepath = './experiments/mmd_repartition_compare/'

    if not os.path.isdir('./experiments'):

        os.mkdir('./experiments')

    if not os.path.isdir('./experiments/mmd_repartition_compare'):

        os.mkdir('./experiments/mmd_repartition_compare')

    filename = 'experiment_1'
    

    generate_random_experiment(filepath,filename,dataset=mnist,number=1)

    execute_experiment(filepath,filename,dataset=mnist)
    