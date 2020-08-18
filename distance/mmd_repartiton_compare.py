from dataset_generator.dataset_generator import generate_random_subdataset_repetition,generate_random_repartition,generate_dataset_repartition_list
from distance.repartition import distance_repartition_dict

import numpy as np

from distance.mmd import distance_mmd

from torchvision import datasets, transforms

def experiment(start,end,dataset,step=10,mmd_step=50,subdataset_size=5000):

    print("Step 1: dataset_generation")

    delta = generate_dataset_repartition_list(start,end,step)

    dataset_list = [generate_random_subdataset_repetition(dataset.data,dataset.targets,element,subdataset_size) for element in delta]
    data_list = [data.data.numpy() for (labels,data) in dataset_list]

    delta_x = [ distance_repartition_dict(delta[0],delta[i]) for i in range(len(delta)) ]

    print("Step 2: distance computing")

    delta_y = np.array([0 for i in range(len(data_list))])
    j=0

    for i in range(mmd_step):

        delta_y = delta_y + np.array([distance_mmd(data_list[0],data_list[i],sample_size=100) for i in range(len(data_list))])
        j+=1

        if (j==mmd_step//10):

            j=0
            print("    MMD computing : "+str((i/mmd_step)*100)+"%")

    norm_y = delta_y
    norm_x = delta_x

    return (norm_x,norm_y)