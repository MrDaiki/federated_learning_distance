import torch
import random
import numpy as np

def generate_random_repartition(labels_list):

    labels_number = len(labels_list)

    unweigthed_dict = { labels_list[i]:random.uniform(0,1) for i in range(labels_number)}

    total = sum([value for value in unweigthed_dict.values()])

    return {int(labels_list[i]):unweigthed_dict[labels_list[i]]/total for i in range(labels_number)}



def convert_to_dataset(point_list):

    return {i:point_list[i] for i in range(len(point_list))}


def generate_dataset_repartition_list(point_1,point_2,pas):

    size = len(point_1)
    size_test = len(point_2)

    # test if vectors dimensions match
    if size != size_test :
        
        print("Error : probability vectors dimensions doesn't match. None returned")
        return None

    divergence_list = [convert_to_dataset(point_1)]
    
    delta_vector = [(point_2[j]-point_1[j])/pas for j in range(size)]

    for i in range(1,pas+1):

        next = [point_1[j]+i*delta_vector[j] for j in range(size)]
        divergence_list.append(convert_to_dataset(next))

    return(divergence_list)


"""
Test function to generate a sample of labels under probability constraint
verify the strong law of large number, so it is working
"""
def check_random():

    randrange  = [0.5,0.3,0.6,0.2]
    randvalue  = [1,2,3,4]
    range_size = 1000000

    num_max = 0
    num_range_cumulative = [0]

    for element in randrange:

        num_max += element
    
    for element in randrange:

        num_range_cumulative.append(num_range_cumulative[-1]+element/num_max) 

    return_label = []

    for value in range(range_size):

        value = random.uniform(0, 1)

        for j in range(1, len(num_range_cumulative)):

            if (value>= num_range_cumulative[j-1] and value < num_range_cumulative[j] ):

                return_label.append(randvalue[j-1])
                

#dictionarry methods
"""
input : data : torch.tensor, labels : torch.tensor , probability : dict
output : data :torch.tensor, label,torch.tensor

nb : this method does not include data repetition
"""
def partionate_dataset(data,labels,probability):

    return_data   = []
    return_labels = []

    for element,label in zip(data,labels):

        proba = probability[int(label.numpy())]

        if (random.randint(0,100)/100) <= proba :

            return_data.append(element.numpy())
            return_labels.append(int(label.numpy()))


    if (not type(data)==list):
        return_data   = torch.tensor(return_data)
        return_labels = torch.tensor(return_labels)



    return (return_data,return_labels)



"""
input : data : torch.tensor, labels : torch.tensor , probability : dict
output : data :torch.tensor, label,torch.tensor


"""
def generate_random_subdataset_repetition(data,labels,probability,dataset_size):

    data_sorted_list = {}

    num_max = 0
    num_range_cumulative = [0]

    # generation of empty list for sorted datasets by label
    for label,proba in probability.items():

        data_sorted_list[label] = []
        num_max += proba

    for label,proba in probability.items():

        num_range_cumulative.append(num_range_cumulative[-1]+proba/num_max) 


    

    for (value,label) in zip(data,labels):

        if (not type(label)==int):

            data_sorted_list[int(label.numpy())].append(value.numpy())

        else :
            data_sorted_list[label].append(value)


    return_labels = []
    return_data = []

    for value in range(dataset_size):

        value = random.uniform(0, 1)

        for j in range(1, len(num_range_cumulative)):
            
            #choosing a labelunder random probability from dictionnnary probability
            if (value>= num_range_cumulative[j-1] and value < num_range_cumulative[j] ):

                size_of_data = len(data_sorted_list[j-1])
                random_data_row = random.randint(0,size_of_data-1)

    
                return_labels.append(j-1)
                return_data.append(data_sorted_list[j-1][random_data_row])

                continue

                
    if (not type(data)==np.ndarray):


        return_data   = torch.tensor(return_data)
        return_labels = torch.tensor(return_labels)


    return(return_data,return_labels)



    
