import numpy as np


""" 
We compute Shannon entropy over dataset labels
to compute it's imbalancement
details can be found at : https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
return 0 if dataset is very imbalanced
return 1 if dataset is balanced
"""
def dataset_balance(labels,label_list):
    
    dataset_size = len(labels)

    if dataset_size == 0:

        return 0

    labels_numpy = labels.value.numpy()
    entropy = 0

    for label in label_list:
        
        idx = np.where(labels_numpy==label)

        label_size = labels_numpy[idx]

        if label_size > 0:

            entropy += - ((label_size)/dataset_size) * np.log((label_size)/dataset_size)

    return entropy/np.log(len(label_list))


def dataset_balance_repartition(repartition):

    entropy = 0
    label_size = 0

    for _,proportion in repartition.items():
        
        label_size +=1
        entropy+= -proportion * np.log(proportion)

    return entropy/np.log(label_size)