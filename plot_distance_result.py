import json
import numpy as np

import matplotlib.pyplot as plt

def prompt_experiments(filepath):

    with open(filepath,"r") as file:

        values = json.load(file)

    for value in values:

        x = np.array(value['distance_repartition'])
        y = np.array(value['distance_mmd'])

        ax = plt.axes(xlim=(np.min(x),np.max(x)),ylim=(np.min(y),np.max(y)),xlabel='class repartition distance',ylabel='maximum mean discrepancy')
        
        plt.plot(x,y,axes=ax)

        plt.show()



if __name__ == '__main__' : 

    prompt_experiments('experiments/mmd_repartition_compare/experiment_1_result.json')