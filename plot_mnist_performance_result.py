import json
import numpy as np
import argparse
import math

from scipy import stats
from distance.dataset_balance import dataset_balance_repartition 


import matplotlib.pyplot as plt

def exp_transform(data):

    return np.exp(data)

def log_transform(data):

    return np.log(data)

def square_transform(data):

    return (data)**2

def root_transform(data):

    return np.sqrt(data)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    
    # parser.add_argument("-f","--filename",type=str,help="Name of the file where experiments will be saved (file extension is not needed)",default="experiment_1")

    with open("./experiments/performance_distance/experiment_1_result.json","r") as file:

        json_value = json.load(file)

    perf_x = [[] for i in range(11)]
    perf_y = [[] for i in range(11)]

    delta_x = [[] for i in range(11)]
    delta_y = [[] for i in range(11)]
 
    disorder_list_x = []
    disorder_list_y = []
    for training_node in json_value:
        
        repartition = training_node['repartition']
        size_list = training_node['sizes']
        
        disorder = dataset_balance_repartition(repartition)
        perf_mean = 0
        data_number = 0

        for index,size_element in enumerate(size_list):

            size = size_element['size']
            training_accuracy = size_element['accuracy']
            testing_list = size_element['result']
          
            temp_mean = 0
            temp_mean_2 = 0
  
         
            for testing_node in testing_list:
                
                distance = testing_node['distance']
                performance = testing_node['accuracy']

                perf_x[index].append(distance)
                perf_y[index].append(performance)

                value = (np.abs(training_accuracy-performance)*distance)/disorder
                
                perf_mean += value
                data_number+=1

                delta_y[index].append((np.abs(training_accuracy-performance)*distance))
                delta_x[index].append(size)

        disorder_list_x.append(disorder)
        disorder_list_y.append(perf_mean/data_number)

    # for i in range(len(perf_x)):
    #     x_tot = np.array(perf_x[i])
    #     y_tot = np.array(perf_y[i])


    #     slope, intercept, r_value, p_value, std_err = stats.linregress(x_tot,y_tot)

    #     print("Correlation coeficient : "+str(r_value))

    #     x1,y1 = 0, intercept
    #     x2,y2 = 0.5,(0.5*slope+intercept)
        
    #     ax = plt.axes(xlabel='Repartition distance',ylabel='Accuracy')

    #     plt.plot(axes=ax)
    #     plt.plot([x1,x2],[y1,y2],color="red")

    #     plt.plot(x_tot,y_tot,"x")
    #     plt.show()
    Mean = []
    for value in delta_y:
        Mean.append(np.mean(value))


   
    plt.figure("Performance variation mean over training dataset size")
    ax = plt.axes(xlabel='Training dataset size',ylabel='(performance variation) * distance mean')

    x =  [1000+ k*(9000//10) for k in range(11)]

    plt.plot(np.array(x),np.array(Mean),axes=ax)
    plt.show()

    x = disorder_list_x
    y = disorder_list_y

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    print("Correlation coeficient : "+str(r_value))



    plt.plot(disorder_list_x,np.sqrt(disorder_list_y),"x")

    plt.show()

    ax = plt.axes(xlabel='Training dataset size',ylabel='(performance variation) * distance')

    # plt.plot(delta_x,delta_y,"x")
    plt.plot(axes=ax)
    plt.boxplot(delta_y)
    plt.xticks([i for i in range(12)],['']+ [1000+ k*(9000//10) for k in range(11)])
    
    plt.savefig("experiments/performance_distance/boxplot.png")
    
    plt.show()

    