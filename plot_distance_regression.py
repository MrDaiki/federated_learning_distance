import json
import numpy as np
import argparse

from scipy import stats

import matplotlib.pyplot as plt

def exp_transform(data):

    return np.exp(data)

def log_transform(data):

    return np.log(data)

def square_transform(data):

    return (data)**2

def root_transform(data):

    return np.sqrt(data)


def plot_all_points(filepath,x_transform=None,y_transform=None):

    with open(filepath,"r") as file:

         values = json.load(file)

    x_tot = []
    y_tot = []

    fig = plt.figure("Regression MMD = rd")
    ax = plt.axes(xlabel='Repartition distance',ylabel='Maximum mean discrepancy')

    for element in values:

        x_list = element['distance_repartition']
        y_list = element['distance_mmd']


        for x,y in zip(x_list,y_list):
            
            if y >= 0 and x>=0.001:
                
                if x_transform is not None :

                    x = x_transform(x)

                if y_transform is not None :

                    y = y_transform(y)

                plt.plot(x,y,'x',color="blue",axes=ax)
                x_tot.append(x)
                y_tot.append(y)


    x_np = np.array(x_tot)
    y_np = np.array(y_tot)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_np,y_np)

    print("Correlation coeficient : "+str(r_value))

    x1,y1 = 0, intercept
    x2,y2 = 0.5,(0.5*slope+intercept)


    plt.plot([x1,x2],[y1,y2],color="red")
    plt.savefig("plot_result.png")

    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f","--filename",type=str,help="Name of the file where experiments will be saved (file extension is not needed)",default="experiment_1")
    parser.add_argument("-xt","--xtransform",type=str,choices=["None","Square","Root","Exp","Log"],help="Perform transform over repartition distance before regression",default="None")
    parser.add_argument("-yt","--ytransform",type=str,choices=["None","Square","Root","Exp","Log"],help="Perform transform over mmd distance before regression",default="None")

    filepath = "./experiments/mmd_repartition_compare/"

    args = parser.parse_args()

    x_transform = None
    if args.xtransform == "Square":
        x_transform = square_transform
    elif args.xtransform == "Log":
        x_transform = log_transform
    elif args.xtransform == "Root":
        x_transform = root_transform
    elif args.xtransform == "Exp":
        y_transform = exp_transform

    y_transform = None
    if args.ytransform == "Square":
        y_transform = square_transform
    elif args.ytransform == "Log":
        y_transform = log_transform
    elif args.ytransform == "Root":
        y_transform = root_transform
    elif args.ytransform == "Exp":
        y_transform = exp_transform

    filepath = './experiments/mmd_repartition_compare/'
    filename = args.filename + "_result.json"

    plot_all_points(filepath+filename,x_transform,y_transform)

