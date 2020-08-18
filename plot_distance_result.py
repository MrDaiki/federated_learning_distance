import json
import argparse

import numpy as np

import matplotlib.pyplot as plt


def exp_transform(data):

    return np.exp(data)

def log_transform(data):

    return np.log(data)

def square_transform(data):

    return (data)**2

def root_transform(data):

    return np.sqrt(data)


def prompt_experiments(filepath,x_transform=None,y_transform=None):

    with open(filepath,"r") as file:

        values = json.load(file)


    for value in values:

        x = np.array(value['distance_repartition'])
        y = np.array(value['distance_mmd'])

        if x_transform is not None:
            
            x = x_transform(x)

        if y_transform is not None:
            
            y = y_transform(y)


        ax = plt.axes(xlim=(0,1),ylim=(0,1),xlabel='class repartition distance',ylabel='maximum mean discrepancy')
        
        plt.plot(x/np.max(x),y/np.max(y),axes=ax)

        plt.show()



if __name__ == '__main__' : 

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f","--filename",type=str,help="Name of the file where experiments will be saved (file extension is not needed)",default="experiment_1")
    parser.add_argument("-xt","--xtransform",type=str,choices=["None","Square","Root","Exp","Log"],help="Perform transform over repartition distance before prompt",default="None")
    parser.add_argument("-yt","--ytransform",type=str,choices=["None","Square","Root","Exp","Log"],help="Perform transform over mmd distance before prompt",default="None")


    args = parser.parse_args()

    x_transform = None

    print(args.xtransform)

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

    prompt_experiments(filepath+filename,x_transform=x_transform,y_transform=y_transform)
