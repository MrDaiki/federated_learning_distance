import numpy as np
from math import sqrt


def distance_repartition_list(a,b):

    distance = 0

    for i in range(len(a)):

        distance += (a[i]-b[i])**2

    return(sqrt(distance))

def distance_repartition_dict(a,b):

    distance = 0

    for a_i,b_i in zip(a.items(),b.items()):

        a_l,a_v = a_i
        b_l,b_v = b_i

        distance += (a_v - b_v)**2

    return sqrt(distance)


def distance_repartition_numpy(a,b):
    return np.dot(a-b,a-b)

