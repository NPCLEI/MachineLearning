import matplotlib.pyplot as plt
import numpy as np
import math

#产生[-1，1]的随机数
def ThreRandom():
    return random.randint(-10,10)/10.0

def Sigmoid(value):
    return 1/(1+math.pow(math.e,-value))

def DeSigmoid(value):
    return Sigmoid(value)*(1-Sigmoid(value))

def MutpMatx(l1,l2):
    sum=0
    for i,j in zip(l1,l2):
        sum+=i*j
    return sum

class Neural:
    def __init__(self):
        self.threhold = ThreRandom()
     
