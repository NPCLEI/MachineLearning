import matplotlib.pyplot as plt
import numpy as np
import math
import random
import cv2

def Draw(inputdata,func):
    for i in inputdata:
        plt.plot([i[0]-0.01,i[0]+0.01],[i[1]+0.01,i[1]-0.01])
        plt.plot([i[0]-0.01,i[0]+0.01],[i[1]-0.01,i[1]+0.01])

    last = [-15,func(-15)]
    for j in range(-150,230):
        i = [j/10,func(j/10)]
        plt.plot([i[0],last[0]],[i[1],last[1]])
        last = i        

    plt.show()

def ReadData(filepath):
    f = open(filepath,"rb")
    x1 = f.readline().decode('utf8').split(',')
    x2 = f.readline().decode('utf8').split(',')
    dataset = []
    for i,j in zip(x1,x2):
        dataset.append([float(i),float(j),-1])
    return dataset


#激活函数
def Sigmoid(value):
    try:
        return 1/(1+math.pow(math.e,-1*value))
    except:
        if -1*value > 700:
            return 0
        else:
            return 1
#激活函数
def SigmoidArray(array:np.array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i][j]=Sigmoid(array[i][j])
    return array

#矩阵函数
def FunArray(array:np.array,func):
    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i][j]=func(array[i][j])
    return array

#激活函数的导数
def SigmoidDerivative(value):
    return Sigmoid(value)*(1-Sigmoid(value))

#产生[-1，1]的随机数
def TherRandom():
    return random.randint(-5,5)/10.0

class NeuronLayer:
    def __init__(self,Inputs:int,Neurons:int,alp:float):
        weightMatx=[]
        for i in range(Neurons):
            weightMatx.append([TherRandom() for i in range(Inputs+1)])
        self.weightMatx = np.array(weightMatx)
        self.alp = alp
        self.category = "normal"#0是正常神经元,1是输出神经元

    def ChangeOutput(self):
        self.category = 1

    def PrintWeightMatx(self):
        for i in self.weightMatx:
            print(i)

    def SetInput(self,input:list):
        self.input=input

    def SetEorr(self,eorr:list):
        self.eorr = eorr

    def CalcuEorrMatix(self):
        t = self.weightMatx.transpose()
        #return t*self.eorr
        if len(t[0])<len(self.eorr):
            self.eorr=self.eorr[:-1]
        return np.dot(t,self.eorr)

    def Output(self):
        result = SigmoidArray(np.dot(self.weightMatx,self.input))
        if self.category == "normal":
            return np.append(result,[[-1]],axis=0)
        else:
            return result

    def Adjust(self):
        def CutMatix(matix):
            t=[]
            for i in range(len(matix)-1):
                t.append(matix[i])
            return np.array(t)
        s = self.Output()
        ds = self.alp*s*(1-s)
        c = ds*self.eorr
        if len(c)>len(self.weightMatx):
            c=CutMatix(c)
        #print("\nc是：")
        #print(c)
        #print("\n权值变化前：")
        #print(self.weightMatx)
        #print()
        tw = []
        j=0
        for i in c:
           #print(i[0]*(self.input.transpose()))
           tw.append((i[0]*(self.input.transpose()))[0])
           j+=1
        #print(np.array(tw))
        self.weightMatx=self.weightMatx+np.array(tw)
        #print("\n权值变化后：")
        #print(self.weightMatx)



innerlayers = [NeuronLayer(72,72,0.2) for i in range(2)]
innerlayers[0]=NeuronLayer(36,72,0.4)
outputlayer = NeuronLayer(72,3,0.3)
outputlayer.ChangeOutput()

def Train(data,Y):
    for ii,yi in zip(data,Y):
        i=np.array([ii])
        i=i.transpose()
        for j in range(len(innerlayers)):
            innerlayers[j].SetInput(i)
            i=innerlayers[j].Output()
        outputlayer.SetInput(i)
        outmatx = outputlayer.Output()

        eorr =np.array(yi)-outmatx 

        outputlayer.SetEorr(eorr)
        outputlayer.Adjust()

        innerlayers[-1].SetEorr(outputlayer.CalcuEorrMatix())
        for j in range(len(innerlayers)-2,-1,-1):
            innerlayers[j].SetEorr(innerlayers[j+1].CalcuEorrMatix())
            innerlayers[j].Adjust()

def NeuralFunc(x1):
    try:
        for j in range(-6000,6000):
            i=np.array([[x1,j/1000,-1]])
            i=i.transpose()
            innerlayer.SetInput(i)
            innerlayer1.SetInput(innerlayer.Output())
            outputlayer.SetInput(innerlayer1.Output())
            outmatx = outputlayer.Output()
            if outmatx[0][0] < 0.52 and outmatx[0][0] > 0.48:
                return j/100
    except:
        return 1

def Active(input):        
    i=np.array([input])
    i=i.transpose()
    for j in range(len(innerlayers)):
        innerlayers[j].SetInput(i)
        i=innerlayers[j].Output()
    outputlayer.SetInput(i)
    return outputlayer.Output()
    
#data = ReadData("D:/Work_Space/x.txt")
#Y = [1 if i <1000 else 0 for i in range(2000)]

#data = [[0,0,-1],[0,1,-1],[1,0,-1],[1,1,-1]]
#Y = [[[0]],[[1]],[[1]],[[0]]]


def Main():
    d1=[0,0,0,0,0,0
    ,0,0,0,1,0,0
    ,0,0,1,1,0,0
    ,0,0,0,1,0,0
    ,0,0,0,1,0,0
    ,0,0,0,0,0,0,-1]

    d2=[0,0,0,0,0,0
    ,0,1,1,1,0,0
    ,1,0,0,1,0,0
    ,0,0,1,0,0,0
    ,0,1,0,0,0,0
    ,1,1,1,1,1,0,-1]

    d0=[0,0,0,0,0,0
    ,0,0,1,1,0,0
    ,0,1,0,0,1,0
    ,0,1,0,0,1,0
    ,0,1,0,0,1,0
    ,0,0,1,1,0,0,-1]

    d3=[0,0,1,1,0,0
        ,0,1,0,0,1,0
        ,0,0,0,1,1,0
        ,0,0,0,0,1,0
        ,0,1,0,0,1,0
        ,0,0,1,1,0,0,-1]

    d4=[0,0,0,1,0,0
        ,0,0,1,1,0,0
        ,0,1,0,1,0,0
        ,1,1,1,1,1,1
        ,0,0,0,1,0,0
        ,0,0,0,1,0,0,-1]

    d5=[0,1,0,0,0,0
        ,0,1,1,1,1,0    
        ,0,1,0,0,0,0    
        ,0,0,1,1,0,0    
        ,0,0,0,0,1,0    
        ,0,1,1,1,0,0,-1]

    data = [d0,d1,d2,d3,d4,d5]
    Y=[
       [[0],[0],[0]],
       [[0],[0],[1]],
       [[0],[1],[0]],
       [[0],[1],[1]],
       [[1],[0],[0]],
       [[1],[0],[1]]]

    for i in range(100):
        if i%10==0:
            print("训练第%d次"%(i))
        Train(data,Y)

    def Che(list):
        sum = 0
        for i in range(3):
            if list[2-i][0]>0.8:
                sum+=2**i
        return sum

    for i in data:
        for j in range(36):
            print(' ' if i[j]==0 else '*',end='')
            if (j+1)%6==0:
                print()
        res = Active(i)
        print("这是%d"%(Che(res)))
        
Main()
