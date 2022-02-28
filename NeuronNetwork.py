import matplotlib.pyplot as plt
import numpy as np
import math
import random
import cv2
import sys      
import os   
from tqdm import tqdm



def ClearConsole():
    f_handler=open('out.log', 'w')      # 打开out.log文件
    oldstdout = sys.stdout              # 保存默认的Python标准输出
    sys.stdout=f_handler                # 将Python标准输出指向out.log
    os.system('cls')                    # 清空Python控制台       
    sys.stdout = oldstdout              # 恢复Python默认的标准输出

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
        weightMatx = np.full((1,Inputs+1),TherRandom())
        print("准备中.......")
        for i in tqdm(range(Neurons-1)):
            t = np.full((1,Inputs+1),TherRandom())
            weightMatx = np.row_stack((weightMatx,t[-1]))

        self.weightMatx = weightMatx
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
        s = self.Output()
        ds = self.alp*s*(1-s)
        c = ds*self.eorr
        if len(c)>len(self.weightMatx):
            c=c[:-1]
        wempoint = 0
        tw = []
        j=0
        for i in c:
           tw.append((i[0]*(self.input.transpose()))[0])
           j+=1
        self.weightMatx=self.weightMatx+np.array(tw)



class NeuronNet:
    def __init__(self,innerlayers,outputlayer):
        self.innerlayers = innerlayers
        self.outputlayer = outputlayer
        self.datapoint=0
    def Train(self,data,Y):
        innerlayers = self.innerlayers
        outputlayer = self.outputlayer
        for ii,yi in zip(data,Y):
            #print("真在训练数据：%d"%(self.datapoint))
            #self.datapoint+=1
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

    def NeuralFunc(self,x1):
        innerlayers = self.innerlayers
        outputlayer = self.outputlayer
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

    def Active(self,input): 
        innerlayers = self.innerlayers
        outputlayer = self.outputlayer
        i=np.array([input])
        i=i.transpose()
        for j in range(len(innerlayers)):
            innerlayers[j].SetInput(i)
            i=innerlayers[j].Output()
        outputlayer.SetInput(i)
        return outputlayer.Output()

    def Save(self,filepath,dirname):
        pass
    


