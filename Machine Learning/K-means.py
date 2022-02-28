import math
import sys
import matplotlib.pyplot as plt
import time as Time
from NPCMLPakge import MLData

#求两个集合的闵可夫斯基距离
def MinkowskiDistance(list1,list2,p):
    sum = 0
    for i,j in zip(list1,list2):
        sum+=abs(i-j)**p
    return math.pow(sum,1/p)

def PrintDic(dic):
    for i in dic:
        print("C_%d="%(i),end='')
        print(dic[i])

def PrintU(list):
    for i in range(len(list)):
        print("u_%d=("%(i),end='')
        for j in range(len(list[i])):
            print("%0.3lf"%(list[i][j]),end=' ,' if j<len(list[i])-1 else ' )\n')

def DrawResult(c,pausetime):
    color = ['red','orange','yellow','green','blue','purple']
    for i,j in zip(c,color):
        for k in c[i]:
            data.DrawAData(k-1,j)
    plt.pause(pausetime)



class K_Means:
    def __init__(self,mldata:MLData,k):
        self.mldata = mldata
        self.k=k
        self.U=None

    def SetK(self,k):
        self.k=k

    def CreateU(self):
        self.U = [self.mldata.dataset[i] for i in range(0,self.mldata.length,self.mldata.length//self.k)]

    def Calculate(self,Time):
        dataset = self.mldata.dataset
        if self.U==None:
            self.CreateU()
        U=self.U

        for time in range(Time):
            #初始化簇集合
            c = {}
            for i in range(len(U)):
                c.update({i+1:[]})

            for i,index in zip(dataset,range(len(dataset))):
                #计算样本与各均值向量的距离
                dmin,ci = sys.maxsize,0
                for j in range(len(U)):
                    d = MinkowskiDistance(i,U[j],2)
                    if d<dmin:
                        dmin,ci = d,j+1            
                #将最小距离的放入对应的簇中 
                c[ci].append(index+1)
    
            newU = []
            for i in c:
                #计算均值向量
                u = []
                #获取数据集的维度
                for dim in range(len(dataset[0])):
                    sum = 0
                    for j in c[i]:
                        sum+=dataset[j-1][dim]
                    u.append(sum/len(c[i]))
                newU.append(u)
            U=newU
    
    def DrawByPlot(self,Time):
        if self.U==None:
            self.CreateU()
        U=self.U
        dataset = self.mldata.dataset
        for time in range(Time):
            #初始化簇集合
            c = {}
            for i in range(len(U)):
                c.update({i+1:[]})

            for i,index in zip(dataset,range(len(dataset))):
                #计算样本与各均值向量的距离
                dmin,ci = sys.maxsize,0
                for j in range(len(U)):
                    d = MinkowskiDistance(i,U[j],2)
                    if d<dmin:
                        dmin,ci = d,j+1            
                #将最小距离的放入对应的簇中 
                c[ci].append(index+1)
    
            PrintDic(c)
    
            newU = []
            for i in c:
                #计算均值向量
                u = []
                #获取数据集的维度
                for dim in range(len(dataset[0])):
                    sum = 0
                    for j in c[i]:
                        sum+=dataset[j-1][dim]
                    u.append(sum/len(c[i]))
                newU.append(u)
            U=newU
            PrintU(U)
            plt.title("%d time:"%(time+1))
            DrawResult(c,0.9)

if __name__ == "__main__":

    data = MLData()
    data.ReadData("./西瓜数据4.0.txt")
    data.TrainDigit()

    k = K_Means(data,3)
    k.CreateU()
    k.DrawByPlot(4)
    plt.show()
