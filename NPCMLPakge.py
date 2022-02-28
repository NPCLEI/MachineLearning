import math
import matplotlib.pyplot as plt



class MLData:
    def __init__(self):
        self.splitchar = ','

    #读入数据
    def ReadData(self,filepath):
        splitchar = self.splitchar
        f = open(filepath,"rb")
        fields = f.readline().decode('utf8').split(splitchar)
        fields[-1]=fields[-1].replace("\r\n",'')
        dic = {}
        for j in range(len(fields)):
            dic.update({fields[j]:j})
        self.fields = dic
        self.ofields = fields
        dataset = []
        while True:
            dataline = f.readline().decode('utf8').split(splitchar)
            if len(dataline)==1:
                break
            dataline[-1]=dataline[-1].replace("\r\n",'')
            dataset.append(dataline)
        self.length = len(dataset)
        self.dataset = dataset

    def SetSplitChar(self,char):
        self.splitchar=char

    def TrainDigit(self):
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset[i])):
                self.dataset[i][j]=float(self.dataset[i][j])
    #数字化一列
    def TrainDigitIndexOf(self,index):
        for i in range(len(self.dataset)):
            self.dataset[i][index]=float(self.dataset[i][index])

    #按该结点分割数据集
    def SplitDataset(self,nodeName):
        dataset = self.dataset
        #取得根节点的下标
        index = self.fields[nodeName]
        #历遍数据、统计数据类别
        dic = {}
        for i in dataset:
            if i[index] not in dic.keys():
                dic.update({i[index]:[i]})
            else:
                dic[i[index]].append(i)
        return dic

    #用于克隆一个与自己一样属性的数据集
    def Clone(self,dataset):
        tMLdata = MLData()
        tMLdata.dataset = dataset
        tMLdata.ofields = self.ofields
        tMLdata.fields = self.fields        
        tMLdata.length = len(dataset)
        return tMLdata

    def Print(self):
        print(self.ofields)
        for i,ii in zip(self.dataset,range(self.length)):
            print("%d:"%(ii+1),end='')
            print(i)

    def DrawAData(self,index,color):
        x=self.dataset[index][0]
        y=self.dataset[index][1]
        plt.plot([x-0.01,x+0.01],[y+0.01,y-0.01],color=color)
        plt.plot([x-0.01,x+0.01],[y-0.01,y+0.01],color=color)

    def DrawDataset(self,color):
        for i in range(self.length):
            self.DrawAData(i,color)


