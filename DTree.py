import tkinter as tk
import math

class Tree:
    def __init__(self,rootNode,data):
        self.dataset = {}
        self.dataset.update({rootNode:{"id":0}})
        self.dataset.update({rootNode:{"data":data}})
        self.id = 1
    #添加叶子节点
    def Append(self,rootNode,leafNode,Gain,fromr):
        if rootNode not in self.dataset.keys():
            raise "错误！找不到根节点！"
        leafInfo = {}
        leafInfo.update({"belongTo":rootNode})
        leafInfo.update({"From":fromr})
        leafInfo.update({"Gain":Gain})
        leafInfo.update({"id":self.id+1})
        self.id+=1
        if leafNode in self.dataset.keys():
            while True:
                leafNode+='_'
                if leafNode not in self.dataset.keys():
                    self.dataset.update({leafNode:leafInfo})
                    break
        else:
            self.dataset.update({leafNode:leafInfo})

class DecisionTree:
    #读取数据
    def ReadData(self,filepath):
        f = open(filepath,"rb")
        fields = f.readline().decode('utf8').split(';')
        fields[-1]=fields[-1].replace("\r\n",'')
        dic = {}
        for j in range(len(fields)):
            dic.update({fields[j]:j})
        self.fields = dic
        self.ofields = fields
        dataset = []
        while True:
            dataline = f.readline().decode('utf8').split(';')
            if len(dataline)==1:
                break
            dataline[-1]=dataline[-1].replace("\r\n",'')
            dataset.append(dataline)
        self.dataset = dataset
    #求该节点的信息熵
    def Ent(self,dataset):
        #取得根节点的下标
        index = -1
        #历遍数据、统计数据类别
        dic = {}
        #print()
        #print(dataset)
        
        for i in dataset:
            if i[index] not in dic.keys():
                dic.update({i[index]:1})
            else:
                dic[i[index]]+=1
             
        #计算根节点信息熵
        ent_d = 0
        data_len = len(dataset)
        ii=0
        for i in dic.keys():
            t = dic[i]/data_len
            print("%0.3lf/%0.3lf log_2 (%0.3lf/%0.3lf)"%(dic[i],data_len,dic[i],data_len),end=" + " if ii<len(dic.keys())-1 else "")
            ent_d+= t*math.log2(t)
            ii+=1
        ent_d*=-1
        print("=%lf"%(ent_d))
        return ent_d
    #求结点相对于其他结点的信息熵
    def RealtiveEnt(self,node,dataset):
        #取得节点的下标
        index = self.fields[node]
        indexRoot = -1
        #历遍数据、统计数据类别和数据所在行
        dic = {}
        for i in range(len(dataset)):
            key = dataset[i][index]
            if key not in dic.keys():
                dic.update({key:[i]})
            else:
                dic[key].append(i)
        #计算节点的类别分支结点信息熵
        ent_ds={}
        ii=0
        for i in dic:
            otherdata = []
            tdic = {}
            for j in dic[i]:
                tdata = dataset[j][indexRoot]
                if  tdata not in  tdic.keys():
                    tdic.update({tdata:1})
                else:
                    tdic[tdata]+=1
            ent_d = 0
            dici_len = len(dic[i])
            for k in tdic:
                t = tdic[k]/dici_len
                ent_d+= t*math.log2(t)
                tstr= "%d/%d log_2 (%d/%d)"%(tdic[k],dici_len,tdic[k],dici_len)
                otherdata.append(tstr)
            ent_d*=-1

            if ent_d not in ent_ds.keys():
                ent_ds.update({ent_d:dici_len})
            else:
                while True:
                    ent_d-=0.000000001
                    if ent_d not in ent_ds.keys():
                        ent_ds.update({ent_d:dici_len})
                        break
            print("Ent(D^%d)="%(ii),end="")
            ii+=1
            ll = 0
            for l in otherdata:
                print(l,end=' + ' if ll<len(otherdata)-1 else '')
                ll+=1
            print("=%lf"%(ent_d))
            
        return ent_ds,otherdata
    #按该结点分割数据集
    def SplitDataset(self,nodeName,dataset):
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
    #根据结点和数据集创造一颗子树
    def CreateTree(self,rootNode,croot,dataset):
        print("\n根节点是:"+rootNode)
        print("当前数据集：")
        for i in dataset:
            print(i)
        t = self.CheckDataset(dataset)
        if type(t)!=bool:
            self.decisionTree.Append(rootNode,t,0,croot)
            return
        #数据总数
        data_len = len(dataset)
        #计算根节点的信息熵
        print("Ent("+rootNode+")=",end="")
        ent_D = self.Ent(dataset)
        #求各个属性的信息增益
        gain={}
        maxField = ""
        maxGain = -1
        for i in self.fields:
            if i not in self.decisionTree.dataset.keys():
                tdic,other=self.RealtiveEnt(i,dataset)
                sum=0
                print("Gain("+i+")=ent("+rootNode+")-",end="(")

                for j in tdic:
                    sum+=j*tdic[j]/data_len
                    print("((%0.3lf*%d)/%d)"%(j,tdic[j],data_len),end="+")
                    #print("%0.3lf*%0.3lf"%(j,tdic[j]),end="+"if j==len(tdic)-1 else "")
                gainData = ent_D - sum
                print(")=%0.4lf"%(gainData))      
                if maxGain <= gainData:
                    maxGain = gainData
                    maxField = i
                gain.update({i:gainData})
        print("最大的节点是："+maxField)
        #将子叶结点挂到根结点上
        if maxField not in self.ofields:
            print("eorr")
            return
        self.decisionTree.Append(rootNode,maxField,maxGain,croot)
        #根据计算所得的数据划分数据集
        spdaset = self.SplitDataset(maxField,dataset)
        for i in spdaset:
            if len(i) != 0:
                self.CreateTree(maxField,i,spdaset[i])
    #根据数据集自动生成决策树
    def AutoCreate(self):
        #print(self.ofields[-1])
        self.decisionTree = Tree(self.ofields[-1],{"ent":self.Ent(self.dataset)})
        self.CreateTree(self.ofields[-1],"",self.dataset)
        return self.decisionTree
    #检查数据集是否满足递归条件
    def CheckDataset(self,tdataset):
        if type(tdataset) == str:
            return tdataset
        t = tdataset[0][-1]
        for i in tdataset:
            if t != i[-1]:
                return True
        return t

tree = DecisionTree()
tree.ReadData("D:/Work_Space/西瓜数据2.0new.txt")
print('Read complete')
dataset = tree.AutoCreate()
print('calculate complete')
for i in dataset.dataset.keys():
    print(i,end=":")
    print(dataset.dataset[i])

