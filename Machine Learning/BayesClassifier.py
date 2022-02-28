class BayesClassifier:
    def __init__(self,data:MLData):
        self.data = data
    #计算先验概率
    def CalcuPrior(self):
        data = self.data
        field = data.ofields[-1]
        dic = data.SplitDataset(field)
        prior = {}
        
        for i in dic.keys():
            t=len(dic[i])/data.length
            prior.update({i:t})
            print("P(%s = %s) = %d/%d = %0.3lf"%(field,i,len(dic[i]),data.length,t))
        return prior

    #估计参数属性全部的的条件概率
    def CalcuAllCondProbi(self,fieldName):
        data = self.data
        ofields = data.ofields
        dic = data.SplitDataset(ofields[-1])
        for i in dic.keys():
            tMLdata = data.Clone(dic[i])
            tdic = tMLdata.SplitDataset(fieldName)
            for j in tdic.keys():
                tdata = len(tdic[j])/len(dic[i])
                print("P(%s = %s | %s = %s) = %d/%d = %0.3lf"%\
                    (fieldName,i,ofields[-1],j,len(tdic[j]),len(dic[i]),tdata))
    #估计参数属性某个的的条件概率
    def CalcuCondProbi(self,fieldName,which):
        data = self.data
        ofields = data.ofields
        dic = data.SplitDataset(ofields[-1])
        result={}
        for i in dic.keys():
            tMLdata = data.Clone(dic[i])
            tdic = tMLdata.SplitDataset(fieldName)
            tdata = len(tdic[which])/len(dic[i])
            print("P(%s = %s | %s = %s) = %d/%d = %0.3lf"%\
                (fieldName,i,ofields[-1],which,len(tdic[which]),len(dic[i]),tdata))
            result.update({i:tdata})
        return result   
    #估计参数线性属性某个的的条件概率
    def CalcuLinerCondProbi(self,fieldName,which):
        data = self.data
        ofields = data.ofields
        #按条件分割
        dic = data.SplitDataset(ofields[-1])
        #在条件下计算
        res = {}
        for i in dic.keys():
            tMLdata = data.Clone(dic[i])
            #先算平均值
            #返回which的下标
            index = data.fields[fieldName]
            sum = 0
            for j in tMLdata.dataset:
                sum+=float(j[index])
            avg = sum / tMLdata.length
            #再算方差
            sum = 0
            for j in tMLdata.dataset:
                sum+=(float(j[index])-avg)**2
            var = sum/tMLdata.length
            rootvar = math.pow(var,1/2)
            #算吧！
            t1 = math.pow(2*math.pi,1/2)*rootvar
            t2 = -1 * math.pow((float(which)-avg),2)
            t3 = 2*var
            t4 = t2/t3
            result = (1/t1)*math.pow(math.e,t4)
            print("P(%s = %lf | %s = %s) = %0.3lf"%\
                (fieldName,which,ofields[-1],i,result))
            res.update({i:result})
        return res
    #计算测试集
    def CalcuTest(self,testdata:MLData):
        prior = self.CalcuPrior()
        dataset = testdata.dataset[0]
        calresult = {}
        t={}
        for i in range(len(testdata.ofields)):
            if dataset[i][0] != '0':
                t=self.CalcuCondProbi(testData.ofields[i],dataset[i])
            else:
                t=self.CalcuLinerCondProbi(testData.ofields[i],float(dataset[i]))
            calresult.update({dataset[i]:t})
        #算是
        for i in prior.keys():
            sum = prior[i]
            for j in calresult.keys():
                sum*= calresult[j][i]
            print(sum)


