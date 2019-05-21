import torch.optim as opt
import torch.nn as nn
from model import CharRNN
from util import *

#准备数据 诗句 
data=['abc','def']

#构建字典
vo={}
for sentance in data:
    for word in sentance:
        if word not in vo:
            vo[word]=len(vo)
vo['<EOP>']=len(vo)
vo['<START>']=len(vo)

for i in range(len(data)):
    data[i]=toList(data[i])
    data[i].append("<EOP>") #给每句诗加个换行结尾

model=CharRNN(len(vo),256,256)
optimizer=opt.RMSprop(model.parameters(),lr=0.01,weight_decay=0.0001) #RMSprop算法 lr学习率 wd权重衰减
criterion=nn.NLLLoss() #Negative Log Likelihood负对数似然损失函数

one_hot_var_target={}
for a in vo:
    one_hot_var_target.setdefault(w,make_one_hot_var_target(w,vo))
#setdefault方法是如果之前没有就设置为default的值 这里好像是把字符映射到了tensor上

epoch=200
batch=100
Train_size=len(data)

def test():
    loss=0
    counts=0

    for case in range(v*batch,min((v+1)*batch,Train_size)): #处理单个输入
        s=data[case]
        hidden=model.init_hidden()
        t,o=makeforOneCase(s,one_hot_var_target)
        output,hidden=model(t,hidden)
        loss+=criterion(output,o)
        counts+=1
    loss=loss/counts

for T in range(epoch):
    for i in range(int(Train_size/batch)):
        model.zero_grad()
        loss=0
        counts=0
        for case in range(i*batch,min((i+1)*batch,Train_size)):
            s=data[case]
            hidden=model.init_hidden()
            t,o=makeforOneCase(s,one_hot_var_target)
            output,hidden=model(t,hidden)
            loss+=criterion(output,o)
            counts+=1
        loss=loss/counts
        loss.backward()#梯度下降
        optimizer.step()
    test()




