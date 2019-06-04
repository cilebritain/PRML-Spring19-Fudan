import re
import torch.optim as opt
import torch.nn as nn
from model import CharRNN
from util import *

#准备数据 诗句 
raw_data=open('tangshi.txt',encoding='utf-8').read()
data=re.split('，|。|\n',raw_data)
for a in data:
    if a=='':
        data.remove('')

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
    one_hot_var_target.setdefault(a,make_one_hot_vec_target(a,vo))
#setdefault方法是如果之前没有就设置为default的值 这里好像是把字符映射到了tensor上

epoch=10
batch=10
Train_size=len(data)

def test():
    loss=0
    counts=0
    v=int(Train_size/batch)

    for case in range(v*batch,min((v+1)*batch,Train_size)): #处理单个输入
        s=data[case]
        hidden=model.init_hidden()
        t,o=makeforOneCase(s,one_hot_var_target)
        output,hidden=model(t,hidden)
        loss+=criterion(output,o)
        counts+=1
    loss=loss/counts

for T in range(epoch):
    print(T)
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
        print(T,loss.data[0])
        optimizer.step()
    test()

def invert_dict(d):
    return dict((v,k) for k,v in d.iteritems())

ivo=invert_dict(vo)

def gener():
    st='花'.decode('utf-8')
    input=make_one_hot_vec_target(st,vo)
    hidden=model.init_hidden()
    output_name=st
    for i in range(100):
        output,hidden=model(input,hidden)
        topv,topi=output.data.topk(1)
        topi=topi[0][0]
        w=ivo[topi]
        if w=="<EOP>":
            break
        else:
            output_name+=w
        input=make_one_hot_vec_target(w,vo)
    return output_name

print(gener())





