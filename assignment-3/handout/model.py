import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CharRNN(nn.Module): #继承torch.nn.module
    def __init__(self,input_size,hidden_size,output_size,n_layers=1):
        super(CharRNN,self).__init__()
        self.input_size=input_size #输入维数
        self.hidden_size=hidden_size #隐藏层维数
        self.output_size=output_size #输出维数
        self.n_layers=n_layers
        self.encoder=nn.Embedding(input_size,hidden_size) #embedding操作
        self.rnn=nn.LSTM(hidden_size,output_size,n_layers) #rnn：LSTM
        self.decoder=nn.Linear(output_size,input_size) #线性输出
        self.softmax=nn.LogSoftmax()

    def forward(self,input,hidden):
        batch_size=input.size()[0] #batch_size看一次输入几组数据 看输入的第一维即可
        encoded=self.encoder((input).view(batch_size,1,-1)) #embedding操作
        output,hidden=self.rnn(encoded,hidden) 
        #view是一个把矩阵reshape的函数 原来的数据是batch_size*time(LSTM延续的长度)*input_size 现在变成1*batch_size*(-1代表那一维自行计算)
        output=F.relu(self.decoder(output.view(batch_size,-1))) #将输出在转化为一个一维的 relu是一个整流函数
        output=self.softmax(output)
        return output,hidden

    def init_hidden(self,batch_size=1):
        return(Variable(torch.zeros(batch_size,1,self.hidden_size)),
               Variable(torch.zeros(batch_size,1,self.hidden_size)))


