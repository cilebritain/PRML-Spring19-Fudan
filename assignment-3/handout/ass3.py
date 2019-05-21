import os
import re
import math
import numpy as np
from fastNLP import Vocabulary
from fastNLP import DataSet

text=open('tangshi.txt',encoding="utf-8").read()
text,number=re.subn("，","",text)
text,number=re.subn("。","",text)
text,number=re.subn("\n","",text)

vo=Vocabulary()
vo.update(list(text))

#print(int(round(len(text)*0.8)))
train=text[0:int(round(len(text)*0.8))]
dev=text[int(round(len(text)*0.8))+1:len(text)-1]
train1={}
dev1={}

for a in train:
    train1[vo.to_index(a)]=a
for a in dev:
    dev1[vo.to_index(a)]=a

#print(train1)
#print(dev1)

ds=DataSet()
ds.__init__(train1)

#print(ds.__getitem__(1))


