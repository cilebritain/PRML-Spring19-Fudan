import os
import re
import math
import numpy as np
from fastNLP import Vocabulary
from fastNLP import DataSet

text=open('tangshi.txt',encoding="utf-8").read()
data=re.split('，|。|\n| ',text)
print(data)

def ppp():
    return('fuck')

print(ppp())