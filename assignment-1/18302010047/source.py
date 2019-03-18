#encoding:utf-8
from pylab import *
import matplotlib as plt
import seaborn as sn
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
import numpy
import time
import sys
sys.path.append('../handout')
from __init__ import get_data


#how the number of the data influence the histogram

ion()

subplot(2,2,1)
a=get_data(100)
hist(a,normed=True,bins=50)  #hist means histgram
title("x=100")

subplot(2,2,2)
a=get_data(500)
hist(a,normed=True,bins=50)
title("x=500")

subplot(2,2,3)
a=get_data(1000)
hist(a,normed=True,bins=50)
title("x=1000")

subplot(2,2,4)
a=get_data(10000)
hist(a,normed=True,bins=50)
title("x=10000")

show()
close()

#how the number of the data influence the kernel density
#use package sklearn for easy plot

subplot(2,2,1)
a=np.array(get_data(100))[:,np.newaxis]
a_plot=np.linspace(min(a),max(a),1000)
kde=KernelDensity(kernel='gaussian',bandwidth=0.2).fit(a)
dens=kde.score_samples(a_plot)
plot(a_plot,np.exp(dens))
title("x=100")

subplot(2,2,2)
a=np.array(get_data(500))[:,np.newaxis]
a_plot=np.linspace(min(a),max(a),1000)
kde=KernelDensity(kernel='gaussian',bandwidth=0.2).fit(a)
dens=kde.score_samples(a_plot)
plot(a_plot,np.exp(dens))
title("x=500")

subplot(2,2,3)
a=np.array(get_data(1000))[:,np.newaxis]
a_plot=np.linspace(min(a),max(a),1000)
kde=KernelDensity(kernel='gaussian',bandwidth=0.2).fit(a)
dens=kde.score_samples(a_plot)
plot(a_plot,np.exp(dens))
title("x=1000")

subplot(2,2,4)
a=np.array(get_data(10000))[:,np.newaxis]
a_plot=np.linspace(min(a),max(a),1000)
kde=KernelDensity(kernel='gaussian',bandwidth=0.2).fit(a)
dens=kde.score_samples(a_plot)
plot(a_plot,np.exp(dens))
title("x=10000")

show()
close()

#how the number of the data influence the nearest neighbor
#use sklearn for getting k-nearest easily

subplot(2,2,1)
a=np.array(get_data(100))[:,np.newaxis]
a_plot=np.linspace(min(a),max(a),2000)
b_plot=np.zeros_like(a_plot)
nbrs=NearestNeighbors(n_neighbors=30).fit(a)
knear=nbrs.kneighbors(a_plot)
for i in range(2000):
    b_plot[i]=30/(100*(max(a[knear[1][i]])-min(a[knear[1][i]])))
plot(a_plot,b_plot)
title("n=100 k=30")

subplot(2,2,2)
a=np.array(get_data(500))[:,np.newaxis]
a_plot=np.linspace(min(a),max(a),2000)
b_plot=np.zeros_like(a_plot)
nbrs=NearestNeighbors(n_neighbors=30).fit(a)
knear=nbrs.kneighbors(a_plot)
for i in range(2000):
    b_plot[i]=30/(500*(max(a[knear[1][i]])-min(a[knear[1][i]])))
plot(a_plot,b_plot)
title("n=500 k=30")

subplot(2,2,3)
a=np.array(get_data(1000))[:,np.newaxis]
a_plot=np.linspace(min(a),max(a),2000)
b_plot=np.zeros_like(a_plot)
nbrs=NearestNeighbors(n_neighbors=30).fit(a)
knear=nbrs.kneighbors(a_plot)
for i in range(2000):
    b_plot[i]=30/(1000*(max(a[knear[1][i]])-min(a[knear[1][i]])))
plot(a_plot,b_plot)
title("n=1000 k=30")

subplot(2,2,4)
a=np.array(get_data(10000))[:,np.newaxis]
a_plot=np.linspace(min(a),max(a),2000)
b_plot=np.zeros_like(a_plot)
nbrs=NearestNeighbors(n_neighbors=30).fit(a)
knear=nbrs.kneighbors(a_plot)
for i in range(2000):
    b_plot[i]=30/(10000*(max(a[knear[1][i]])-min(a[knear[1][i]])))
plot(a_plot,b_plot)
title("n=10000 k=30")

show()
close()

#determine which is the best number of bins for histogram
#use seaborn to fit a curve to the histogram

a=get_data(200)

subplot(2,2,1)
sn.distplot(a,bins=2,kde=True)
title("bins=10")

subplot(2,2,2)
sn.distplot(a,bins=25,kde=True)
title("bins=50")

subplot(2,2,3)
sn.distplot(a,bins=100,kde=True)
title("bins=100")

subplot(2,2,4)
sn.distplot(a,bins=200,kde=True)
title("bins=200")

show()
close()

#find the best h for kernel density

a=get_data(100)
a_plot=linspace(min(a)-2,max(a)+2,1000)
b_plot=zeros_like(a_plot)
h=0.35

for i in range(1000):
    for p in a:
        b_plot[i] += np.exp ( -1* ( ( ( p-a_plot[i] ) ** 2) / ( 2 * (h ** 2) ) ) )/( ( 2 * np.pi * (h ** 2) ) ** 0.5)/100

plot(a_plot,b_plot)
xlabel("x")
ylabel("f(x)")
show()     
close()

#k-nearest neighbors method

subplot(2,2,1)
a=np.array(get_data(200))
a_plot=linspace(min(a)-2,max(a)+2,1000)
rrange=max(a)-min(a)+4
b_plot=zeros_like(a_plot)
K=10
p=0

for i in range(1000):
    b=abs(a-ones_like(a)*a_plot[i])
    c=b.argsort()
    d=c[0:K]
    b_plot[i]=K/(200*(max(max(a[d]),a_plot[i])-min(min(a[d]),a_plot[i])))
    p=p+b_plot[i]*rrange/1000

print(p)
plot(a_plot,b_plot)
title("K=10")

subplot(2,2,2)
a=np.array(get_data(200))
a_plot=linspace(min(a)-2,max(a)+2,1000)
rrange=max(a)-min(a)+4
b_plot=zeros_like(a_plot)
K=30
p=0

for i in range(1000):
    b=abs(a-ones_like(a)*a_plot[i])
    c=b.argsort()
    d=c[0:K]
    b_plot[i]=K/(200*(max(max(a[d]),a_plot[i])-min(min(a[d]),a_plot[i])))
    p=p+b_plot[i]*rrange/1000

print(p)
plot(a_plot,b_plot)
title("K=30")

subplot(2,2,3)
a=np.array(get_data(200))
a_plot=linspace(min(a)-2,max(a)+2,1000)
rrange=max(a)-min(a)+4
b_plot=zeros_like(a_plot)
K=50
p=0

for i in range(1000):
    b=abs(a-ones_like(a)*a_plot[i])
    c=b.argsort()
    d=c[0:K]
    b_plot[i]=K/(200*(max(max(a[d]),a_plot[i])-min(min(a[d]),a_plot[i])))
    p=p+b_plot[i]*rrange/1000

print(p)
plot(a_plot,b_plot)
title("K=50")

subplot(2,2,4)
a=np.array(get_data(200))
a_plot=linspace(min(a)-2,max(a)+2,1000)
rrange=max(a)-min(a)+4
b_plot=zeros_like(a_plot)
K=100
p=0

for i in range(1000):
    b=abs(a-ones_like(a)*a_plot[i])
    c=b.argsort()
    d=c[0:K]
    b_plot[i]=K/(200*(max(a[d])-min(a[d])))
    p=p+b_plot[i]*rrange/1000

print(p)
plot(a_plot,b_plot)
title("K=100")

show()
close()


        



