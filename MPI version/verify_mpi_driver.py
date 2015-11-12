# -*- coding: utf-8 -*-
from pylab import *

from fuzzme_mpi import *
import time

maxiter=10000
ndim=12432
#ndim=512
nclass=1024
ndata=1504
#ndata=1024
U=rand(ndata,nclass).astype(float32)
#print U
data=zeros([ndata,ndim]).astype(float32)
#print data
toldif=0.1
phi=2
f=open('dataset','r')
for line in f:
  row=line.split(' ')
  docId=int32(row[0])
  wordId=int32(row[1])
  countWord=float32(row[2])
  data[(docId-1)][(wordId-1)]=countWord
start=time.clock()
fuzme_mpi(nclass,data,U,phi,maxiter,toldif)
elapsed=(time.clock()-start)
print "total time elapsed for "+str(nclass)+" in mpi is "+str(elapsed)
