# -*- coding: utf-8 -*-
from mpi4py import MPI
import random
import numpy
from pylab import *
from fuzzme import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size=comm.Get_size()

def euclidean_distance(vector1,vector2):
  return norm(vector1-vector2)

def distmat(points,centroids):
  distmat1=zeros([points.shape[0],centroids.shape[0]])
  for i in range(points.shape[0]):
    for j in range(centroids.shape[0]):
      distmat1[i,j]=euclidean_distance(points[i],centroids[j])
  return distmat1

def fuzme_mpi(nclass,data,U,phi,maxiter,toldif):
#unction [U, centroid, dist, W, obj] = fuzme(nclass,data,U,phi,maxiter,distype,toldif)


    
 ndata = data.shape[0];         # number of data 
 ndim = data.shape[1];         #number of dimension
 #centroid=zeros([nclass,ndim]);
 #dist=zeros([ndata,nclass]);
 dif=zeros(1,dtype=float32)



 obj=0;
 if rank==0:
  uphi = U**phi;   

 for i in range(maxiter):

     #calculate centroid
    if rank==0:
      U_val=copy(U)
      uphi_transposed=uphi.transpose()
     
      #print "Uphi transpose"
      #print uphi_transposed	
      uphi_transposed=uphi_transposed.reshape(size,int(U.shape[0]*U.shape[1]/size))
      
    else:
      uphi_transposed=zeros((nclass,ndata),dtype=float32)
      data=zeros(data.shape,dtype=float32)
    
    comm.Bcast([data,MPI.FLOAT])
    uphi_transposed=comm.scatter(uphi_transposed,root=0)
    uphi_transposed=uphi_transposed.reshape(nclass/size,ndata)
    #so far so good
    #print "broadcasted and scattered array"
    centroid=dot(uphi_transposed,data );
    t1=(uphi_transposed.sum(axis=1));
    
    #t1.reshape(t1.shape[0],1)
    
    
    for i in range(centroid.shape[1]):
      centroid[:,i]=centroid[:,i]/t1
    
    

    #make t1 have ndim columns or rows i dont know yet
    #t1=t1(:,ones(ndim,1));
    
   
    #calculate distance of data to centroid
    
    dist=distmat(data, centroid);
    #print dist
    if rank==0:
      #print "rank 0 ready to receive partial results"
      recv_list=[]
      for i in range(1,size):
	recv_buff=comm.recv(source=i)
	#print "rank 0 received partial results from "+str(i)
	recv_list.append(recv_buff)
      for x in recv_list:
	dist=hstack((dist,x))
      #print dist
    else:
      #print str(rank)+" sending dist matrix to 0"
      comm.send(dist,dest=0)
   
    #exit()
#works so far
    #save previous iterations
    if rank==0:
      U_old=U;
      obj_old=obj;

    #calculate new membership matrix
    distHeight=dist.shape[0]
    distWidth=dist.shape[1]
    
    dist=dist.reshape(size,distHeight*distWidth/size)
    dist=comm.scatter(dist,root=0)
    
    dist=dist.reshape(ndata/size,nclass)
    
    
    #print "broadcasted dist"
    #works so far
    tmp = dist**(-2/(phi-1));      
    t1=tmp.sum(axis=1);
    #print tmp.shape
    #print t1.shape
    #break
    for i in range(tmp.shape[1]):
      tmp[:,i]=tmp[:,i]/t1
    #make t2 be t1 but have nclass rows
    #t2=t1[:,ones(nclass,1)];
    U_partial = tmp;
    #print U.shape
    uphi_partial = U_partial**phi;   
    #print "computed u phi partial"
    #calculate objective function
    o1=(dist**2)*uphi_partial;
    #print "o1 for rank "+str(rank)+" is "+str(o1)
    obj = sum(o1);
    #print obj
    if rank==0:
      for i in range(1,size):
	curr_obj=comm.recv(source=i)
	obj+=curr_obj
      #print obj
      dif[0]=(obj_old-obj);
    else:
      comm.send(obj,dest=0)
    
    dif=comm.bcast(dif,root=0)
    U=comm.gather(U_partial,root=0)
    if rank==0:
      U_new=U[0]
      for i in range(1,len(U)):
	U_new=vstack((U_new,U[i]))
      U=U_new
      print U
      
#works so far
    
    
    #check for convergence
    if rank==0:
      difU=sqrt((U - U_old)*(U - U_old));
      Udif=sum(difU);
    else:
      Udif=0

    Udif=comm.bcast(Udif,root=0)
    #print"Udif is "+str(Udif)
    #print "obj is "+str(dif[0])
    comm.barrier()
    if (abs(Udif-dif[0])<toldif):
      print "finished after "+str(i)+ " iterations"
      break
    comm.barrier()
 if rank==0:
    fuzme(nclass,data,U_val,phi,maxiter,toldif)
    rel_error = norm(U-U_val) / norm(U_val)
    print "Relative Error: %f" % rel_error

    if rel_error < 1.0e-5:
      print "MPI test passed."
    else:
      print "MPI test failed."
    
