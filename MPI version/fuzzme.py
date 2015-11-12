# -*- coding: utf-8 -*-
from pylab import *

def euclidean_distance(vector1,vector2):
  return norm(vector1-vector2)

def distmat(points,centroids):
  distmat1=zeros([points.shape[0],centroids.shape[0]])
  for i in range(points.shape[0]):
    for j in range(centroids.shape[0]):
      distmat1[i,j]=euclidean_distance(points[i],centroids[j])
  return distmat1

def fuzme(nclass,data,U,phi,maxiter,toldif):
#unction [U, centroid, dist, W, obj] = fuzme(nclass,data,U,phi,maxiter,distype,toldif)


    
 ndata = data.shape[0];         # number of data 
 ndim = data.shape[1];         #number of dimension
 centroid=zeros([nclass,ndim]);
 dist=zeros([ndata,nclass]);



 obj=0;
 uphi = U**phi;   

 for i in range(maxiter):

     #calculate centroid
    c1=dot((uphi.conj().transpose() ),data );
    t1=(uphi.sum(axis=0));
    
    #t1.reshape(t1.shape[0],1)
    
    
    for i in range(c1.shape[1]):
      centroid[:,i]=c1[:,i]/t1
    
    

    #make t1 have ndim columns or rows i dont know yet
    #t1=t1(:,ones(ndim,1));
    
   
    #calculate distance of data to centroid
  
    dist=distmat(data, centroid);
    #print dist

    
    #save previous iterations
    U_old=U;
    obj_old=obj;

    #calculate new membership matrix
    tmp = dist**(-2/(phi-1));      
    t1=((tmp.conj().transpose()).sum(axis=0)).conj().transpose();
    #print tmp.shape
    #print t1.shape
    #break
    for i in range(tmp.shape[1]):
      tmp[:,i]=tmp[:,i]/t1
    #make t2 be t1 but have nclass rows
    #t2=t1[:,ones(nclass,1)];
    U = tmp;
    #print U.shape
    uphi = U**phi;   

    #calculate objective function
    o1=(dist**2)*uphi;
    obj = sum(o1); 
    
    #check for convergence
    dif=(obj_old-obj);
    difU=sqrt((U - U_old)*(U - U_old));
    Udif=sum(sum(difU));
    """
    if printing==1,
 	    fprintf('Iteration = %d, obj. fcn = %f.  diff = %f\n', i, obj, Udif);
    end
    """
    if (abs(Udif-dif)<toldif):
      print "finished after "+str(i)+ " iterations"
      break
    
    
