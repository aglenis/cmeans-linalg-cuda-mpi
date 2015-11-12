#include <iostream>
#include <functional>
#include <numeric>
#include <cstdlib>
#include <cmath>
float CalculateDistanceCPU(const float * events,const float *clusters,int eventIndex,int clusterIndex,int ndim){
   float sum=0.0;
   const float * event_ptr=&events[eventIndex*ndim];
   const float * cluster_ptr=&clusters[clusterIndex*ndim];
   float tmp=0.0;
   for (int i=0;i<ndim;i++){
    tmp=event_ptr[i]-cluster_ptr[i];
    sum+=tmp*tmp;
   }
  sum=sqrt(sum);
  return sum;
    }


void ComputeDistanceMatrix_kernelCPU(const float* clusters, const float* events, float* matrix, int my_num_events,int nclass,int ndim) {
    

 for(int i=0;i<my_num_events;i++) {
    for(int j=0;j<nclass;j++){
        matrix[i*nclass+j] = CalculateDistanceCPU(events,clusters,i,j,ndim);}
    
   
}
}

void row_sum_kernelCPU(float * outputMatrix,float * inputMatrix,int numRows,int numColumns){

for (int i=0;i<numRows;i++){
outputMatrix[i]=std::accumulate(&inputMatrix[i*numColumns],&inputMatrix[i*numColumns]+numColumns,0);
}
}
void row_sum_kernelCPU2(float * outputMatrix,float * inputMatrix,int numRows,int numColumns){

for (int i=0;i<numRows;i++){
  outputMatrix[i]=0;
for (int j=0;j<numColumns;j++){
//   std::cout<<"adding "<<inputMatrix[i*numColumns+j]<<" to the current sum "<<std::endl;
  outputMatrix[i]+=inputMatrix[i*numColumns+j];
}
// std::cout<<"finally storing "<<curr_sum<<" to position "<<i<<std::endl;
// std::cout<<outputMatrix[i]<<std::endl;
}
}

void to_power_kernelCPU(float* outputMatrix,float * inputMatrix,float exp_arg,int numElem ){
for (int i=0;i<numElem;i++)
  outputMatrix[i] = pow(inputMatrix[i],exp_arg);

}

void costum_to_power_kernelCPU(float* outputMatrix,float * inputMatrix,float* uphi,int numElem ){
for (int i=0;i<numElem;i++){
  float curr_element=inputMatrix[i];
  outputMatrix[i] = (curr_element*curr_element)*uphi[i];
}
}

void costum_minus_kernelCPU(float* outputMatrix,float * inputMatrix1,float* inputMatrix2,int numElem ){
for (int i=0;i<numElem;i++){
float tmp=(inputMatrix1[i]-inputMatrix2[i])*(inputMatrix1[i]-inputMatrix2[i]);
  outputMatrix[i]=sqrt(tmp); 
}
}

void cw_div_kernelCPU(float *odata, float* idata, int width, int height){

for (int y=0;y<height;y++){
for (int x=0;x<width;x++){


odata[y*width+x]=odata[y*width+x]/idata[y];

}
}
}

void transpose_kernelCPU(float* odata, float* idata,
			 const  int inRows,const  int inCols)
{
  for(  int y = 0; y < inRows; ++y) {
    for(  int x = 0; x < inCols; ++x) {
      odata[(x * inRows) + y] = idata[(y * inCols) + x];
    }
  }
}

void
MatrixMulKernel(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}