#include "serial_kernels.cpp"
#include <stdio.h>
// #include <mkl_blas.h>
#include "include_files.h"
#include "gpu_kernels.cu"
#include <cutil_inline.h>  
#define DEBUG 0
#if DEBUG ==1
#define MY_SAFE_CALL(x) CUDA_SAFE_CALL(x)
#else
#define MY_SAFE_CALL(x) x
#endif
#define PRINT_DIF 0
#define TILE_DIM_CURR 32
#include <cublas_v2.h>
#include "timer.h"
// #define USE_SGEMV_SUM

void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

template <typename T>
struct to_power_functor : public thrust::unary_function<T,T>
{
const T phi;
to_power_functor(T _phi) : phi(_phi) {}
    __host__ __device__
    T operator()(T x)
    {
        return pow(x,phi);
    }
};

float * cmeans_cuda(float *data,float *U,float phi,int max_iter,float toldif,int &total_iterations){

 
//  U_device_old=cu.mem_alloc( U.nbytes )


//  height_U=int32(U.shape[0])
int height_U=NDATA;
int width_U=NCLASS;
int heightData=NDATA;
int widthData=NDIM;
int widthA    = height_U;
int widthB      = widthData;
int widthC=widthB;
int heightC=widthA;
int height_c1=NCLASS;
int width_c1=NDIM;
int widthDist=NCLASS;
int heightDist=NDATA;
int height_t1=NCLASS;
int height_t2=NDATA;
int height_tmp=NDATA;
int width_tmp=NCLASS;
int num_rows_row_ind=max(height_tmp,height_t1);
int total_iter=0;
float * U_device;
int * rows_cuda;
MY_SAFE_CALL(cudaMalloc((void**)&U_device,NDATA*NCLASS*sizeof(float)));

#ifdef USE_SGEMV_SUM
float* input_vector;
MY_SAFE_CALL(cudaMalloc(&input_vector,num_rows_row_ind*sizeof(float)));
thrust::device_ptr<float> input_vector_ptr(input_vector);
thrust::fill(input_vector_ptr,input_vector_ptr+num_rows_row_ind,1.0);
#else
MY_SAFE_CALL(cudaMalloc((void**)&rows_cuda,num_rows_row_ind*sizeof(int)));
thrust::device_ptr<int> row_indices(rows_cuda);
#endif
// U_device=cu.mem_alloc( U.nbytes )
// cu.memcpy_htod( U_device, U )
MY_SAFE_CALL(cudaMemcpy(U_device,U,NDATA*NCLASS*sizeof(float),cudaMemcpyHostToDevice));
 
//  U_device_old=cu.mem_alloc( U.nbytes )
float * U_device_old;
MY_SAFE_CALL(cudaMalloc((void**)&U_device_old,NDATA*NCLASS*sizeof(float)));
//  diffU_device=gpuarray.empty(U.shape,dtype=float32)
float * diffU_device;
MY_SAFE_CALL(cudaMalloc((void**)&diffU_device,NDATA*NCLASS*sizeof(float)));

//  data_gpu=cu.mem_alloc( data.nbytes )
float * data_gpu;
MY_SAFE_CALL(cudaMalloc((void**)&data_gpu,NDATA*NDIM*sizeof(float)));
//  cu.memcpy_htod( data_gpu, data )
MY_SAFE_CALL(cudaMemcpy(data_gpu,data,NDATA*NDIM*sizeof(float),cudaMemcpyHostToDevice));

 float * U_phi_transposed_gpu;
//  U_phi_transposed=cu.mem_alloc( U.nbytes )
MY_SAFE_CALL(cudaMalloc((void**)&U_phi_transposed_gpu,height_U*width_U*sizeof(float)));

float * c1_gpu;
//  c1=cu.mem_alloc(U.shape[1]*data.shape[1]*a.nbytes)
MY_SAFE_CALL(cudaMalloc((void**)&c1_gpu,height_c1*width_c1*sizeof(float)));
//  t1=cu.mem_alloc(U.shape[1]*a.nbytes)
float * t1_gpu;
MY_SAFE_CALL(cudaMalloc((void**)&t1_gpu,width_U*sizeof(float))); 
//  t2=cu.mem_alloc(data.shape[0]*a.nbytes)
float * t2_gpu;
MY_SAFE_CALL(cudaMalloc((void**)&t2_gpu,heightData*sizeof(float)));
 

//  distMat_device=cu.mem_alloc(NDATA*nclass*a.nbytes)
float * distMat_device;
MY_SAFE_CALL(cudaMalloc((void**)&distMat_device,heightDist*widthDist*sizeof(float)));

int n_TPB = 2;
int n_blocks  = NDATA/n_TPB;
//  tmp=gpuarray.empty((NDATA,nclass),dtype=float32)
float * tmp_gpu;
MY_SAFE_CALL(cudaMalloc((void**)&tmp_gpu,NDATA*NCLASS*sizeof(float)));

float * U_phi_gpu;
//  U_phi=cu.mem_alloc( U.nbytes )
MY_SAFE_CALL(cudaMalloc((void**)&U_phi_gpu,NDATA*NCLASS*sizeof(float)));

float * o1_gpu; 
//  o1=gpuarray.empty((NDATA,nclass),dtype=float32)
MY_SAFE_CALL(cudaMalloc((void**)&o1_gpu,NDATA*NCLASS*sizeof(float)));
//  dim3 gridU(height_U/TILE_DIM, width_U/TILE_DIM); 
//  dim3 threadsT(TILE_DIM, TILE_DIM);
// printf("starting thrust stuff \n");
thrust::device_ptr<float> U_device_ptr(U_device);
thrust::device_ptr<float> U_phi_ptr(U_phi_gpu);
thrust::transform(U_device_ptr,U_device_ptr+NDATA*NCLASS, U_phi_ptr, to_power_functor<float>(phi));
// printf("finished thrust stuff \n"); 




float alpha=1.0;
float beta=0.0f;
// int heightA=width_U;

// int heightB=NDATA;

float obj_gpu=0;
float obj_old_gpu=0;
float Udif_gpu=0.0;
for (total_iter=0;total_iter<max_iter;total_iter++){


transpose_wrapper(U_phi_transposed_gpu,U_phi_gpu,width_U,height_U,TILE_DIM_CURR);
#if DEBUG ==1
	cudaThreadSynchronize();

		// check for error
		cudaError_t error2 = cudaGetLastError();
		if(error2 != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error2));
printf("error at transpose_kernel \n");
			exit(-1);
		}
#endif

cublasStatus_t ret;
matMul_wrapper(c1_gpu,U_phi_transposed_gpu,data_gpu,width_U, height_U,heightData, widthData,ret);
#if DEBUG ==1
if(ret!=CUBLAS_STATUS_SUCCESS){
printf("cublasSgemm error: %d", ret);}
else{printf("multiplied \n");}
#endif
#ifdef USE_SGEMV_SUM
sum_rows_cublas(U_phi_transposed_gpu,t1_gpu,input_vector,height_t1,height_U);
#else
thrust::device_ptr<float> matout_ptr(t1_gpu);
thrust::device_ptr<float> matin_ptr(U_phi_transposed_gpu);
sum_rows_thrust<float>(matin_ptr,matout_ptr,row_indices,height_t1,height_U);
#endif


// printf("finished row sum kernel1 \n");

cw_div_kernel_wrapper(c1_gpu,t1_gpu,height_c1,width_c1,TILE_DIM_CURR);
#if DEBUG ==1
	cudaThreadSynchronize();

		// check for error
		error2 = cudaGetLastError();
		if(error2 != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error2));
printf("error at cw_div_kernel \n");
			exit(-1);
		}
#endif
//   
compute_distance_kernel_wrapper(c1_gpu,data_gpu,distMat_device,NDATA,NCLASS,NDIM,256);
#if DEBUG ==1
	cudaThreadSynchronize();

		// check for error
		error2 = cudaGetLastError();
		if(error2 != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error2));
printf("ComputeDistanceMatrix_kernel \n");
			exit(-1);
		}
#endif  


    U_device_old=U_device;
    
    
    // to_power_kernel<<<dim3(heightDist/TILE_DIM,widthDist/TILE_DIM),dim3(TILE_DIM,TILE_DIM)>>>(tmp_gpu,distMat_device,-2.0/(phi-1),widthDist,heightDist);


thrust::device_ptr<float> tmp_ptr(tmp_gpu);
thrust::device_ptr<float> distMat_ptr(distMat_device);
thrust::transform(distMat_ptr,distMat_ptr+NDATA*NCLASS, tmp_ptr, to_power_functor<float>(-2.0/(phi-1)));

  
//     row_sum_kernel2<<<height_t2,128>>>(t2_gpu, tmp_gpu);
#ifdef USE_SGEMV_SUM
sum_rows_cublas(tmp_gpu,t2_gpu,input_vector,height_tmp,width_tmp);
#else
thrust::device_ptr<float> matIn(tmp_gpu);
thrust::device_ptr<float> matOut(t2_gpu);
sum_rows_thrust<float>(matIn,matOut,row_indices,height_tmp,width_tmp);
#endif


 
//     cw_div_kernel<<<dim3(height_tmp/TILE_DIM,width_tmp/TILE_DIM),dim3(TILE_DIM,TILE_DIM)>>>(tmp_gpu,t2_gpu,width_tmp,height_tmp);
cw_div_kernel_wrapper(tmp_gpu,t2_gpu,height_tmp,width_tmp,TILE_DIM_CURR);
#if DEBUG == 1
	cudaThreadSynchronize();

		// check for error
		error2 = cudaGetLastError();
		if(error2 != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error2));
printf("error at cw_div_kernel \n");
			exit(-1);
		}
#endif     



    U_device=tmp_gpu;
   
    

thrust::device_ptr<float> U_device_ptr2(U_device);   
    
    
    

thrust::transform(U_device_ptr2,U_device_ptr2+NDATA*NCLASS, U_phi_ptr, to_power_functor<float>(phi));
    


costum_to_power_kernel_wrapper(o1_gpu, distMat_device,U_phi_gpu, widthDist,heightDist,TILE_DIM_CURR);
#if DEBUG ==1
	cudaThreadSynchronize();

		// check for error
		error2 = cudaGetLastError();
		if(error2 != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error2));
			exit(-1);
		}
#endif
    
    //#works so far
    
  

    obj_old_gpu=obj_gpu;
#if DEBUG ==1
    std::cout<<"obj old_gpu is now "<<obj_old_gpu<<std::endl;
#endif



    thrust::device_ptr<float> o1_ptr(o1_gpu);
  float init=0.0;
   obj_gpu = thrust::reduce(o1_ptr, o1_ptr+NDATA*NCLASS,init,thrust::plus<float>()); 


    
    

    float dif_gpu=(obj_old_gpu-obj_gpu);
#if DEBUG ==1
printf("dif_gpu is %f \n",dif_gpu);
#endif    
    


// costum_minus_kernel<<<gridU,dim3(TILE_DIM_CURR,TILE_DIM_CURR)>>>(diffU_device,U_device,U_device_old,width_U,height_U);
costum_minus_kernel_wrapper(diffU_device, U_device,U_device_old,width_U,height_U,TILE_DIM_CURR);
#if DEBUG ==1
	cudaThreadSynchronize();

		// check for error
		error2 = cudaGetLastError();
		if(error2 != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error2));
			exit(-1);
		}
#endif
  
    

   

 thrust::device_ptr<float> diffU_device_ptr(diffU_device);
float Udif_gpu=thrust::reduce(diffU_device_ptr,diffU_device_ptr+NDATA*NCLASS,init,thrust::plus<float>());   
#if DEBUG ==1
   printf("Udif_gpu is %f \n",Udif_gpu);
   printf("dif_gpu is %f \n",dif_gpu);

   std::cout<<"finished iteration "<<total_iter<<std::endl;
#endif 
    if (abs(Udif_gpu-dif_gpu)<toldif){
    
      break;}



}


MY_SAFE_CALL(cudaMemcpy(U,U_device,NDATA*NCLASS*sizeof(float),cudaMemcpyDeviceToHost));
total_iterations=++total_iter;
return U;

}
#include <iostream>
#include <fstream>
using namespace std;

int main(){
/*
  int R=5;
int C=8;

float* inputMatrix=(float*)malloc(R*C*sizeof(float));
float* sums=(float*)malloc(R*sizeof(float));  
  for(int i = 0; i < C*R; i++)inputMatrix[i]=i;
  sum_rowsCPU(sums,inputMatrix,R,C);
  for(int i = 0; i < R; i++)
  {
    std::cout << "[ ";
    for(int j = 0; j < C; j++)
      std::cout << inputMatrix[i * C + j] << " ";
    std::cout << "] = " << sums[i] << "\n";
  }

  return 0;
*/

 ifstream data_file;
 data_file.open ("data.txt");

 ofstream U_out_file;
 U_out_file.open ("U_out2.txt");
float * data;
data=(float*)malloc(NDATA*NDIM*sizeof(float));
float * U;
U=(float*)malloc(NDATA*NCLASS*sizeof(float));
randomInit(data,NDATA*NDIM);
randomInit(U,NDATA*NCLASS);
// for (int i=0;i<NDATA*NDIM;i++)printf("%f  ",data[i]);
printf("running for %d DATA %d DIM %d CLASS \n",NDATA,NDIM,NCLASS);
int total_iter;
timer total_timer;
U=cmeans_cuda(data,U,2.0,100,0.1,total_iter);
float total_time_elapsed=total_timer.seconds_elapsed();
printf("total time elapsed was %f \n",total_time_elapsed);
printf("total iterations were %d \n",total_iter);

  for(int i = 0; i < NDATA; i++)
  {
    U_out_file << "[ ";
    for(int j = 0; j < NCLASS; j++)
      U_out_file << U[i * NCLASS + j] << " ";
    U_out_file<< "]"<<"\n";
  }

//   myfile << "Writing this to a file.\n";
  data_file.close();
  U_out_file.close();
  return 0;









}
