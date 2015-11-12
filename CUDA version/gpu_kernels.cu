#include "include_files.h"
#include <cublas_v2.h>


__device__ float CalculateDistanceGPU(const float * events,const float *clusters,int eventIndex,int clusterIndex,int ndim){
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

__global__ void ComputeDistanceMatrix_kernel(const float* clusters, const float* events_tr, float* matrix, int my_num_events,int nclass,int ndim) {

	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < my_num_events) {

		//matrix[i*nclass+blockIdx.y] = CalculateDistanceGPU(events,clusters,i,blockIdx.y,ndim);
		matrix[i*nclass+blockIdx.x] = CalculateDistanceGPU(events,clusters,i,blockIdx.x,ndim);

	}
}

__device__ float CalculateDistanceGPU2(const float * events,const float *clusters,int eventIndex,int clusterIndex,int ndim){
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

__global__ void ComputeDistanceMatrix_kernel2(const float* clusters, const float* events_tr, float* matrix, int my_num_events,int nclass,int ndim) {

	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < my_num_events) {

		//matrix[i*nclass+blockIdx.y] = CalculateDistanceGPU(events,clusters,i,blockIdx.y,ndim);
		matrix[i*nclass+blockIdx.x] = CalculateDistanceGPU(events,clusters,i,blockIdx.x,ndim);

	}
}

void compute_distance_kernel_wrapper(float *c1_gpu,float *data_gpu,float *distMat_device,int ndata,int nclass,int ndim,int num_threads){

	int n_TPB     = num_threads;
	int n_blocks;
	if((ndata%n_TPB)==0){
		n_blocks  = (ndata/n_TPB) ;}
	else{
		n_blocks  = (ndata/n_TPB)+1; 
	}
	// printf("n_blocks is %d \n",n_blocks);

	//dim3 threadsCompDist(n_TPB,1);
	//dim3 gridCompDist(n_blocks,nclass);
	dim3 threadsCompDist(1, n_TPB);
	dim3 gridCompDist(nclass, n_blocks);
	//ComputeDistanceMatrix_kernel<<<dim3(n_blocks,nclass),n_TPB>>>(c1_gpu,data_gpu,distMat_device,ndata,nclass,ndim);
	ComputeDistanceMatrix_kernel<<<gridCompDist,threadsCompDist>>>(c1_gpu,data_gpu,distMat_device,ndata,nclass,ndim);

}

template <int TILE_DIM_CW>
__global__ void cw_div_kernel(float *odata, float* idata, int width, int height)
{
	int xIndex = blockIdx.x * TILE_DIM_CW + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM_CW + threadIdx.y;

	int indexMat  = yIndex + width * xIndex;

	odata[indexMat] = odata[indexMat]/idata[xIndex];

}

void cw_div_kernel_wrapper(float *array,float*vector,int nrows,int ncols,int tile_dimension){

	dim3 threads2(tile_dimension,tile_dimension);
	dim3 gridC1(nrows/tile_dimension, ncols/tile_dimension);
	if(tile_dimension==16){ 
		cw_div_kernel<16><<<gridC1,threads2>>>(array,vector,ncols,nrows);}
	else
	{cw_div_kernel<32><<<gridC1,threads2>>>(array,vector,ncols,nrows);}

}

void matMul_wrapper(float* d_C, const float* d_A, const float* d_B, unsigned int hA, unsigned int wA,unsigned int hB, unsigned int wB,cublasStatus_t &ret){


	unsigned int m=wA;
	unsigned int k=hB;
	unsigned int n=wB;
	cublasHandle_t handle;
	cublasCreate(&handle);
	const float alpha = 1.0f;
	const float beta = 0.0f;
	ret = cublasSgemm(handle, 
		CUBLAS_OP_N, CUBLAS_OP_N, 
		n, m, k,
		&alpha, d_B, n, d_A, k, &beta, d_C, n);

}

template <int TD_T,int BLOCK_ROWS>
__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height)
{
	__shared__ float tile[TD_T][TD_T+1];

	int xIndex = blockIdx.x * TD_T + threadIdx.x;
	int yIndex = blockIdx.y * TD_T + threadIdx.y;  
	int index_in = xIndex + (yIndex)*width;

	xIndex = blockIdx.y * TD_T + threadIdx.x;
	yIndex = blockIdx.x * TD_T + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;


	for (int i=0; i<TD_T; i+=BLOCK_ROWS) {
		tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
	}

	__syncthreads();

	for (int i=0; i<TD_T; i+=BLOCK_ROWS) {
		odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
	}

}

void transpose_wrapper(float *odata,float*idata,int width,int height,int tile_dimension){

	dim3 grid(width/tile_dimension, height/tile_dimension), threads(tile_dimension,tile_dimension);
	if(tile_dimension==16){
		transposeNoBankConflicts<16,16><<<grid, threads>>>(odata, idata, width, height);}
	else{transposeNoBankConflicts<32,32><<<grid, threads>>>(odata, idata, width, height);}

}






// not used because of thrust is doing the exact same thing easier
/*

__global__ void to_power_kernel(float *odata, float *idata,float exp_arg, int width, int height)
{


	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;  
	int index_in = xIndex + (yIndex)*width;


	int index_out = xIndex + (yIndex)*width;

	odata[index_out] = pow(idata[index_in],exp_arg);


}
*/
template <int TILE_DIM_CPK>
__global__ void costum_to_power_kernel(float *odata, float *idata,float *uphi, int width, int height)
{


	int xIndex = blockIdx.x * TILE_DIM_CPK + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM_CPK + threadIdx.y;  
	int index_in = xIndex + (yIndex)*width;


	int index_out = xIndex + (yIndex)*width;
	float curr_element=idata[index_in];
	odata[index_out] = (curr_element*curr_element)*uphi[index_out];


}

void costum_to_power_kernel_wrapper(float *odata, float *idata1,float *uphi, int width, int height,int tile_dimension){

	dim3 threads2(tile_dimension,tile_dimension);
	dim3 gridU(width/tile_dimension, height/tile_dimension);
	if(tile_dimension==16){ 
		costum_to_power_kernel<16><<<gridU,threads2>>>(odata,idata1,uphi,width,height);}
	else
	{costum_to_power_kernel<32><<<gridU,threads2>>>(odata,idata1,uphi,width,height);}

}

// #define TILE_DIM 32
#include <math.h>

template <int TILE_DIM_CM>
__global__ void costum_minus_kernel(float *odata, float *idata1,float *idata2, int width, int height)
{


	int xIndex = blockIdx.x * TILE_DIM_CM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM_CM + threadIdx.y;  
	int index_in = xIndex + (yIndex)*width;


	int index_out = xIndex + (yIndex)*width;

	float tmp;
	tmp=(idata1[index_in]-idata2[index_in])*(idata1[index_in]-idata2[index_in]);
	odata[index_out]=sqrt(tmp); 

}

void costum_minus_kernel_wrapper(float *odata, float *idata1,float *idata2, int width, int height,int tile_dimension){

	dim3 threads2(tile_dimension,tile_dimension);
	dim3 gridU(width/tile_dimension, height/tile_dimension);
	if(tile_dimension==16){ 
		costum_minus_kernel<16><<<gridU,threads2>>>(odata,idata1,idata2,width,height);}
	else
	{costum_minus_kernel<32><<<gridU,threads2>>>(odata,idata1,idata2,width,height);}

}

template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
	T C; // number of columns

	__host__ __device__
		linear_index_to_row_index(T C) : C(C) {}

	__host__ __device__
		T operator()(T i)
	{
		return i / C;
	}
};
template <class T>
void sum_rows_thrust(thrust::device_ptr<T> matIn,thrust::device_ptr<T> matOut,thrust::device_ptr<int> row_indices,int numRows,int numCols){

	// allocate storage for row sums and indices

// 	thrust::device_vector<int> row_indices(numRows);

	// compute row sums by summing values with equal row indices
	thrust::reduce_by_key
		(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(numCols)),
		thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(numCols)) + (numRows*numCols),
		matIn,
		row_indices,
		matOut,
		thrust::equal_to<int>(),
		thrust::plus<T>());



}
void sum_rows_cublas(float * matIn,float * vector_out,float * vector_of_ones,int numRows,int numCols)
{
float alpha=1.0;
float beta=0.0;
int lda=numCols;
int m=numCols;
int n=numRows;

cublasStatus_t cublasStat;
cublasHandle_t cublasHandle;
cublasStat = cublasCreate(&cublasHandle);
cublasStat= cublasSgemv(cublasHandle,CUBLAS_OP_T ,m, n,&alpha,matIn,lda,vector_of_ones, 1,&beta,vector_out,1);
if (cublasStat!=CUBLAS_STATUS_SUCCESS)
{printf("execution failed \n");}

}
