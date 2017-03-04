
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
using namespace std;

#define BDIMX 16
#define BDIMY 16
const int TILE_DIM = 4;
#define IPAD 2
#define TILE_DIM1 32
#define BLOCK_ROWS 8

inline double cpuSecond(){
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp,&tzp);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Simple transformation kernel
__global__ void transformKernel(float* output,
                                const int width, const int height)
{
     // Calculate normalized texture coordinates
     unsigned int x = blockIdx.x * blockDim.x*TILE_DIM + threadIdx.x;
     unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

     //if(x == 0) printf("y = %d, %f \n")

     unsigned int ti = x + y*width;

     if(x + (TILE_DIM-1)*blockDim.x < width && y < height){
			 #pragma unroll
			 for(int i = 0; i < TILE_DIM; i++)
			 output[ti + i*blockDim.x] = tex2D(texRef, y, x+i*blockDim.x);
     }


     /*
     int offset = x*height + y;

     if(x < width && y < height){
       output[offset] = tex2D(texRef, x, y);
     }
     */


     // Read from texture and write to global memory
     //output[y * width + x] = tex2D(texRef, x, y);
}

__global__ void transposeNaive(float *output, const float *idata,const int width, const int height)
{
	unsigned int x = blockIdx.x * blockDim.x*TILE_DIM + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int width = gridDim.x * TILE_DIM;

  //for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    //odata[x*width + (y+j)] = idata[(y+j)*width + x];
	unsigned int ti = y*width+x;
	unsigned int to = x*height+y;

	if(x+(TILE_DIM-1)*blockDim.x < height && y < width){
		#pragma unroll
		for(int i = 0; i < TILE_DIM; i++)
		output[ti + i*blockDim.x] = idata[to + i*blockDim.x*height];
	}
}

__global__ void transposeSmemUnrollPad(float *out, float *in, const int nx,
                                       const int ny)
{
    // static 1D shared memory with padding
    __shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];

    // coordinate in original matrix
    unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // linear global memory index for original matrix
    unsigned int ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    unsigned int ix2 = blockIdx.y * blockDim.y + icol;
    unsigned int iy2 = 2 * blockIdx.x * blockDim.x + irow;

    // linear global memory index for transposed matrix
    unsigned int to = iy2 * ny + ix2;

    if (ix + blockDim.x < nx && iy < ny)
    {
        // load two rows from global memory to shared memory
        unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) +
            threadIdx.x;
        tile[row_idx]         = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];

        // thread synchronization
        __syncthreads();

        // store two rows to global memory from two columns of shared memory
        unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}

__global__ void transposeNoBankConflicts(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM1][TILE_DIM1+1];

  int x = blockIdx.x * TILE_DIM1 + threadIdx.x;
  int y = blockIdx.y * TILE_DIM1 + threadIdx.y;
  int width = gridDim.x * TILE_DIM1;

  for (int j = 0; j < TILE_DIM1; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM1 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM1 + threadIdx.y;

  for (int j = 0; j < TILE_DIM1; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// Host code
int main(int argc, char** argv)
{
     int height = 1<<14;
     int width = 1<<14;
     int blockx = 32;
     int blocky = 32;

     if(argc > 1) height = atoi(argv[1]);
     if(argc > 2) width = atoi(argv[2]);
     if(argc > 3) blockx = atoi(argv[3]);
     if(argc > 4) blocky = atoi(argv[4]);
		 cout << height << ' ' << width << ' ' << blockx << ' ' << blocky << endl;

     // Allocate CUDA array in device memory
     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
                                                               cudaChannelFormatKindFloat);
     cudaArray* cuArray;
     cudaMallocArray(&cuArray, &channelDesc, width, height);

     // Copy to device memory some data located at address h_data

     int size = height*width;
     float *h_data = (float*)malloc(sizeof(float)*size);
     float *compare_data = (float*)malloc(sizeof(float)*size);
     for (int loop = 0; loop<size; loop++){
         h_data[loop] = (float)loop;
     }
     for(int n = 0; n<width*height; n++) {
        int i = n/height;
        int j = n%width;
        compare_data[n] = h_data[width*j + i];
    }

     for(int i = 0; i < 10; i++){
       printf("%f\n", compare_data[i]);
     }
     // in host memory

     cudaMemcpyToArray(cuArray, 0, 0, h_data, size*sizeof(float),
     cudaMemcpyHostToDevice);

     // Set texture reference parameters
     //texRef.addressMode[0] = cudaAddressModeWrap;
     //texRef.addressMode[1] = cudaAddressModeWrap;
     //texRef.filterMode = cudaFilterModeLinear;
     //texRef.normalized = true;

     // Bind the array to the texture reference
     cudaBindTextureToArray(texRef, cuArray, channelDesc);

     // Allocate result of transformation in device memory
     float* output;
     cudaMalloc(&output, width * height * sizeof(float));
		 float* output1;
     cudaMalloc(&output1, width * height * sizeof(float));
		 float* output2;
			cudaMalloc(&output2, width * height * sizeof(float));
		 float* input;
     cudaMalloc(&input, width * height * sizeof(float));
		 cudaMemcpy(input, h_data, sizeof(float)*size, cudaMemcpyHostToDevice);

     // Invoke kernel
     dim3 dimBlock(blockx, blocky);
		 dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x/TILE_DIM, (height + dimBlock.y - 1) / dimBlock.y, 1);

		 double t = cpuSecond();
		 transformKernel<<<dimGrid, dimBlock>>>(output, width, height);
		 cudaDeviceSynchronize();
		 cout << "Time spent on the gpu is " << cpuSecond()-t << endl; //print time

		 memset(h_data,0,size);
     cudaMemcpy(h_data, output, sizeof(float)*size, cudaMemcpyDeviceToHost);
     for(int i = 0; i < 10; i++){
       if(h_data[i] != compare_data[i]) {
         printf("error: wrong number in %d, cpu: %f, gpu: %f. \n",i,compare_data[i],h_data[i]);
         break;
       }
     }


		 ////////////////////////////////// naive implementation
		 double t1 = cpuSecond();
		 transposeNaive<<<dimGrid, dimBlock>>>(output1, input,width, height);//transposeNaive, copySharedMem
		 cudaDeviceSynchronize();
		 cout << "Time spent on the gpu naive1 is " << cpuSecond()-t1 << endl; //print time
		 memset(h_data,0,size);
     cudaMemcpy(h_data, output1, sizeof(float)*size, cudaMemcpyDeviceToHost);
     for(int i = 0; i < size; i++){
       if(h_data[i] != compare_data[i]) {
         printf("error: wrong number in %d, cpu: %f, gpu: %f. \n",i,compare_data[i],h_data[i]);
         break;
       }
     }

		 //////////////////////////////// book implementation
		 //dim3 dimBlock1(16, 16);
		 //dim3 dimGrid1((width + dimBlock1.x*2 - 1) / (dimBlock1.x*2), (height + dimBlock1.y - 1) / dimBlock1.y, 1);
		 //double t1 = cpuSecond();
		 //transposeSmemUnrollPad<<<dimGrid1, dimBlock1>>>(output1, input,width, height);//transposeNaive, copySharedMem
		 //cudaDeviceSynchronize();
		 //cout << "Time spent on the gpu naive1 is " << cpuSecond()-t1 << endl; //print time

		 ////////////////////////////////post implementation, faster than book's
		 int nx = width, ny = height;
		 dim3 dimGrid2(nx/TILE_DIM1, ny/TILE_DIM1, 1);
	   dim3 dimBlock2(TILE_DIM1, BLOCK_ROWS, 1);
		 double t2 = cpuSecond();
		 transposeNoBankConflicts<<<dimGrid2, dimBlock2>>>(output2, input);//transposeNaive, copySharedMem
		 cudaDeviceSynchronize();
		 cout << "Time spent on the gpu padding is " << cpuSecond()-t2 << endl; //print time
		 memset(h_data,0,size);
     cudaMemcpy(h_data, output2, sizeof(float)*size, cudaMemcpyDeviceToHost);
     for(int i = 0; i < size; i++){
       if(h_data[i] != compare_data[i]) {
         printf("error: wrong number in %d, cpu: %f, gpu: %f. \n",i,compare_data[i],h_data[i]);
         break;
       }
     }

     // Free device memory
     free(h_data);
     cudaFreeArray(cuArray);
		 cudaFree(input);
     cudaFree(output);
		 cudaFree(output1);
		 cudaFree(output2);
     cudaUnbindTexture(&texRef);
     return 0;
}
