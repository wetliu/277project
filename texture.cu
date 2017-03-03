#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <stdio.h>


using namespace std;
texture<float, 1, cudaReadModeElementType> texreference;

__global__ void kernel(float* doarray, int size)
{

    //calculate each thread global index
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    //int y = blockIdx.y*blockDim.y + threadIdx.y;

    //int offset = x + y*blockDim.x*gridDim.x;
    //fetch global memory through texture reference
    doarray[x] = tex1Dfetch(texreference, x);
    return;
}



int main(int argc, char** argv)
{
    int size = 64;
    float* harray;
    float* oarray;
    float* diarray;
    float* doarray;
    //allocate host and device memory
    harray = (float*)malloc(sizeof(float)*size);
    oarray = (float*)malloc(sizeof(float)*size);
    cudaMalloc((void**)&diarray, sizeof(float)*size);
    cudaMalloc((void**)&doarray, sizeof(float)*size);
    //initialize host array before usage
    for (int loop = 0; loop<size; loop++)
        harray[loop] = (float)loop;
    //copy array from host to device memory
    cudaMemcpy(diarray, harray, sizeof(float)*size, cudaMemcpyHostToDevice);
    //bind texture reference with linear memory
    cudaBindTexture(0, texreference, diarray, sizeof(float)*size);
    //execute device kernel
    kernel << <(int)ceil((float)size / 64), 64 >> >(doarray, size);
    //unbind texture reference to free resource
    cudaUnbindTexture(texreference);
    //copy result array from device to host memory
    cudaMemcpy(oarray, doarray, sizeof(float)*size, cudaMemcpyDeviceToHost);
    //free host and device memory
    for(int i = 0; i < size; i++){
      printf("%f\n", oarray[i]);
    }
    free(harray);
    free(oarray);
    cudaUnbindTexture(&texreference);
    cudaFree(diarray);
    cudaFree(doarray);
    return 0;
}
