#include "Utilidades/book.h"
#include "Utilidades/cpu_bitmap.h"

#define DIM 1000

struct cuComplex{
    float r, i;
    __device__ cuComplex(float a, float b): r(a), i(b){}
    __device__ float magnitud2(){
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a){
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a){
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int, int);
__global__ void kernel(unsigned char *);

int main(){
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**) &dev_bitmap, bitmap.image_size()));

    dim3 grid(DIM, DIM); //global?
    kernel<<<grid,1>>>(dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_bitmap));
    
    bitmap.display_and_exit();

    return 0;
}

__device__ int julia(int x, int y){
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.51, -0.601);
    cuComplex a(jx, jy);

    for(int i = 0; i < 200; ++i){
        a = a * a + c;
        if(a.magnitud2() > 1000)
            return 0;
    }
    return 1;
}

__global__ void kernel(unsigned char *ptr){
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x + y * gridDim.x;

    int juliaValue = julia(x, y);

    ptr[offset * 4 + 0] = 255 * juliaValue; //Change this
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}