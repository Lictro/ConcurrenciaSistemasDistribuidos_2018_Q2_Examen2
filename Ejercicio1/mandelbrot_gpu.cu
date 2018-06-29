#include "../Utilidades/book.h"
#include "../Utilidades/cpu_bitmap.h"

// Definimos la dimension a 1000
#define DIM 1000

// Struct que representa un numero complejo
struct cuComplex{
    float r, i;

    // Magnitud 2 ya que la magnitud real es la raiz cuadrada de esta
    __device__ cuComplex(float a, float b): r(a), i(b){}
    __device__ float magnitud2(void){
        return r * r + i * i;
    }

    // Funcion simplificada en clase
    __device__ cuComplex operator*(const cuComplex& a){
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    // La suma de dos numeros complejos es la suma de sus partes
    //  reales y sus partes imaginarias respectivamente
    __device__ cuComplex operator+(const cuComplex& a){
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int mandelbrot(int, int);
__global__ void kernel(unsigned char *);

int main(){
    // Buena costumbre para manipular imagenes
    CPUBitmap bitmap(DIM, DIM);

    // Apuntador a memoria GPU
    unsigned char* dev_bitmap;

    // Reservamos memoria en la RAM de la GPU 
    HANDLE_ERROR(cudaMalloc((void**) &dev_bitmap, bitmap.image_size()));

    // Creamos la malla para manipular de manera mas facil
    dim3 grid(DIM, DIM);

    // Mandamos la malla para tener un proceso por celda de la misma
    //  y un thread por proceso. Mandamos el apuntador de memoria ya 
    //  ya reservada anteriormente.
    kernel<<<grid,1>>>(dev_bitmap);

    // Copiamos el resultado de la GPU al bitmap que representa la imagen
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    
    // Liberamos el espacio de memoria en la GPU
    HANDLE_ERROR(cudaFree(dev_bitmap));
    
    // Mostramos el resultado y salimos
    bitmap.display_and_exit();

    return 0;
}

// Calcula si un pixel (o punto) pertenece al set de mandelbrot
__device__ int mandelbrot(int x, int y){
    // Escala de la imagen, mientras mas pequena la escala mas grande la imagen
    const float scale = 1;

    // Calculamos los puntos y los normalizamos. En este caso le restamos 250 a
    //  el punto x ya que queremos desplazar el resultado para visualizar una
    //  imagen centrada.
    float jx = scale * (float)(DIM/2 - x - 250)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    // Dado el fractal de Mandelbrot, dice que z comienza en 0, y el complejo se calcula
    //  basado en los puntos
    cuComplex a(0, 0);
    cuComplex c(jx, jy);

    // Revisamos si tiende a infinito siguiendo la sucesion de Z~n+1~ = Z~n~^2 + c
    //  aumentamos de 200 a 1000 ciclos por bloque para mejorar la calidad de imagen
    for(int i = 0; i < 1000; i++){
        //  Z~n+1~ = Z~n~^2 + c
        a = a * a + c;

        // Asumimos que diverge si es mayor a 100, mayor precicion que 200.
        //  Asumimos que converge si es menor o igual.
        if(a.magnitud2() > 1000)
            return 0;
    }
    return 1;
}

__global__ void kernel(unsigned char *ptr){
    // Identificamos en que bloque de la malla estamos
    int x = blockIdx.x;
    int y = blockIdx.y;

    // Traducimos el bloque a un offset de arreglo lineal
    int offset = x + y * gridDim.x;

    // Calculamos si este punto pertenece al set de mandelbrot
    int mandelbrotValue = mandelbrot(x, y);

    // Coloreamos la imagen de rojo si pertenece y de negro si no
    ptr[offset * 4 + 0] = 255 * !mandelbrotValue;
    ptr[offset * 4 + 1] = 255 * !mandelbrotValue;
    ptr[offset * 4 + 2] = 255 * !mandelbrotValue;
    ptr[offset * 4 + 3] = 255;
}