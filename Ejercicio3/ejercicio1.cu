#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//El numero de caractares en el texto encriptado
#define N 1024

void checkCUDAError(const char*);
void read_encrypted_file(int*);


/* Ejercicio 1.1 */
int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void affine_decrypt(int *d_input, int *d_output)
{
	/* Ejercicio 1.2 */
}

__global__ void affine_decrypt_multiblock(int *d_input, int *d_output)
{
	/* Ejercicio 1.8 */
}


int main(int argc, char *argv[])
{
	int *h_input, *h_output;
	int *d_input, *d_output;
	unsigned int size;
	int i;

	size = N * sizeof(int);

	/* Alojar memoria en el host */
	h_input = (int *)malloc(size);
	h_output = (int *)malloc(size);

	/* Ejercicio 1.3: Alojar Memoria en el dispositivo */
	//cudaMalloc(???);
	//cudaMalloc(???);
	checkCUDAError("Alojamiento de Memoria");

	/* Lectura del texto encriptado */
	read_encrypted_file(h_input);

	/* Ejercicio 1.4: copiar las entradas del host al dispositivo */
	//cudaMemcpy(???);
	checkCUDAError("Transferencia de entradas al dispositivo");

	/* Ejercicio 1.5: Configurar el grid y correr el kernel */
	//dim3 blocksPerGrid(???);
	//dim3 threadsPerBlock(???);
	//affine_decrypt(???);

	/* Espera a que todos los hilos esten completos*/
	cudaThreadSynchronize();
	checkCUDAError("Ejecucion del Kernel");

	/* Ejercicio 1.6: Copiar la salida de la GPU al host */
	//cudaMemcpy(???);
	checkCUDAError("Transferencia de resultados al host");

	/* Imprimir resultados */
	for (i = 0; i < N; i++) {
		printf("%c", (char)h_output[i]);
	}
	printf("\n");

	/* Ejercicio 1.7: Liberacion de Memoria */
	//cudaFree(???);
	//cudaFree(???);
	checkCUDAError("Liberacion de Memoria");

	/* Limpiear buffer del host*/
	free(h_input);
	free(h_output);

	return 0;
}


void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void read_encrypted_file(int* input)
{
	FILE *f = NULL;
	f = fopen("encrypted01.bin", "rb"); 
	if (f == NULL){
		fprintf(stderr, "Error:  No se pudo encontrar encrypted01.bin file \n");
		exit(1);
	}
	//Lectura de data encriptada
	fread(input, sizeof(unsigned int), N, f);
	fclose(f);
}