e1: Ejercicio1/mandelbrot_gpu.cu
	@clear
	@/usr/local/cuda-9.2/bin/nvcc $^ -o mandelbrot -lglut -lGL
	@-./mandelbrot
	@rm mandelbrot

e2: Ejercicio2/suma_matrices.cu
	@clear
	@/usr/local/cuda-9.2/bin/nvcc Ejercicio2/suma_matrices.cu -o suma
	@-./suma
	@rm suma