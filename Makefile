e1: Ejercicio1/mandelbrot_gpu.cu
	@clear
	@/usr/local/cuda-9.2/bin/nvcc $^ -o mandelbrot -lglut -lGL
	@-./mandelbrot
	@rm mandelbrot

e2: Ejercicio2/suma_matrices.cu
	@clear
	@/usr/local/cuda-9.2/bin/nvcc $^ -o suma
	@-./suma
	@rm suma

e3: Ejercicio3/ejercicio1.cu
	@clear
	@-cd Ejercicio3; /usr/local/cuda-9.2/bin/nvcc ejercicio1.cu -o encriptacion; ./encriptacion; rm encriptacion; cd ..