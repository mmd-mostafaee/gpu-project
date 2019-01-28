#include "Header.h"

	/*
				width
		 ___________________
		|					|
		|					|
		|					|
height	|					|
		|					|
		|					|
		|					|
		 ___________________

	*/


float lagrange(int ** image, int x, int y, int si, int ei) {
	int ii, jj;
	float s = 1, t = 1, k = 0;
	for (ii = si;ii < ei;ii += 2)
	{
		s = 1;
		t = 1;
		for (jj = si;jj < ei;jj += 2)
		{
			if (jj != ii)
			{
				s = s * (y - jj);
				t = t * (ii - jj);
			}
		}
		k = k + ((s / t) * image[x][ii / 2]);
	}
	return k;
}

void naive(int ** image, int ** zoomed_image) {

	int i, j;
	int si, ei;

	// putting original cells in zoomed_image
	for (i = 0; i < height;i++) {
		for (j = 0; j < width;j++) {
			zoomed_image[i][j * 2] = image[i][j];
		}
	}

	for (i = 0; i < height;i++) {
		for (j = 1; j < width * 2;j += 2) {

			// Set start data
			si = j - padding;
			if (si % 2 == 0) si--;
			if (si < 0) si = 0;

			// Set end data
			ei = j + padding;
			if (ei % 2 == 0) ei++;
			if (ei > width * 2) ei = width * 2;

			// Run lagrange Algorithm
			zoomed_image[i][j] = (int)lagrange(image, i, j, si, ei);
		}
	}
}

__global__ void kernel(int ** image_d, int ** zoomed_image_d, int work_per_thread_in_width, int work_per_thread_in_height) {

	int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int i, j;
	int si_x, ei_y;
	int data_si, data_ei;

	si_x = tid_x * work_per_thread_in_height;
	ei_y = tid_y * work_per_thread_in_width;

	for (i = si_x;i < work_per_thread_in_height && i < height;i++) {
		for (j = ei_y;j < work_per_thread_in_width && j < width * 2;j += 2) {

			// Set start data
			data_si = j - padding;
			if (data_si % 2 == 0) data_si--;
			if (data_si < 0) data_si = 0;

			// Set end data
			data_ei = j + padding;
			if (data_ei % 2 == 0) data_ei++;
			if (data_ei > width * 2) data_ei = width * 2;

			// Lagrange
			int ii, jj;
			float s = 1, t = 1, k = 0;
			for (ii = data_si;ii < data_ei;ii += 2)
			{
				s = 1;
				t = 1;
				for (jj = data_si;jj < data_ei;jj += 2)
				{
					if (jj != ii)
					{
						s = s * (j - jj);
						t = t * (ii - jj);
					}
				}
				k = k + ((s / t) * image_d[i][ii / 2]);
			}

			zoomed_image_d[i][j] = k;
		}
	}

}

int main()
{
	int i;
	int ** image, ** zoomed_image;
	int ** image_d;
	int ** zoomed_image_h, ** zoomed_image_d;
	int block_size_x = 32, block_size_y = 32;
	int work_per_thread_width, work_per_thread_height;
	int grid_size = 2;

	// allocating and initializing image array with random data
	initialize_data_random(&image, height, width);

	// allocating memory for zoomed_image (result)
	initialize_data_zero(&zoomed_image, height, width * 2);

	set_clock();
	naive(image, zoomed_image);
	elapsed_time = get_elapsed_time();
	printf("Naive %.2fms\n", elapsed_time);

	// allocating memory for zoomed_image_d (result)
	initialize_data_zero(&zoomed_image_d, height, width * 2);


	work_per_thread_height = height / (grid_size * block_size_x);
	work_per_thread_width = (width * 2) / (grid_size * block_size_x);
	dim3 grid_dime(grid_size, 1, 1);
	dim3 block_dime(block_size_x, block_size_y, 1);

	cudaMalloc((void **)&image_d, sizeof(int) * height * width);
	cudaMalloc((void **)&zoomed_image_d, sizeof(int) * height * width * 2);

	set_clock();

	cudaMemcpy(image_d, image, sizeof(int) * height * width, cudaMemcpyHostToDevice);

	kernel<<<grid_dime, block_dime >>>(image_d, zoomed_image_d, work_per_thread_width, work_per_thread_height);

	cudaDeviceSynchronize();

	cudaMemcpy(zoomed_image_h, zoomed_image_d, sizeof(int) * height * width * 2, cudaMemcpyDeviceToHost);

	elapsed_time = get_elapsed_time();
	printf("Cuda %.2fms\n", elapsed_time);

	for (i = 0;i < height;i++) {
		free(image[i]);
		free(zoomed_image[i]);
	}
	free(image);
	free(zoomed_image);

	return 1;
}

// Helper function for using CUDA to add vectors in parallel.
/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}*/
