#ifndef ZOOM_H
#define ZOOM_H
#define _CRT_SECURE_NO_WARNINGS

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<cuda.h>
#include<math_functions.h>
#include<math.h>

#include "device_launch_parameters.h"
#include<windows.h>
#include <stdint.h> 


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

#define RANDOM_NUMBER_MAX 1000
#define width 500
#define height 1000
#define padding 15

//Macro for checking cuda errors following a cuda launch or api call
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


struct timeval start, end;
double elapsed_time;

int gettimeofday(struct timeval * tp, struct timezone * tzp) {
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}

void set_clock(){
	gettimeofday(&start, NULL);
}

double get_elapsed_time(){
	gettimeofday(&end, NULL);
	double elapsed = (end.tv_sec - start.tv_sec) * 1000000.0;
	elapsed += end.tv_usec - start.tv_usec;
	return elapsed / 1000;
}

void validate(int **a, int **b, int rows, int cols) {
	int i, j;
	for (i = 0;i < rows;i++) {
		for (j = 0;j < cols;j++) {
			if (a[i][j] != b[i][j]) {
				printf("Different value detected at position: [%d][%d],"
						" expected %d but get %d\n", i, j, a[i], b[i]);
				return;
			}
		}
	}
	printf("Tests PASSED successfully! There is no differences \\;)/\n");
}

void initialize_data_random(int ***data, int rows, int cols) {
	int i, j;

	static time_t t;
	srand((unsigned)time(&t));

	*data = (int **)malloc(sizeof(int*) * rows);
	for (i = 0;i < rows;i++) {
		(*data)[i] = (int *)malloc(sizeof(int) * cols);
	}
	for (i = 0; i < rows; i++) {
		for (j = 0;j < cols;j++) {
			(*data)[i][j] = rand() % RANDOM_NUMBER_MAX;
		}
	}
}

void initialize_data_zero(int ***data, int rows, int cols) {
	int i, j;

	*data = (int **)malloc(sizeof(int*) * rows);
	for (i = 0;i < rows;i++) {
		(*data)[i] = (int *)malloc(sizeof(int) * cols);
	}
	for (i = 0; i < rows; i++) {
		for (j = 0;j < cols;j++) {
			(*data)[i][j] = 0;
		}
	}
}

void printImage(int ** image, int rows, int cols) {
	int i, j;
	printf("------------------\n");
	for (i = 0;i < rows;i++) {
		printf("%d: < ", i);
		for (j = 0;j < cols;j++) {
			printf("%d ", image[i][j]);
		}
		printf("> \n");
	}
}
#endif
