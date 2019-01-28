#define _CRT_SECURE_NO_WARNINGS

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<math.h>

using namespace std;

#define RANDOM_NUMBER_MAX 100
#define width 1000
#define height 1000
#define padding 15


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

void naive(int ** image, int ** zoomed_image) {

	int i, j, ii, jj;
	int from, to;
	float s, t, k;

	// putting original cells in zoomed_image
	for (i = 0; i < height;i++) {
		for (j = 0; j < width;j++) {
			zoomed_image[i][j * 2] = image[i][j];
		}
	}

	for (i = 0; i < height;i++) {
		for (j = 1; j < width * 2;j += 2) {

			// Set start data
			from = j - padding;
			if (from % 2 == 0) from--;
			if (from < 0) from = 0;

			// Set end data
			to = j + padding;
			if (to % 2 == 0) to++;
			if (to > width * 2) to = width * 2;

			// Lagrange Algorithm
			s = 1; t = 1; k = 0;
			for (ii = from;ii < to;ii += 2)
			{
				s = 1;
				t = 1;
				for (jj = from;jj < to;jj += 2)
				{
					if (jj != ii)
					{
						s = s * (j - jj);
						t = t * (ii - jj);
					}
				}
				k = k + ((s / t) * image[i][ii / 2]);
			}
			zoomed_image[i][j] = k;
		}
	}
}

int main()
{


	int i, j, ii, jj;

	int ** image, ** zoomed_image;

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

	// initial image array with random data
	initialize_data_random(&image, height, width);

	// printImage(image, height, width);

	// allocating memory for zoomed_image (result)
	initialize_data_zero(&zoomed_image, height, width * 2);
	
	// sequential begin
	naive(image, zoomed_image);
	// sequential end

	// printImage(zoomed_image, height, width * 2);


	for (i = 0;i < height;i++) {
		free(image[i]);
		free(zoomed_image[i]);
	}
	free(image);
	free(zoomed_image);

	return 1;
}