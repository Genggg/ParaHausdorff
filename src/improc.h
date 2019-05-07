#ifndef __IMPROC_H
#define __IMPROC_H

#include <iostream>
#include <cmath>
#include <omp.h>
#include <opencv2/opencv.hpp>
using namespace cv;

/**
 * Allocate 2D memory on CPU
 */
double** memAlloc2D(int rows, int cols);

/**
 * Allocate 2D unified memory
 */
double** cudaMallocManaged2D(int rows, int cols);

/**
 * Get a 1D Gaussian kernel
 */
 double* get1DGaussianKernel(int rows, double sigma);

/**
 * Get a 2D Gaussian kernel
 */
double** getGaussianKernel(int rows, int cols, double sigmax, double sigmay);

/**
 * Store an image in a 2D array
 */
double** img2Array(Mat image);

/**
 * Convert a 2D array to a Mat
 */
Mat array2Img(double **src, int rows, int cols);

/**
 * Convolution
 */
void conv(double **src, int src_rows, int src_cols, double **kernel, int ker_rows, int ker_cols, double **dst);

/**
 * Distance transform
 */
void distTrans(double **src, int src_rows, int src_cols, double **dst);

/**
 * Dilate the binary image based on its distance map
 */
void dilate(double **src, int src_rows, int src_cols, int d, double **dst);


/**
 * Thresholding 
 */

void nonMaxSupression(double **src, int src_rows, int src_cols, int t_rows, int t_cols, double p, double **dst);

/**
 * Non-maximum supression
 */

#endif // __IMPROC_H