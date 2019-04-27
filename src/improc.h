#ifndef __IMPROC_H
#define __IMPROC_H

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace cv;

/**
 * Allocate 2D memory on CPU
 */
double** memAlloc2D(int rows, int cols);

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
 * Distance Transform
 */
void distTrans(double **src, int src_rows, int src_cols, double **dst);

#endif // __IMPROC_H