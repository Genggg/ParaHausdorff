/**
 * GPU Kernels for Hausdorff Matching
 * 
 * Creator: Yang Jiao
 * 
 * Created: 01:32 AM, May 3 2019
 * Last modified: 01:35 AM, May 3 2019
 *
 * This is a header file for GPU kernels used in Hausdorff Matching.
 * 
 * Current plan for the kernels are: 
 * 
 * convGPU: 2D convolution on GPU
 * dilateGPU: Dilate the binary image based on its distance map
 * gradGPU: 2D gradient computation, including the magnitude and orientation
 * searchGPU: TBD
 *
 * This file is a part of the Spring 2019 APMA2822B final project.
 * 
 */


#ifndef __GPUKERNELS_H
#define __GPUKERNELS_H

#include <iostream>

/**
 * 2D convolution on GPU (Global Memory Only)
 *
 * This function applies a 2D convolution on GPU, only using GPU's global memory.
 * It could be the slowest version
 *
 * @params[in] src the source data to be convoluted
 * @params[in] src_rows the number of rows of `src`
 * @params[in] src_cols the number of columns of `src`
 * @params[in] kernel the convolution kernel
 * @params[in] ker_rows the number of rows of `kernel`
 * @params[in] ker_rows the number of columns of `kernel`
 * @params[out] dst the result of the convolution
 */
__global__ void convGPUGlobal (double **src, int src_rows, int src_cols, double **kernel, int ker_rows, int ker_cols, double **dst);



#endif 