/**
 * GPU Kernels for Hausdorff Matching
 * 
 * Creator: Yang Jiao
 * 
 * Created: 01:37 AM, May 3 2019
 * Last modified: 02:20 PM, May 5 2019
 *
 * This is the source code for GPU kernels used in Hausdorff Matching.
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

 #include "gpukernels.h"

 /**
 * 2D convolution on GPU (Global Memory Only)
 * It could be the slowest version
 *
 * Note that all pointers are **single**
 */
__global__ void convGPUGlobal (double *src, int src_rows, int src_cols, double *kernel, int ker_rows, int ker_cols, double *dst){
    int offset_rows = ker_rows / 2; // 3 -> 1, 4 -> 2, 5 -> 2
    int offset_cols = ker_cols / 2; // 3 -> 1, 4 -> 2, 5 -> 2
    // In most situations ker_rows = ker_cols 
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idy = threadIdx.y + blockIdx.y * blockDim.y;

    int global_id = global_idx * src_cols + global_idy;

    double sum = 0.0;
    // double pixel_i = 0.0;


    // dst[global_idy][global_idx] = 0.0;
    if(global_idx < src_cols && global_idy < src_rows){
        for(int kernel_indy = -offset_rows; kernel_indy <= offset_rows; kernel_indy++ ){
            for(int kernel_indx = -offset_cols; kernel_indx <= offset_cols; kernel_indx++){
                double pixel_intensity = 0.0;
                int conv_indx = global_idx+kernel_indx;
                int conv_indy = global_idy+kernel_indy;
                int conv_ind = conv_indx * src_cols + conv_indy;
                if(conv_indx >= 0 && conv_indx < src_cols && 
                conv_indy >= 0 && conv_indy < src_rows){
                    pixel_intensity = src[conv_ind];
                }
                else
                    pixel_intensity = 0;
                int kernel_ind = (offset_cols + kernel_indx) * ker_cols + (offset_rows + kernel_indy);
                // sum = kernel[0];
                sum += kernel[kernel_ind] * pixel_intensity;
            }
        }

        dst[global_id] = sum;
    }

    

}



