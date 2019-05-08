/**
 * GPU Kernels for Hausdorff Matching
 * 
 * Creator: Yang Jiao
 * 
 * Created: 01:37 AM, May 3 2019
 * Last modified: 05:14 PM, May 5 2019
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

 #define MAX_2D_THREADS_PER_BLOCK 32

 /**
 * 2D convolution on GPU (Global Memory Only)
 * It could be the slowest version (220x220: 0.07 ms; 3456x4606: 9.76 ms)
 *
 */
__global__ void convGPUGlobal (double **src, int src_rows, int src_cols, double **kernel, int ker_rows, int ker_cols, double **dst){
    const int offset_rows = ker_rows / 2; // 3 -> 1, 4 -> 2, 5 -> 2
    const int offset_cols = ker_cols / 2; // 3 -> 1, 4 -> 2, 5 -> 2
    // In most situations ker_rows = ker_cols 
    const int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int global_idy = threadIdx.y + blockIdx.y * blockDim.y;

    double sum = 0.0;

    // dst[global_idy][global_idx] = 0.0;
    if(global_idx < src_cols && global_idy < src_rows){
        for(int kernel_indy = -offset_rows; kernel_indy <= offset_rows; kernel_indy++ ){
            for(int kernel_indx = -offset_cols; kernel_indx <= offset_cols; kernel_indx++){
                double pixel_intensity = 0.0;
                int conv_indx = global_idx+kernel_indx;
                int conv_indy = global_idy+kernel_indy;
                if(conv_indx >= 0 && conv_indx < src_cols && 
                conv_indy >= 0 && conv_indy < src_rows){
                    pixel_intensity = src[conv_indy][conv_indx];
                }
                sum += kernel[offset_cols + kernel_indy][offset_rows + kernel_indx] * pixel_intensity;
            }
        }

        dst[global_idy][global_idx] = sum;
    }

    

}


 /**
 * 2D convolution on GPU (Shared Memory)
 * Speed up the convolution using shared memory (which actually doesn't)
 *
 */
 __global__ void convGPUShared (double **src, int src_rows, int src_cols, double **kernel, int ker_rows, int ker_cols, double **dst){
    const int offset_rows = ker_rows / 2; // 3 -> 1, 4 -> 2, 5 -> 2, also the size of apron
    const int offset_cols = ker_cols / 2; // 3 -> 1, 4 -> 2, 5 -> 2
    // In most situations ker_rows = ker_cols 
    
    const int tile_rows = MAX_2D_THREADS_PER_BLOCK + 2*offset_rows; // Larger than blockDim.y
    const int tile_cols = MAX_2D_THREADS_PER_BLOCK + 2*offset_cols; // blockDim.x 

    extern __shared__ double s[]; // The whole chunk of shared memory

    double* shared_src = s;
    double* shared_kernel = (double*)&shared_src[tile_rows * tile_cols];

    const int num_sub_blocks = (tile_rows + blockDim.y) / blockDim.y; // Number of sub-blocks

    // Filter size must smaller than 32 x 32
    if(threadIdx.y < ker_rows && threadIdx.x < ker_cols){
        shared_kernel[threadIdx.y * ker_rows + threadIdx.x] = kernel[threadIdx.y][threadIdx.x];
    }

    // Find global Idx (in the image) of the head and tail of each block
    const int block_start_pix_col = blockIdx.x * MAX_2D_THREADS_PER_BLOCK;
    const int block_end_pix_col = block_start_pix_col + MAX_2D_THREADS_PER_BLOCK;
    const int block_src_end_pix_col = CMIN(block_end_pix_col, src_cols);

    const int block_start_pix_row = blockIdx.y * MAX_2D_THREADS_PER_BLOCK;
    const int block_end_pix_row = block_start_pix_row + MAX_2D_THREADS_PER_BLOCK;
    const int block_src_end_pix_row = CMIN(block_end_pix_row, src_rows);

    const int tile_start_col = block_start_pix_col - offset_cols;
    const int tile_start_row = block_start_pix_row - offset_rows;

    // Load the "padded" image into shared tile
    int local_id_col = threadIdx.x; // Local ID
    int pixel_id_col = tile_start_col + local_id_col; // Global Position
    
    for(unsigned sub_block_num = 0; sub_block_num < num_sub_blocks; sub_block_num++){
        int local_id_row = threadIdx.y + sub_block_num * blockDim.y;
        int pixel_id_row = tile_start_row + local_id_row;
        if(local_id_col >= 0 && local_id_col < tile_cols
        && local_id_row >= 0 && local_id_row < tile_rows){
            if(pixel_id_row >= 0 && pixel_id_row < src_rows 
            && pixel_id_col >= 0 && pixel_id_col < src_cols){
                shared_src[local_id_row * blockDim.x + local_id_col] = src[pixel_id_row][pixel_id_col];
            }
            else{
                shared_src[local_id_row * blockDim.x + local_id_col] = 0.0;
            }
        }
    }

    __syncthreads();
    // Perform convolution
    local_id_col = threadIdx.x; // Local ID
    pixel_id_col = tile_start_col + local_id_col; // Global position (for output)
    for(unsigned sub_block_num = 0; sub_block_num < num_sub_blocks; sub_block_num++){
        int local_id_row = threadIdx.y + sub_block_num * blockDim.y; // Local ID
        int pixel_id_row = tile_start_row + local_id_row; // Global position (for output)
        double sum = 0.0;
        // Make sure the output position is in a block (also only theses threads are enabled)
        if(pixel_id_row >= block_start_pix_row && pixel_id_row < block_src_end_pix_row 
        && pixel_id_col >= block_start_pix_col && pixel_id_col < block_src_end_pix_col
        && local_id_col >= 0 && local_id_col < tile_cols && local_id_row >= 0 && local_id_row < tile_rows){
            for(int kernel_indy = -offset_rows; kernel_indy <= offset_rows; kernel_indy++){
                for(int kernel_indx = -offset_cols; kernel_indx <= offset_cols; kernel_indx++){
                    // The "conv indices" will always be in the tile
                    int conv_indx = local_id_col + kernel_indx;
                    int conv_indy = local_id_row + kernel_indy;
                   sum += shared_kernel[(offset_rows + kernel_indy) * ker_rows + (offset_cols + kernel_indx)]
                        * shared_src[conv_indy * blockDim.x + conv_indx];
                }
            }
            // Save the result into global memory
            dst[pixel_id_row][pixel_id_col] = sum;
        }
    }
 }


 /**
 * 1D column convolution on GPU (Shared Memory)
 *
 *
 */
 __global__ void convGPUCol (double **src, int src_rows, int src_cols, double *kernel_col, int ker_cols, double **dst){
    const int offset_cols = ker_cols / 2; // 3 -> 1, 4 -> 2, 5 -> 2 Only cols

    const int tile_cols = MAX_2D_THREADS_PER_BLOCK + 2*offset_cols; 
    
    extern __shared__ double s[]; // The whole chunk of shared memory

    double* shared_src = s;
    double* shared_kernel = (double*)&shared_src[MAX_2D_THREADS_PER_BLOCK * tile_cols];

    const int num_sub_blocks = (tile_cols + blockDim.x) / blockDim.x;

    if(threadIdx.x < ker_cols){
        shared_kernel[threadIdx.x] = kernel_col[threadIdx.x];
    }

    // Find global Idx (in the image) of the head and tail of each block
    const int block_start_pix_col = blockIdx.x * MAX_2D_THREADS_PER_BLOCK;
    const int block_end_pix_col = block_start_pix_col + MAX_2D_THREADS_PER_BLOCK;
    const int block_src_end_pix_col = CMIN(block_end_pix_col, src_cols);

    const int block_start_pix_row = blockIdx.y * MAX_2D_THREADS_PER_BLOCK;
    const int block_end_pix_row = block_start_pix_row + MAX_2D_THREADS_PER_BLOCK;
    const int block_src_end_pix_row = CMIN(block_end_pix_row, src_rows);
    
    const int tile_start_col = block_start_pix_col - offset_cols;
    const int tile_start_row = block_start_pix_row;


    // Load the "padded" image into shared tile
    int local_id_row = threadIdx.y; // Local ID (Always in the tile)
    int pixel_id_row = tile_start_row + local_id_row; // Global Position
    
    for(unsigned sub_block_num = 0; sub_block_num < num_sub_blocks; sub_block_num++){
        int local_id_col = threadIdx.x + sub_block_num * blockDim.x;
        int pixel_id_col = tile_start_col + local_id_col;
        if(local_id_col >= 0 && local_id_col < tile_cols){
            if(pixel_id_row >= 0 && pixel_id_row < src_rows 
            && pixel_id_col >= 0 && pixel_id_col < src_cols){
                shared_src[local_id_row * tile_cols + local_id_col] = src[pixel_id_row][pixel_id_col];
            }
            else{
                shared_src[local_id_row * tile_cols + local_id_col] = 0.0;
            }
        }
    }

    __syncthreads();


    // Perform convolution
    local_id_row = threadIdx.y; // Local ID
    pixel_id_row = tile_start_row + local_id_row; // Global Position (for output)
    for(unsigned sub_block_num = 0; sub_block_num < num_sub_blocks; sub_block_num++){
        int local_id_col = threadIdx.x + sub_block_num * blockDim.x;
        int pixel_id_col = tile_start_col + local_id_col;
        double sum = 0.0;
        if(pixel_id_row >= block_start_pix_row && pixel_id_row < block_src_end_pix_row 
        && pixel_id_col >= block_start_pix_col && pixel_id_col < block_src_end_pix_col
        && local_id_col >= 0 && local_id_col < tile_cols){
            for(int kernel_indx = -offset_cols; kernel_indx <= offset_cols; kernel_indx++){
                 // The "conv indices" will always be in the tile
                 int conv_indx = local_id_col + kernel_indx;
                 sum += shared_kernel[offset_cols + kernel_indx]
                      * shared_src[local_id_row * tile_cols + conv_indx];
            }

            dst[pixel_id_row][pixel_id_col] = sum;
        }
    }

 }


  /**
 * 1D row convolution on GPU (Shared Memory)
 *
 *
 */
 __global__ void convGPURow (double **src, int src_rows, int src_cols, double *kernel_row, int ker_rows, double **dst){
    const int offset_rows = ker_rows / 2; // 3 -> 1, 4 -> 2, 5 -> 2 Only rows

    const int tile_rows = MAX_2D_THREADS_PER_BLOCK + 2*offset_rows; 
    
    extern __shared__ double s[]; // The whole chunk of shared memory

    double* shared_src = s;
    double* shared_kernel = (double*)&shared_src[tile_rows * MAX_2D_THREADS_PER_BLOCK];

    const int num_sub_blocks = (tile_rows + blockDim.y) / blockDim.y;

    if(threadIdx.y < ker_rows){
        shared_kernel[threadIdx.y] = kernel_row[threadIdx.y];
    }

    // Find global Idx (in the image) of the head and tail of each block
    const int block_start_pix_col = blockIdx.x * MAX_2D_THREADS_PER_BLOCK;
    const int block_end_pix_col = block_start_pix_col + MAX_2D_THREADS_PER_BLOCK;
    const int block_src_end_pix_col = CMIN(block_end_pix_col, src_cols);

    const int block_start_pix_row = blockIdx.y * MAX_2D_THREADS_PER_BLOCK;
    const int block_end_pix_row = block_start_pix_row + MAX_2D_THREADS_PER_BLOCK;
    const int block_src_end_pix_row = CMIN(block_end_pix_row, src_rows);
    
    const int tile_start_col = block_start_pix_col;
    const int tile_start_row = block_start_pix_row - offset_rows;


    // Load the "padded" image into shared tile
    int local_id_col = threadIdx.x; // Local ID (Always in the tile)
    int pixel_id_col = tile_start_col + local_id_col; // Global Position
    
    for(unsigned sub_block_num = 0; sub_block_num < num_sub_blocks; sub_block_num++){
        int local_id_row = threadIdx.y + sub_block_num * blockDim.y;
        int pixel_id_row = tile_start_row + local_id_row;
        if(local_id_row >= 0 && local_id_row < tile_rows){
            if(pixel_id_row >= 0 && pixel_id_row < src_rows 
            && pixel_id_col >= 0 && pixel_id_col < src_cols){
                shared_src[local_id_row * blockDim.x + local_id_col] = src[pixel_id_row][pixel_id_col];
            }
            else{
                shared_src[local_id_row * blockDim.x + local_id_col] = 0.0;
            }
        }
    }

    __syncthreads();

    // Perform convolution
    local_id_col = threadIdx.x; // Local ID
    pixel_id_col = tile_start_col + local_id_col; // Global position (for output)
    for(unsigned sub_block_num = 0; sub_block_num < num_sub_blocks; sub_block_num++){
        int local_id_row = threadIdx.y + sub_block_num * blockDim.y; // Local ID
        int pixel_id_row = tile_start_row + local_id_row; // Global position (for output)
        double sum = 0.0;
        if(pixel_id_row >= block_start_pix_row && pixel_id_row < block_src_end_pix_row 
        && pixel_id_col >= block_start_pix_col && pixel_id_col < block_src_end_pix_col
        && local_id_row >= 0 && local_id_row < tile_rows){
            for(int kernel_indy = -offset_rows; kernel_indy <= offset_rows; kernel_indy++){
                 // The "conv indices" will always be in the tile
                 int conv_indy = local_id_row + kernel_indy;
                 sum += shared_kernel[offset_rows + kernel_indy]
                      * shared_src[conv_indy * blockDim.x + local_id_col];
            }

            dst[pixel_id_row][pixel_id_col] = sum;
        }
    }

 }


 