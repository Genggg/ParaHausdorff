#include "improc.h"
#include "gpukernels.h"
#include <cstdio>

using namespace cv;
using namespace std;

#define MAX_2D_THREADS_PER_BLOCK 32
#define MAX_THREADS_PER_BLOCK 1024

int main(int argc, char** argv)
{
	
	char* imageName = argv[1];
	Mat image;
	image = imread(imageName, 0);
	int img_rows = image.rows;
	int img_cols = image.cols;
	const int ker_rows = 15;
	const int ker_cols = 15;
	const int offset_rows = ker_rows / 2; // 3 -> 1, 4 -> 2, 5 -> 2, also the size of apron
    const int offset_cols = ker_cols / 2; // 3 -> 1, 4 -> 2, 5 -> 2
	
    /** Load the image using OpenCV */
	double **src = img2Array(image);


	/** Gaussian filtering test */
	double **dst = cudaMallocManaged2D(img_rows, img_cols);
	double **gauss_kernel = getGaussianKernel(ker_rows,ker_cols,2,2);


	conv(src, img_rows, img_cols, gauss_kernel, ker_rows, ker_cols, dst);
	Mat res = array2Img(dst, img_rows, img_cols);
	imwrite( "Smoothed_Image.jpg", res);

    /** Distance transform test */
	double **dst1 = cudaMallocManaged2D(img_rows, img_cols);
    distTrans(src, img_rows, img_cols ,dst1);
    Mat res1 = array2Img(dst1, img_rows, img_cols);
	imwrite( "cell_distance.jpg", res1);

	/** Image dilation test */
	double **dst2 = cudaMallocManaged2D(img_rows, img_cols);
	dilate(dst1, img_rows, img_cols, 2, dst2);
	Mat res2 = array2Img(dst2, img_rows, img_cols);
	imwrite( "dilated_cell.jpg", res2);

	/** Non maximum supression test */
	int t_rows = 10;
	int t_cols = 10;
	double p = 0.9;
	double **dst3 = memAlloc2D(img_rows, img_cols);
	nonMaxSupression(dst2, img_rows, img_cols, t_rows, t_cols, p, dst3);
	Mat res3 = array2Img(dst3, img_rows, img_cols);
	imwrite( "nms_cell.jpg", res3);
	
	/** Test on GPU Gaussian Filtering Kernel on Global Memory */ 

	// Show some related infomation regarding the GPU

		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		if (deviceCount == 0) {
			fprintf(stderr, "error: no devices supporting CUDA.\n");
			exit(EXIT_FAILURE);
		}
		int dev = 0;
		cudaSetDevice(dev);
	
		cudaDeviceProp devProps;
		if (cudaGetDeviceProperties(&devProps, dev) == 0)
		{
			printf("Using device %d:\n", dev);
			printf("%s; Global Memory: %fGB; Shared Memory/block: %lu KB; Compute v%d.%d; Clock: %f GHz\n",
				devProps.name, (float)devProps.totalGlobalMem / (1024*1024*1024), 
				devProps.sharedMemPerBlock/(1024), (int)devProps.major, (int)devProps.minor, 
				(float)devProps.clockRate/(1000*1000));
		}

	
	double **dstg = cudaMallocManaged2D(img_rows, img_cols);

	// Block dim: 32 x 32
	const unsigned num_threads_row = MAX_2D_THREADS_PER_BLOCK;
	const unsigned num_threads_col = MAX_2D_THREADS_PER_BLOCK;
	// Assign the number of blocks
	const unsigned num_blocks_row = (img_rows + num_threads_row) / num_threads_row;
	const unsigned num_blocks_col = (img_cols + num_threads_col) / num_threads_col;

	const dim3 num_blocks (num_blocks_col, num_blocks_row);
	const dim3 num_threads (num_threads_col, num_threads_row);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	// No shared memory
	for(int i = 0; i < 100; ++i){
		convGPUGlobal<<< num_blocks, num_threads >>>
		(src, img_rows, img_cols, gauss_kernel, ker_rows, ker_cols, dstg);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	fprintf(stdout, "Done Gaussian-Global on GPU.\n");
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); 
	fprintf(stdout, "Time elapsed: %f ms\n", elapsedTime/100);

	cv::Mat resg = array2Img(dstg, img_rows, img_cols);
	imwrite( "Smoothed_Image_GPU.jpg", resg);


	/** Test on GPU Gaussian Filtering Kernel on Shared Memory */ 

	double **dstgs = cudaMallocManaged2D(img_rows, img_cols);

	// Block dim: H x (32 + 2 * Apron); Total number of threads < 1024
	const unsigned tile_cols = MAX_2D_THREADS_PER_BLOCK + 2 * offset_cols;
	const unsigned tile_rows = MAX_2D_THREADS_PER_BLOCK + 2 * offset_rows;

	const unsigned num_threads_col_s = tile_cols; // blockDim.x
	const unsigned num_threads_row_s = MAX_THREADS_PER_BLOCK / num_threads_col_s; // blockDim.y
	
	// Assign the number of blocks
	const unsigned num_blocks_col_s = (img_cols + MAX_2D_THREADS_PER_BLOCK) / MAX_2D_THREADS_PER_BLOCK;
	const unsigned num_blocks_row_large_s = (img_rows + MAX_2D_THREADS_PER_BLOCK) / MAX_2D_THREADS_PER_BLOCK;
	const unsigned sub_blocks_per_large = (tile_rows + num_threads_row_s) / num_threads_row_s;
	const unsigned num_blocks_row_s = num_blocks_row_large_s * sub_blocks_per_large;


	const dim3 num_blocks_s (num_blocks_col_s, num_blocks_row_s);
	const dim3 num_threads_s (num_threads_col_s, num_threads_row_s);

	const int TILE_BYTES = sizeof(double) * tile_cols * tile_rows;
	const int KERN_BYTES = sizeof(double) * ker_cols * ker_rows;

	cudaEvent_t start_s, stop_s;
    cudaEventCreate(&start_s);
	cudaEventCreate(&stop_s);
	cudaEventRecord(start_s, 0);
	for(int i = 0; i < 100; ++i){
	convGPUShared<<< num_blocks_s, num_threads_s, TILE_BYTES + KERN_BYTES >>>
		(src, img_rows, img_cols, gauss_kernel, ker_rows, ker_cols, dstgs);
	}
	cudaEventRecord(stop_s, 0);
	cudaEventSynchronize(stop_s);
	fprintf(stdout, "Done Gaussian-Shared on GPU.\n");
	float elapsedTime_g;
	cudaEventElapsedTime(&elapsedTime_g, start_s, stop_s); 
	fprintf(stdout, "Time elapsed: %f ms\n", elapsedTime_g/100);

	cv::Mat resgs = array2Img(dstgs, img_rows, img_cols);
	imwrite( "Smoothed_Image_GPUs.jpg", resgs);



	return 0;											
}
