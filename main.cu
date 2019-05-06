#include "improc.h"
#include "gpukernels.h"
#include <cstdio>

using namespace cv;
using namespace std;

#define MAX_CUDA_THREADS_PER_BLOCK 32

int main(int argc, char** argv)
{
	/** Unified memory pointers */
	char* imageName = argv[1];
	Mat image;
	image = imread(imageName, 0);
	int img_rows = image.rows;
	int img_cols = image.cols;
	const int ker_rows = 5;
	const int ker_cols = 5;
	
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
	
	/** Test on GPU Global Gaussian Filtering Kernel */ 

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

	// Assign the number of blocks
	const unsigned num_threads_row = MAX_CUDA_THREADS_PER_BLOCK;
	const unsigned num_threads_col = MAX_CUDA_THREADS_PER_BLOCK;
	// Block dim: 32 x 32
	const unsigned num_blocks_row = (img_rows + num_threads_row) / num_threads_row;
	const unsigned num_blocks_col = (img_cols + num_threads_col) / num_threads_col;

	const dim3 num_blocks (num_blocks_col, num_blocks_row);
	const dim3 num_threads (num_threads_col, num_threads_row);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	// No shared memory
	convGPUGlobal<<< num_blocks, num_threads >>>
	 (src, img_rows, img_cols, gauss_kernel, ker_rows, ker_cols, dstg);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	fprintf(stdout, "Done Gaussian-Global on GPU.\n");
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); 
	fprintf(stdout, "Time elapsed: %f ms\n", elapsedTime);

	// Transfer the output to the CPU
	fprintf(stdout, "Memory copy done.\n");
	cv::Mat resg = array2Img(dstg, img_rows, img_cols);
	imwrite( "Smoothed_Image_GPU.jpg", resg);


	return 0;											
}
