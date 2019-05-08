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
	char* templName = argv[2];
	Mat image_rgb = imread(imageName, 1);
	Mat image = imread(imageName, 0);
	Mat templ = imread(templName, 0);

	const int img_rows = image.rows;
	const int img_cols = image.cols;
	const int tmp_rows = templ.rows;
	const int tmp_cols = templ.cols;
	const int ker_rows = 15;
	const int ker_cols = 15;
	const int offset_rows = ker_rows / 2; // 3 -> 1, 4 -> 2, 5 -> 2, also the size of apron
	const int offset_cols = ker_cols / 2; // 3 -> 1, 4 -> 2, 5 -> 2
	const int ker_tmp_rows = 15;
	const int ker_tmp_cols = 15;
	const int offset_tmp_cols = ker_tmp_cols / 2;
	const int offset_tmp_rows = ker_tmp_rows / 2;
	
    /** Load the image using OpenCV */
	double **src = img2Array(image);
	double **T = img2Array(templ);

	/** Gaussian filtering on CPU test */
	double **smoothed_img = cudaMallocManaged2D(img_rows, img_cols);
	double **gauss_kernel = getGaussianKernel(ker_rows,ker_cols,2,2);


	conv(src, img_rows, img_cols, gauss_kernel, ker_rows, ker_cols, smoothed_img);
	Mat res_smoothed = array2Img(smoothed_img, img_rows, img_cols);
	imwrite( "GaussFiltering_result.jpg", res_smoothed);
	
	/** Test on the GPU Gaussian Filtering Kernel on Global Memory */ 

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


	/** Test on the GPU Gaussian Filtering Kernel on Shared Memory */ 

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
	for(int i = 0; i < 10; ++i){
	convGPUShared<<< num_blocks_s, num_threads_s, TILE_BYTES + KERN_BYTES >>>
		(src, img_rows, img_cols, gauss_kernel, ker_rows, ker_cols, dstgs);
	}
	cudaEventRecord(stop_s, 0);
	cudaEventSynchronize(stop_s);
	fprintf(stdout, "Done Gaussian-Shared on GPU.\n");
	float elapsedTime_g;
	cudaEventElapsedTime(&elapsedTime_g, start_s, stop_s); 
	fprintf(stdout, "Time elapsed: %f ms\n", elapsedTime_g/10);

	cv::Mat resgs = array2Img(dstgs, img_rows, img_cols);
	imwrite( "Smoothed_Image_GPUs.jpg", resgs);

	/** Test on 1D GPU Gaussian Filtering Kernels on Shared Memory */


	double *gauss_1d_kernel = get1DGaussianKernel(ker_rows,2);
	double **dstgr = cudaMallocManaged2D(img_rows, img_cols);
	double **dstgc = cudaMallocManaged2D(img_rows, img_cols);

	// For 1D Filtering: Share memory size: tile x 32; Thread block size: 32 x 32
	// The number of blocks are also the same as the global one

	const int TILE_BYTES_COLS = sizeof(double) * tile_cols * num_threads_row;
	const int TILE_BYTES_ROWS = sizeof(double) * tile_rows * num_threads_col;

	const int KERN_BYTES_1D = sizeof(double) * ker_rows;

	cudaEvent_t start_1d, stop_1d;
    cudaEventCreate(&start_1d);
	cudaEventCreate(&stop_1d);
	cudaEventRecord(start_1d, 0);
	for(int i = 0; i < 10; ++i){
		convGPUCol<<< num_blocks, num_threads, TILE_BYTES_COLS + KERN_BYTES_1D >>>
			(src, img_rows, img_cols, gauss_1d_kernel, ker_cols, dstgc);
		convGPURow<<< num_blocks, num_threads, TILE_BYTES_ROWS + KERN_BYTES_1D >>>
		(dstgc, img_rows, img_cols, gauss_1d_kernel, ker_rows, dstgr);
	}

	cudaEventRecord(stop_1d, 0);
	cudaEventSynchronize(stop_1d);
	fprintf(stdout, "Done Seperate Gaussian-Shared on GPU.\n");
	float elapsedTime_1d;
	cudaEventElapsedTime(&elapsedTime_1d, start_1d, stop_1d); 
	fprintf(stdout, "Time elapsed: %f ms\n", elapsedTime_1d/10);

	cv::Mat resg1d = array2Img(dstgr, img_rows, img_cols);
	imwrite( "Smoothed_Image_GPUs-1d.jpg", resg1d);

	/** Gaussian filtering of template */
	double **dstmpr = cudaMallocManaged2D(tmp_rows, tmp_cols);
	double **dstmpc = cudaMallocManaged2D(tmp_rows, tmp_cols);

	const unsigned tile_tmp_cols = MAX_2D_THREADS_PER_BLOCK + 2 * offset_tmp_cols;
	const unsigned tile_tmp_rows = MAX_2D_THREADS_PER_BLOCK + 2 * offset_tmp_rows;

	const int TILE_TEMP_COLS = sizeof(double) * tile_tmp_cols * num_threads_row;
	const int TILE_TEMP_ROWS = sizeof(double) * tile_tmp_rows * num_threads_col;

	const int KERN_TEMP_1D = sizeof(double) * ker_tmp_rows;

	cudaEventCreate(&start_1d);
	cudaEventCreate(&stop_1d);
	cudaEventRecord(start_1d, 0);
	for(int i = 0; i < 10; i++){
		convGPUCol<<< num_blocks, num_threads, TILE_TEMP_COLS + KERN_TEMP_1D >>>
				(T, tmp_rows, tmp_cols, gauss_1d_kernel, ker_tmp_cols, dstmpc);
		convGPURow<<< num_blocks, num_threads, TILE_TEMP_ROWS + KERN_TEMP_1D >>>
		(dstmpc, tmp_rows, tmp_cols, gauss_1d_kernel, ker_tmp_rows, dstmpr);
	}
	cudaEventRecord(stop_1d, 0);
	cudaEventSynchronize(stop_1d);
	fprintf(stdout, "Done Seperate Gaussian-Shared on template[%d, %d].\n", tmp_rows, tmp_cols);
	cudaEventElapsedTime(&elapsedTime_1d, start_1d, stop_1d); 
	fprintf(stdout, "Time elapsed: %f ms\n", elapsedTime_1d/10);
	cv::Mat restmp = array2Img(dstmpr, tmp_rows, tmp_cols);
	imwrite( "Smoothed_Image_GPUs-T.jpg", restmp);




	/** Double threshold test */
	double lo = 0.008;
	double hi = 0.08;
	double **edge_map = cudaMallocManaged2D(img_rows, img_cols);
	doubleThreshold(dstgr, img_rows, img_cols, lo, hi, edge_map);
	Mat res_edge = array2Img(edge_map, img_rows, img_cols);
	imwrite("edge_result.jpg", res_edge);

	/** Double threshold on template */
	// double lo_tmp = 0.015;
	// double hi_tmp = 0.15;
	double **edge_template = cudaMallocManaged2D(tmp_rows, tmp_cols);
	doubleThreshold(dstmpr, tmp_rows, tmp_cols, lo, hi, edge_template);
	Mat tmp_edge = array2Img(edge_template, tmp_rows, tmp_cols);
	imwrite("edge_template.jpg", tmp_edge);

    /** Distance transform test */
	double **dist_map = cudaMallocManaged2D(img_rows, img_cols);
	distTrans(edge_map, img_rows, img_cols ,dist_map);
	Mat res_dist = array2Img(dist_map, img_rows, img_cols);
	imwrite("distTrans_result.jpg", res_dist);

	/** Image dilation test */
	double **dilated_img = cudaMallocManaged2D(img_rows, img_cols);
	dilate(dist_map, img_rows, img_cols, 2, dilated_img);
	Mat res_dilated = array2Img(dilated_img, img_rows, img_cols);
	imwrite( "dilation_result.jpg", res_dilated);

	/** Search matching test*/
	
	double **matched_map = cudaMallocManaged2D(img_rows, img_cols);
	printf("Search matching on CPU ");
	conv(dilated_img, img_rows, img_cols, edge_template, tmp_rows, tmp_cols, matched_map);
	Mat res_matched = array2Img(matched_map, img_rows, img_cols);
	imwrite( "search_result.jpg", res_matched);
	


    cudaEventCreate(&start_s);
	cudaEventCreate(&stop_s);
	cudaEventRecord(start_s, 0);
	for(int i = 0; i < 10; ++i){
	convGPUShared<<< num_blocks_s, num_threads_s, TILE_BYTES + KERN_BYTES >>>
		(edge_map, img_rows, img_cols, T, tmp_rows, tmp_cols, matched_map);
	}
	cudaEventRecord(stop_s, 0);
	cudaEventSynchronize(stop_s);
	cudaEventElapsedTime(&elapsedTime_g, start_s, stop_s); 
	fprintf(stdout, "Search matching on GPU: %f ms\n", elapsedTime_g/10);

	res_matched = array2Img(matched_map, img_rows, img_cols);
	imwrite( "search_result.jpg", res_matched);

	/** Non maximum supression test */
	int t_rows = tmp_rows;
	int t_cols = tmp_cols;
	double p = 0.75;
	double **nms_map = memAlloc2D(img_rows, img_cols);
	nonMaxSupression(matched_map, img_rows, img_cols, t_rows, t_cols, p, nms_map);
	Mat res_nms = array2Img(nms_map, img_rows, img_cols);
	imwrite( "nms_result.jpg", res_nms);

	/** Draw the matched result*/
	drawBox(nms_map, img_rows, img_cols, tmp_rows, tmp_cols, image_rgb);
	// Mat final_res = array2Img(src, img_rows, img_cols);
	imwrite("../result/final_result.jpg",image_rgb);



	return 0;											
}
