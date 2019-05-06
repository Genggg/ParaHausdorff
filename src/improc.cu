#include "improc.h"
#define USE_OMP 1
#define NUM_THREADS 8
using namespace cv;
using namespace std;

// #define NORMAL_ALLOC
#define DEBUG

/**
 * Allocate 2D memory on CPU
 */
#ifdef NORMAL_ALLOC
double** memAlloc2D(int rows, int cols) {
	double **array = new double*[rows];
	for (int i = 0; i < rows; ++i) 
    	array[i] = new double[cols];
	return array;
}
#else
double** memAlloc2D(int rows, int cols) {
	double **array;
	array = new double *[rows];
	array[0] = new double [rows*cols];
	for (int i = 0; i < rows; ++i) 
    	array[i] = array[0] + i * cols;
	return array;
}
#endif

/**
 * Allocate 2D unified memory
 */
double** cudaMallocManaged2D(int rows, int cols){
	double **array;
	cudaMallocManaged(&array, rows*sizeof(double*));
    cudaMallocManaged(&array[0], rows*cols*sizeof(double));
    for (unsigned i = 1; i < rows; ++i)
		array[i] = array[0] + i*cols;
	return array;
}

/**
 * Get a 2D Gaussian kernel
 */
double** getGaussianKernel(int rows, int cols, double sigmax, double sigmay)
{
    Mat gauss_x = cv::getGaussianKernel(cols, sigmax, CV_64F);
    Mat gauss_y = cv::getGaussianKernel(rows, sigmay, CV_64F);
    Mat gauss_2d = gauss_x * gauss_y.t();
	// cout << gauss_2d << endl << endl;

	double **array = cudaMallocManaged2D(rows, cols);
	for (int i=0; i< rows; ++i) {
		for (int j = 0; j < cols; j++) {
			array[i][j] = gauss_2d.at<double>(i,j);
 		}
	}

    return array;
}

/**
 * Store an image in a 2D array
 */
double** img2Array(Mat img) {
	int rows = img.rows;
	int cols = img.cols;
	#ifdef DEBUG
		cout << "Loading image using OpenCV" << endl;
	#endif
	double **array = cudaMallocManaged2D(rows, cols);
	for (int i = 0; i < rows; i++) 
		for (int j = 0; j < cols; j++) 
			array[i][j] = (double)img.at<uchar>(i,j);
	return array;
}

/**
 * Convert a 2D array to a Mat
 */
Mat array2Img(double **src, int rows, int cols) {
	Mat dst(rows, cols, CV_8U);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			dst.at<uchar>(i,j) = (uchar)src[i][j]; 
	return dst;
}

/**
 * Convolution
 */
void conv(double **src, int src_rows, int src_cols, double **kernel, int ker_rows, int ker_cols, double **dst) {
	
	int offset_x = ker_rows / 2;
	int offset_y = ker_cols / 2;
	double t1, t2;
	double sizeGB = src_rows * src_cols * sizeof(double) / (1024.0 * 1024.0 * 1024.0);

	omp_set_num_threads(NUM_THREADS);
    t1 = omp_get_wtime();
	#pragma omp parallel for if (USE_OMP)
	for (int x = 0; x < src_rows; x++) {
		for (int y = 0; y < src_cols; y++) {
			double sum = 0;
			for (int r = x - offset_x, i = 0; r <= x + offset_x && i < ker_rows; r++, i++) {
				if (r < 0 || r >= src_rows) continue;
				for (int c = y - offset_y, j = 0; c <= y + offset_y && j < ker_cols; c++, j++) {
					if (c < 0 || c >= src_cols) continue;
					sum += src[r][c] * kernel[i][j];
				}
			}
			dst[x][y] = sum;
		}
	}

	t2 = omp_get_wtime();
    printf("Convolution [%d, %d] : %f ms\n", src_rows, src_cols, (t2-t1)*1000);
}

/**
 * Distance Transform
 */
void distTrans(double **src, int src_rows, int src_cols, double **dst) {

	double t1, t2;
	double sizeGB = src_rows * src_cols * sizeof(double) / (1024.0 * 1024.0 * 1024.0);
	omp_set_num_threads(NUM_THREADS);
    t1 = omp_get_wtime();


	/** Initialize distance map */
	#pragma omp parallel for collapse(2) if (USE_OMP)
	for (int i = 0; i < src_rows; i++) {
		for (int j = 0; j < src_cols; j++) {
			if (src[i][j] > 0) dst[i][j] = 0;
			else dst[i][j] = (double)INT_MAX;
		}
	}

	t1 = omp_get_wtime();
	/** First pass */
	for (int k = 0; k <= src_rows + src_cols -2; k++) {
		int start;
		int length;
		int *index_i;
		if (src_rows < src_cols) {
			if (k < src_rows) {
				length = k + 1;
				start = k;
			}
			else if (k < src_cols) {
				length = src_rows;
				start = src_rows - 1;
			}
			else {
				length = src_rows + src_cols - k - 1;
				start = src_rows - 1;
			}
		}
		else {
			if (k < src_cols) {
				length = k + 1;
				start = k;
			}
			else if (k < src_rows) {
				length = src_cols;
				start = k;
			}
			else {
				length = src_rows + src_cols - k - 1;
				start = src_rows - 1;
			}
		}

		index_i = new int[length];
		for (int r = 0; r < length; r++)
			index_i[r] = start--;
		
		#pragma omp parallel for if (USE_OMP)
		for (int r = 0; r < length; r++) {
			int i = index_i[r];
			int j = k - i;
			if (i == 0 && j == 0) continue;
			else if (i == 0) dst[i][j] = min(dst[i][j], dst[i][j-1] + 1);
			else if (j == 0) dst[i][j] = min(dst[i][j], dst[i-1][j] + 1);
			else dst[i][j] = min(dst[i][j], min(dst[i][j-1], dst[i-1][j]) + 1);
		}
		
		delete[] index_i;

		// for (int i = min(k, src_rows-1); i >= 0 && k - i < src_cols; i--) {
		// 	int j = k - i;
		// 	if (i == 0 && j == 0) continue;
		// 	else if (i == 0) dst[i][j] = min(dst[i][j], dst[i][j-1] + 1);
		// 	else if (j == 0) dst[i][j] = min(dst[i][j], dst[i-1][j] + 1);
		// 	else dst[i][j] = min(dst[i][j], min(dst[i][j-1], dst[i-1][j]) + 1);
		// }
	}
	t2 = omp_get_wtime();
	printf("1st pass [%d, %d] : %f ms\n", src_rows, src_cols, (t2-t1)*1000);

	/** Second pass */
	t1 = omp_get_wtime();
	for (int k = src_rows + src_cols - 2; k >= 0; k--) {
		int start;
		int length;
		int *index_j;

		if (src_rows < src_cols) {
			if (k >= src_cols) {
				length = src_rows + src_cols - k - 1;
				start = src_cols - 1;
			}
			else if (k < src_rows) {
				length = k + 1;
				start = k;
			}
			else {
				length = src_rows;
				start = k;
			}
		}
		else {
			if (k >= src_rows) {
				length = src_rows + src_cols - k - 1;
				start = src_cols - 1;
			}
			else if (k < src_cols) {
				length = k + 1;
				start = k;
			}
			else {
				length = src_cols;
				start = src_cols - 1;
			}
		}
		index_j = new int[length];
		for (int r = 0; r < length; r++)
			index_j[r] = start--;
		
		#pragma omp parallel for if (USE_OMP)
		for (int r = 0; r < length; r++) {
			int j = index_j[r];
			int i = k - j;
			if (i == src_rows - 1 && j == src_cols - 1) continue;
			else if (i == src_rows - 1) dst[i][j] = min(dst[i][j], dst[i][j+1] + 1);
			else if (j == src_cols - 1) dst[i][j] = min(dst[i][j], dst[i+1][j] + 1);
			else dst[i][j] = min(dst[i][j], min(dst[i][j+1], dst[i+1][j]) + 1);
		}

		delete[] index_j;

		// for (int j = min(k, src_cols-1); j >= 0 && k - j < src_rows; j--) {
		// 	int i = k - j;
		// 	if (i == src_rows - 1 && j == src_cols - 1) continue;
		// 	else if (i == src_rows - 1) dst[i][j] = min(dst[i][j], dst[i][j+1] + 1);
		// 	else if (j == src_cols - 1) dst[i][j] = min(dst[i][j], dst[i+1][j] + 1);
		// 	else dst[i][j] = min(dst[i][j], min(dst[i][j+1], dst[i+1][j]) + 1);
		// }
	}
	t2 = omp_get_wtime();
	printf("2nd pass [%d, %d] : %f ms.\n", src_rows, src_cols, (t2-t1)*1000);
	// printf("DistTrans [%d, %d] : %gs\n", src_rows, src_cols, t2-t1);
}

/**
 * Dilate the binary image based on its distance map
 */
void dilate(double **src, int src_rows, int src_cols, int d, double **dst) {
	double t1, t2;
	double sizeGB = src_rows * src_cols * sizeof(double) / (1024.0 * 1024.0 * 1024.0);

	omp_set_num_threads(NUM_THREADS);
    t1 = omp_get_wtime();
	#pragma omp parallel for if (USE_OMP)
	for (int i = 0; i < src_rows; i++) {
		for (int j = 0; j < src_cols; j++) {
			if (src[i][j] <= d) dst[i][j] = 255;
		}
	}
	t2 = omp_get_wtime();
	printf("Dilation [%d, %d] : %f ms.\n", src_rows, src_cols, (t2-t1)*1000);
}