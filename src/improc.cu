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
 * Get a 1D Gaussian kernel
 */
 double* get1DGaussianKernel(int rows, double sigma)
 {
	 Mat gauss = cv::getGaussianKernel(rows, sigma, CV_64F);

 
	 double *array;
	 cudaMallocManaged(&array, rows*sizeof(double));
	 for (int i=0; i< rows; ++i) {
			array[i] = gauss.at<double>(i);
	 }
 
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
 * Double threshold
 */
void doubleThreshold (double **src, int src_rows, int src_cols, double lo, double hi, double **dst){
	double t1, t2;

	/** Compute gradient map, magnitude, and normalize the gradient*/
	double gradientX[src_rows][src_cols];
	double gradientY[src_rows][src_cols];
	double gradientMag[src_rows][src_cols];
	int peaks[src_rows][src_cols];

	omp_set_num_threads(NUM_THREADS);
    t1 = omp_get_wtime();
	#pragma omp parallel for if (USE_OMP)
	for (int i = 1; i < src_rows - 1; i++) {
		for (int j = 1; j < src_cols - 1; j++) {
			gradientX[i][j] = (src[i+1][j] - src[i-1][j]) / 255.0;
			gradientY[i][j] = (src[i][j+1] - src[i][j-1]) / 255.0;
			gradientMag[i][j] = sqrt(pow(gradientX[i][j],2) + pow(gradientY[i][j],2));
			gradientX[i][j] /= gradientMag[i][j];
			gradientY[i][j] /= gradientMag[i][j];
		}
	}
	t2 = omp_get_wtime();
	printf("Compute gradient map [%d, %d] : %f ms\n", src_rows, src_cols, (t2-t1)*1000);
	

	/** Find peaks and find strong edges and non-edge pixel*/
	t1 = omp_get_wtime();
	#pragma omp parallel for if (USE_OMP)
	for (int i = 1; i < src_rows - 1; i++) {
		for (int j = 1; j < src_cols - 1; j++) {
			int forward_x = min(max(0, i + (int)round(gradientX[i][j])), src_rows-2);
			int forward_y = min(max(0, j + (int)round(gradientY[i][j])), src_cols-2);
			int backward_x = min(max(0, i - (int)round(gradientX[i][j])), src_rows-2);
			int backward_y = min(max(0, j - (int)round(gradientY[i][j])), src_cols-2);
			if (gradientMag[i][j] > gradientMag[forward_x][forward_y] && gradientMag[i][j] >= gradientMag[backward_x][backward_y] ||
				gradientMag[i][j] >= gradientMag[forward_x][forward_y] && gradientMag[i][j] > gradientMag[backward_x][backward_y]) {
					peaks[i][j] = 1;
					if (gradientMag[i][j] >= hi) {
						dst[i][j] = 255;
						// printf("Strong edge pixel (%d, %d)\n", i, j);
					}
					else if (gradientMag[i][j] < lo) dst[i][j] = 0;
				}
		}
	}
	t2 = omp_get_wtime();
	printf("Find strong edge [%d, %d] : %f ms\n", src_rows, src_cols, (t2-t1)*1000);

	/** Find weak edges*/
	t1 = omp_get_wtime();
	#pragma omp parallel for if (USE_OMP)
	for (int i = 1; i < src_rows - 1; i++) {
		for (int j = 1; j < src_cols - 1; j++) {
			if (peaks[i][j] != 1 || dst[i][j] > 0) continue;
			for (int r = -1; r <= 1; r++) {
				for (int c = -1; c <= 1; c++) {
					if (dst[i+r][j+c] == 255) {
						dst[i][j] = 255;
						// printf("Weak edge pixel (%d, %d)\n", i, j);
					}
				}
			}
		}
	}
	t2 = omp_get_wtime();
	printf("Find weak edge [%d, %d] : %f ms\n", src_rows, src_cols, (t2-t1)*1000);

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
	}

	/** Second pass */
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
	}
	t2 = omp_get_wtime();
	
	printf("DistTrans [%d, %d] : %gms\n", src_rows, src_cols, (t2-t1)*1000);
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
	printf("Dilation [%d, %d] : %f ms\n", src_rows, src_cols, (t2-t1)*1000);
}

/**
 * Non maximum supression 
 */

void nonMaxSupression(double **src, int src_rows, int src_cols, int t_rows, int t_cols, double p, double **dst){
	double t1, t2;
	double sizeGB = src_rows * src_cols * sizeof(double) / (1024.0 * 1024.0 * 1024.0);
	omp_set_num_threads(NUM_THREADS);
    
	/** Get the max */
	double global_max = -1;
	t1 = omp_get_wtime();
	#pragma omp parallel for collapse(2) reduction(max:global_max) if (USE_OMP)
	for (int x = 0; x < src_rows; x++) {
		for (int y = 0; y < src_cols; y++) {
			if (global_max < src[x][y])
				global_max = src[x][y];
		}
	}


	int offset_x = t_rows / 2;
	int offset_y = t_cols / 2;
	double threshold = global_max * p;
	#pragma omp parallel for collapse(2) if (USE_OMP)
	for (int x = 1; x < src_rows - 1; x++) {
		for (int y = 1; y < src_cols - 1; y++) {
			/** Threshold and find local maximum */
			if (src[x][y] >= threshold && 
				src[x][y] >= src[x+1][y] && 
				src[x][y] >= src[x-1][y] && 
				src[x][y] >= src[x][y+1] && 
				src[x][y] >= src[x][y-1]) {
				/** Non maximum supression */
				int local_max = -1;
				int local_sum = 0;
				for (int r = x - offset_x; r <= x + offset_x; r++) {
					if (r < 0 || r >= src_rows) continue;
					for (int c = y - offset_y; c <= y + offset_y; c++) {
						if (c < 0 || c >= src_cols) continue;
						if (local_max < src[r][c]) local_max = src[r][c];
						local_sum += dst[r][c];
					}
				}
				if (src[x][y] == local_max && local_sum == 0) {
					dst[x][y] = src[x][y];
				}
			}
		}
	}
	t2 = omp_get_wtime();

	printf("Non maximum supression [%d, %d] : %gms\n", src_rows, src_cols, (t2-t1)*1000);
}


void drawBox(double **src, int src_rows, int src_cols, int t_rows, int t_cols, Mat &img_rgb){
	double t1, t2;
    double sizeGB = src_rows * src_cols * sizeof(double) / (1024.0 * 1024.0 * 1024.0);
    //omp_set_num_threads(NUM_THREADS);
	int offset_x = t_rows / 2;
        int offset_y = t_cols / 2;

	for (int x = 0; x < src_rows; x++) {
		for (int y = 0; y < src_cols; y++) {
			if (src[x][y] == 0) continue;
			int top = max(x - offset_x, 0);
			int buttom = min(x + offset_x, src_cols - 1);
			int left = max(y - offset_y, 0);
			int right = min(y + offset_y, src_rows - 1);
			for (int j = left; j <= right; j++) {
				img_rgb.at<Vec3b>(top, j) = Vec3b(0,0,255);
				img_rgb.at<Vec3b>(buttom, j) = Vec3b(0,0,255);
				// img_rgb.at<Vec3b>(top, j)[0] = 255;
				// img_rgb.at<Vec3b>(top, j)[1] = 0;
				// img_rgb.at<Vec3b>(top, j)[2] = 0;
				// img_rgb.at<Vec3b>(buttom, j)[0] = 255;
				// img_rgb.at<Vec3b>(buttom, j)[0] = 0;
				// img_rgb.at<Vec3b>(buttom, j)[0] = 0;
				// cout << img_rgb.at<Vec3b>(buttom, j);
			}
			for (int i = top; i <= buttom; i++) {
				img_rgb.at<Vec3b>(i, left) = Vec3b(0,0,255);
				img_rgb.at<Vec3b>(i, right) = Vec3b(0,0,255);

				// img_rgb.at<Vec3b>(i, left)[0] = 255;
				// img_rgb.at<Vec3b>(i, left)[1] = 0;
				// img_rgb.at<Vec3b>(i, left)[2] = 0;
				// img_rgb.at<Vec3b>(i, right)[0] = 255;
				// img_rgb.at<Vec3b>(i, right)[0] = 0;
				// img_rgb.at<Vec3b>(i, right)[0] = 0;
            }
		}
	}
}
