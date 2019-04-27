#include"improc.h"
using namespace cv;
using namespace std;

/**
 * Allocate 2D memory on CPU
 */
double** memAlloc2D(int rows, int cols) {
	double **array = new double*[rows];
	for (int i = 0; i < rows; ++i) 
    	array[i] = new double[cols];
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

	double **array = memAlloc2D(rows, cols);
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
double** img2Array(Mat image) {
	int rows = image.rows;
	int cols = image.cols;
	// cout << "original: " << (int)image.at<uchar>(100,100) << endl;

	double **array = memAlloc2D(rows, cols);
	for (int i = 0; i < rows; i++) 
		for (int j = 0; j < cols; j++) 
			array[i][j] = (double)image.at<uchar>(i,j);
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
}