#include"improc.h"
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	char* imageName = argv[1];
	Mat image;
	image = imread(imageName, 0);
	int img_rows = image.rows;
	int img_cols = image.cols;
	int ker_rows = 5;
	int ker_cols = 5;
	double **gauss_kernel = getGaussianKernel(ker_rows,ker_cols,2,2);
	double **src = img2Array(image);
	double **dst = memAlloc2D(img_rows, img_cols);
	conv(src, img_rows, img_cols, gauss_kernel, ker_rows, ker_cols, dst);
	Mat res = array2Img(dst, img_rows, img_cols);
	imwrite( "Smoothed_Image.jpg", res);

	return 0;											
}
