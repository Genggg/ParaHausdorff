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
	
    /** Gaussian filtering test */
    double **src = img2Array(image);
	double **dst = memAlloc2D(img_rows, img_cols);
    double **gauss_kernel = getGaussianKernel(ker_rows,ker_cols,2,2);
	conv(src, img_rows, img_cols, gauss_kernel, ker_rows, ker_cols, dst);
	Mat res = array2Img(dst, img_rows, img_cols);
	imwrite( "Smoothed_Image.jpg", res);

    /** Distance transform test */
    double **dst1 = memAlloc2D(img_rows, img_cols);
    distTrans(src, img_rows, img_cols ,dst1);
    Mat res1 = array2Img(dst1, img_rows, img_cols);
	imwrite( "cell_distance.jpg", res1);

	/** Image dilation test */
	double **dst2 = memAlloc2D(img_rows, img_cols);
	dilate(dst1, img_rows, img_cols, 3, dst2);
	Mat res2 = array2Img(dst2, img_rows, img_cols);
	imwrite( "dilated_cell.jpg", res2);

	return 0;											
}
