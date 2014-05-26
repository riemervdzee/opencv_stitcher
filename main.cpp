#include "stitcher.h"

#include <iostream>
#include <fstream>
#include <string>

#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

// Optimizing
////https://software.intel.com/en-us/articles/fast-panorama-stitching


// Files
static vector<string> img_names = {
	"dataset2/img01.jpg",
//	"dataset2/img02.jpg",
//	"dataset2/img03.jpg",
	"dataset2/img04.jpg",
//	"dataset2/img05.jpg",
//	"dataset2/img06.jpg",
	"dataset2/img07.jpg",
//	"dataset2/img08.jpg",
//	"dataset2/img09.jpg",
	"dataset2/img10.jpg",
//	"dataset2/img11.jpg",
//	"dataset2/img12.jpg",
	"dataset2/img13.jpg",
//	"dataset2/img14.jpg",
//	"dataset2/img15.jpg",
	"dataset2/img16.jpg",
//	"dataset2/img17.jpg",
//	"dataset2/img18.jpg",
	"dataset2/img19.jpg",
//	"dataset2/img20.jpg",
//	"dataset2/img21.jpg",
	"dataset2/img22.jpg",
//	"dataset2/img23.jpg",
//	"dataset2/img24.jpg",
	"dataset2/img25.jpg",
};
static string result_name = "result";

// Size of the input images
static Size img_size(1920, 1080);

// Our camera object
static Mat intrinsic = getDefaultNewCameraMatrix( Mat::eye( 3, 3, CV_64F), img_size, true);

// Distruption coeffecients
static float Coeffs[] = {-0.00000019f, 0.f, 0.f, 0.f};
static Mat distCoeffs = Mat( 4, 1, CV_32F, Coeffs);


int main(int argc, char* argv[])
{
	vector<Mat> images(img_names.size());
	Mat map1, map2, temp;

	// Calculate distortion maps
	initUndistortRectifyMap( intrinsic, distCoeffs, Mat(), Mat(), img_size, CV_32FC1, map1, map2);

	// Load images and remap them
	for (unsigned int i = 0; i < img_names.size(); ++i) {
		temp = imread(img_names[i]);

		if (temp.empty()) {
			cerr << "Can't open image " << img_names[i] << endl;
			return -1;
		}

		remap(temp, images[i], map1, map2, INTER_LINEAR );

#if 0
		int zeros = 0;
		if( i < 10) zeros = 1;
		string file = "temp/" + result_name + std::string( zeros, '0') + std::to_string( i ) + ".jpg";
		imwrite(file, images[i]);
#endif
	}
	temp.release();
	map1.release();
	map2.release();

	// Tells which image to match with which
	Mat matchMask (img_names.size(), img_names.size(), CV_8U, Scalar(0));
	for (unsigned int i = 0; i < img_names.size() -1; ++i)
		matchMask.at<char>(i,i+1) =1;

	// Stitch
	Mat result, result_mask;
	Stitcher stitcher;
	stitcher.set_matching_mask (matchMask);
	stitcher.set_feat_res      (0.8 * 1e6);
	stitcher.set_seam_res      (0.1 * 1e6);
	stitcher.set_feature_finder( Ptr<FeaturesFinder>( new OrbFeaturesFinder( Size(1,1), 3500)));
	stitcher.set_conf_adjustor (0.95f);
	stitcher.set_conf_featurematching( 0.35f);

	stitcher.stitch( images, result, result_mask, img_size);

	// Save
	string file = result_name + ".jpg";
	imwrite(file, result);

	file = result_name + "_mask.jpg";
	imwrite(file, result_mask);
}
