#include "stitcher.h"

#include <iostream>
#include <fstream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;


// Optimizing
////https://software.intel.com/en-us/articles/fast-panorama-stitching


VideoCapture camera;
static string input = "/home/riemer/101GOPRO/video/GOPR0173.MP4";
static long   start = 80 * 1000;
static long   end   = 92 * 1000;

// Files
/*static vector<string> img_names = {
	"dataset2/img01",
//	"dataset2/img02",
//	"dataset2/img03",
//	"dataset2/img04",
	"dataset2/img05",
	"dataset2/img06",
	"dataset2/img07",
//	"dataset2/img08",
//	"dataset2/img09",
	"dataset2/img10",
//	"dataset2/img11",
//	"dataset2/img12",
	"dataset2/img13",
//	"dataset2/img14",
//	"dataset2/img15",
	"dataset2/img16",
//	"dataset2/img17",
//	"dataset2/img18",
	"dataset2/img19",
//	"dataset2/img20",
//	"dataset2/img21",
	"dataset2/img22",
//	"dataset2/img23",
//	"dataset2/img24",
	"dataset2/img25",
	"result1",
	"result2",
};*/
static string result_name = "result";

// Size of the input (non-stitched) images
static Size img_size(1920, 1080);


// Program options, to be set via cli arg?
static bool arg_remap      = true;
static bool arg_have_masks = false;


int main(int argc, char* argv[])
{
	Mat map1, map2;
	vector<Mat> images, images_masks;
	images.reserve( 10); // TODO calculate 10

	if(arg_have_masks)
		images_masks.reserve( 10);

	// Load images and remap them
	/*for (unsigned int i = 0; i < img_names.size(); ++i) {
		Mat temp = imread(img_names[i] + ".jpg");

		if (temp.empty()) {
			cerr << "Can't open image " << img_names[i] << ".jpg" << endl;
			return -1;
		}

		images.push_back( temp);

#if 0
		// Output remapped images to hdd
		int zeros = 0;
		if( i < 10) zeros = 1;
		string file = "temp/" + result_name + std::string( zeros, '0') + std::to_string( i ) + ".jpg";
		imwrite(file, images[i]);
#endif

		if(arg_have_masks) {
			temp = imread(img_names[i] + "_mask.jpg", CV_8U);

			if (temp.empty()) {
				cerr << "Can't open image " << img_names[i] << "_mask.jpg"  << endl;
				return -1;
			}

			images_masks.push_back( temp);
		}
	}
	map1.release();
	map2.release();*/

	camera.open( input);
	camera.set ( CV_CAP_PROP_POS_MSEC, start);

	int64 t = getTickCount();

	for( int i = 0; i < 10; ++i) {
		Mat temp;

		camera.read( temp);

		images.push_back( temp);

		for( int j = 0; j < 26; j++)
			camera.grab();

#if 1
		// Output remapped images to hdd
		int zeros = 0;
		if( i < 1000) zeros = 1;
		if( i <  100) zeros = 2;
		if( i <   10) zeros = 3;
		string file = "temp/" + result_name + std::string( zeros, '0') + std::to_string( i ) + ".jpg";
		imwrite(file, images[i]);
#endif
	}

	camera.release();
	map1.release();
	map2.release();
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << "\tsec, Video Read" << endl;


	// Tells which image to match with which
	Mat matchMask ( images.size(), images.size(), CV_8U, Scalar(0));
	for (unsigned int i = 0; i < images.size() -1; ++i)
		matchMask.at<char>(i,i+1) =1;

	// Stitch
	Mat result, result_mask;
	Stitcher stitcher;
	stitcher.set_img_res       (img_size.area());
	stitcher.set_feat_res      (1.0 * 1e6);
	stitcher.set_seam_res      (0.1 * 1e6);
	stitcher.set_feature_finder( Ptr<FeaturesFinder>( new OrbFeaturesFinder( Size(3,1), 3500)));
	/*stitcher.set_comp_res      (1.0 * 1e6);*/
	stitcher.set_conf_adjustor (0.95f);
	stitcher.set_conf_featurematching( 0.5f);

	Status ret = stitcher.stitch( images, images_masks, result, result_mask, matchMask);
	if( ret == Status::OK)
	{
		// Save
		string file = result_name + ".jpg";
		imwrite(file, result);

		file = result_name + "_mask.jpg";
		imwrite(file, result_mask);
	}
	else
		cerr << "Stitching failed with errorcode: " << (int)ret << endl;

	return (int)ret;
}
