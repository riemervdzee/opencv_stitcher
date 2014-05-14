#include <iostream>
#include <fstream>
#include <string>

#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

// Optimizing
////https://software.intel.com/en-us/articles/fast-panorama-stitching


// Files
vector<string> img_names = {
	"dataset2/img01.jpg",
	"dataset2/img02.jpg",
	"dataset2/img03.jpg",
	"dataset2/img04.jpg",
	"dataset2/img05.jpg",
	"dataset2/img06.jpg",
	"dataset2/img07.jpg",
	"dataset2/img08.jpg",
	"dataset2/img09.jpg",
	"dataset2/img10.jpg",
	"dataset2/img11.jpg",
	"dataset2/img12.jpg"
};
string result_name = "result";

// Size of the input images
Size img_size(1920, 1080);

// We resize the working copies to smaller sizes
float feat_size = 0.2 * 1e6;
float feat_factor = sqrt( feat_size / static_cast<float>(img_size.area()));

// Options: SurfFeaturesFinder or OrbFeaturesFinder
OrbFeaturesFinder featureFinder;

// Confidences
float conf_featurematching = 0.3f;
float conf_adjustor        = 0.5f;

// options: BundleAdjusterReproj, BundleAdjusterRay
BundleAdjusterRay adjuster;
string ba_refine_mask = "xxxxx";

// Exposure type
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;

// Seamfinder algorithm. Options:
//		NoSeamFinder, VoronoiSeamFinder, GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR),
//		GraphCutSeamFinder, (GraphCutSeamFinderBase::COST_COLOR_GRAD), DpSeamFinder(DpSeamFinder::COLOR),
//		DpSeamFinder(DpSeamFinder::COLOR_GRAD)
GraphCutSeamFinder seam_finder(GraphCutSeamFinderBase::COST_COLOR);

// Options:
//		Blender::NO, Blender::FEATHER, Blender::MULTI_BAND
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;


int main(int argc, char* argv[])
{
	int64 app_start_time = getTickCount();

	cv::setBreakOnError(true);

	cout << "Finding features:" << endl;
	int64 t = getTickCount();

	Mat full_img, full_img2;
	vector<ImageFeatures> features(img_names.size());
	vector<Mat> images(img_names.size());
	vector<Size> full_img_sizes(img_names.size());

	// Our camera object
	Mat intrinsic = getDefaultNewCameraMatrix( Mat::eye( 3, 3, CV_64F), img_size, true);

	// Distruption coeffecients
	Mat distCoeffs = Mat::zeros( 1, 4, CV_32F);
	distCoeffs.at<float>(0) = -0.00000019;
	//distCoeffs.at<double>(1) = -0.00000000000001;

	// Distortion maps
	Mat map1, map2;
	initUndistortRectifyMap( intrinsic, distCoeffs, Mat(), Mat(), img_size, CV_32FC1, map1, map2);

	for (unsigned int i = 0; i < img_names.size(); ++i) {
		full_img = imread(img_names[i]);
		full_img_sizes[i] = full_img.size();

		if (full_img.empty()) {
			cerr << "Can't open image " << img_names[i] << endl;
			return -1;
		}

		remap(full_img, full_img2, map1, map2, INTER_LINEAR );

		resize(full_img2, images[i], Size(), feat_factor, feat_factor);

		featureFinder(images[i], features[i]);
		features[i].img_idx = i;
		cout << "\tImage #" << (i+1) << ": " << features[i].keypoints.size() << endl;

#if 0
		string file = result_name + std::to_string( i ) + ".jpg";
		imwrite(file, full_img2);
#endif
	}

	featureFinder.collectGarbage();
	full_img.release();
	full_img2.release();

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t finding features" << endl;
	t = getTickCount();

	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher(false, conf_featurematching);
	Mat matchMask(features.size(),features.size(),CV_8U,Scalar(0));
	for (unsigned int i = 0; i < img_names.size() -1; ++i)
		matchMask.at<char>(i,i+1) =1;

	matcher(features, pairwise_matches,matchMask);
	matcher.collectGarbage();

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Pairwise matching" << endl;
	t = getTickCount();


#if 0
	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<Mat> img_subset;
	vector<string> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i) {
		img_names_subset.push_back(img_names[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	img_names = img_names_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	img_names.size() = static_cast<int>(img_names.size());
	if (img_names.size() < 2) {
		cerr << "Error: Need more images" << endl;
		return -1;
	}

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Check panorama" << endl;
	t = getTickCount();
#endif

	HomographyBasedEstimator estimator;
	vector<CameraParams> cameras;
	estimator(features, pairwise_matches, cameras);

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Estimator" << endl;
	t = getTickCount();

	for (size_t i = 0; i < cameras.size(); ++i) {
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		//cout << "Initial intrinsics #" << (indices[i]+1) << ":\n" << cameras[i].K() << endl;
	}

	adjuster.setConfThresh(conf_adjustor);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
	adjuster.setRefinementMask(refine_mask);
	adjuster(features, pairwise_matches, cameras);

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Adjustor" << endl;
	t = getTickCount();

	// Find median focal length
	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i) {
		//cout << "Camera #" << indices[i]+1 << ":\n" << cameras[i].K() << endl;
		focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	vector<Point> corners(img_names.size());
	vector<Mat> masks_warped(img_names.size());
	vector<Mat> images_warped(img_names.size());
	vector<Size> sizes(img_names.size());
	vector<Mat> masks(img_names.size());

	// Prepapre images masks
	for (unsigned int i = 0; i < img_names.size(); ++i) {
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks
	Ptr<WarperCreator> warper_creator;
	warper_creator = new cv::PlaneWarper();

	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale));

	for (unsigned int i = 0; i < img_names.size(); ++i) {
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = 1.f;
		K(0,0) *= swa; K(0,2) *= swa;
		K(1,1) *= swa; K(1,2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	vector<Mat> images_warped_f(img_names.size());
	for (unsigned int i = 0; i < img_names.size(); ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t warping images" << endl;
	t = getTickCount();

	// Find seam
	seam_finder.find(images_warped_f, corners, masks_warped);

	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Seam finder" << endl;
	t = getTickCount();

	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Ptr<Blender> blender;

	// Compute relative scales
	float compose_work_aspect = 1 / feat_factor;

	// Update warped image scale
	warped_image_scale *= compose_work_aspect;
	warper = warper_creator->create(warped_image_scale);

	// Update corners and sizes
	for (unsigned int i = 0; i < img_names.size(); ++i) {
		// Update intrinsics
		cameras[i].focal *= compose_work_aspect;
		cameras[i].ppx *= compose_work_aspect;
		cameras[i].ppy *= compose_work_aspect;

		// Update corner and size
		Size sz = full_img_sizes[i];

		Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		Rect roi = warper->warpRoi(sz, K, cameras[i].R);
		corners[i] = roi.tl();
		sizes[i] = roi.size();
	}

	blender = Blender::createDefault(blend_type, false);
	Size dst_sz = resultRoi(corners, sizes).size();
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
	if (blend_width < 1.f)
        blender = Blender::createDefault(Blender::NO, false);
    else if (blend_type == Blender::MULTI_BAND)
    {
        MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
        mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
    }
    else if (blend_type == Blender::FEATHER)
    {
        FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
        fb->setSharpness(1.f/blend_width);
    }

	blender->prepare(corners, sizes);

	Mat img, img2;

	for (unsigned int i = 0; i < img_names.size(); ++i) {
		// Read image
		img = imread(img_names[i]);

		Size img_size = img.size();

		remap(img, img2, map1, map2, INTER_LINEAR );

		Mat K;
		cameras[i].K().convertTo(K, CV_32F);

		// Warp the current image
		warper->warp(img2, K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		compensator->apply(i, corners[i], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		img2.release();
		mask.release();

		dilate(masks_warped[i], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());
		mask_warped = seam_mask & mask_warped;

		// Blend the current image
		blender->feed(img_warped_s, mask_warped, corners[i]);
	}

	Mat result, result_mask;
	blender->blend(result, result_mask);

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Compositing" << endl;

	string file = result_name + ".jpg";
	imwrite(file, result);

	cout << "Finished! total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" << endl;

	return 0;
}
