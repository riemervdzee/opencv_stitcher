#include "stitcher.h"

#include <iostream>

#include "opencv2/stitching/warpers.hpp"

#define DEBUG

using namespace std;
using namespace cv;
using namespace cv::detail;

Stitcher::Stitcher()
{
	set_feat_res( 0.6 * 1e6);
	set_seam_res( 0.1 * 1e6);
	set_comp_res( Stitcher::ORIGINAL_RES);
	set_conf_featurematching( 0.3f);
	set_conf_adjustor( 1.f);
	set_feature_finder( Ptr<FeaturesFinder>( new OrbFeaturesFinder()));
	set_adjuster( Ptr<BundleAdjusterBase>( new BundleAdjusterRay()));
	set_adjuster_mask( "xxxxx");
	set_exposure_type( ExposureCompensator::GAIN);
	set_seam_finder( Ptr<GraphCutSeamFinder>( new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR)));
	set_blend_type( Blender::MULTI_BAND);
	set_blend_strength( 5.f);
}

void Stitcher::stitch( std::vector<cv::Mat> &input,
					   std::vector<cv::Mat> &input_masks,
					   cv::Mat &result,
					   cv::Mat &result_mask,
					   cv::Size &imgSize)
{
#ifdef DEBUG
	cv::setBreakOnError(true);

	cout << "Finding features:" << endl;
	int64 t = getTickCount();
	int64 app_start_time = getTickCount();
#endif

	vector<Mat> images = input;
	vector<ImageFeatures> features( images.size());
	Mat temp;

	double feat_factor = sqrt( feat_res_ /  imgSize.area());
	double seam_factor = sqrt( seam_res_ / feat_res_);
	double comp_factor = 1;

	if( comp_res_ != Stitcher::ORIGINAL_RES)
		comp_factor = sqrt( comp_res_ / imgSize.area());

	for (size_t i = 0; i < images.size(); ++i)
	{
		resize( images[i], temp, Size(), feat_factor, feat_factor);

		feature_finder_->operator()( temp, features[i]);
		features[i].img_idx = i;

		resize( temp, images[i], Size(), seam_factor, seam_factor);

#ifdef DEBUG
		cout << "\tImage #" << (i+1) << ": " << features[i].keypoints.size() << endl;
#endif
	}

	feature_finder_->collectGarbage();
	temp.release();

#ifdef DEBUG
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t finding features" << endl;
	t = getTickCount();
#endif

	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher( false, conf_featurematching_);

	matcher( features, pairwise_matches, matching_mask_);
	matcher.collectGarbage();

#ifdef DEBUG
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Pairwise matching" << endl;
	t = getTickCount();
#endif

	HomographyBasedEstimator estimator;
	vector<CameraParams> cameras;
	estimator( features, pairwise_matches, cameras);

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Estimator" << endl;
	t = getTickCount();

	for (size_t i = 0; i < cameras.size(); ++i) {
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		//cout << "Initial intrinsics #" << (i+1) << ":\n" << cameras[i].K() << endl;
	}

	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (adjuster_mask_[0] == 'x') refine_mask(0,0) = 1;
	if (adjuster_mask_[1] == 'x') refine_mask(0,1) = 1;
	if (adjuster_mask_[2] == 'x') refine_mask(0,2) = 1;
	if (adjuster_mask_[3] == 'x') refine_mask(1,1) = 1;
	if (adjuster_mask_[4] == 'x') refine_mask(1,2) = 1;

	adjuster_->setConfThresh( conf_adjustor_);
	adjuster_->setRefinementMask( refine_mask);
	adjuster_->operator()( features, pairwise_matches, cameras);

	// Cleanup
	features.clear();
	pairwise_matches.clear();

#ifdef DEBUG
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Adjustor" << endl;
	t = getTickCount();
#endif

	// Find median focal length
	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i) {
		//cout << "Camera #" << indices[i]+1 << ":\n" << cameras[i].K() << endl;
		focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end());
	double warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = focals[focals.size() / 2];
	else
		warped_image_scale = (focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	vector<Point> corners       (input.size());
	vector<Mat>   masks_warped  (input.size());
	vector<Mat>   images_warped (input.size());
	vector<Size>  sizes         (input.size());

	// Prepare images masks
	Mat mask( images[0].size(), CV_8U);

	// Warp images and their masks
	Ptr<WarperCreator> warper_creator;
	warper_creator = new cv::PlaneWarper();

	Ptr<RotationWarper> warper = warper_creator->create( warped_image_scale * seam_factor);

	for (size_t i = 0; i < input.size(); ++i) {

		mask.setTo(Scalar::all(255));

		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = seam_factor;
		K(0,0) *= swa; K(0,2) *= swa;
		K(1,1) *= swa; K(1,2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(mask, K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	// Clear memory
	images.clear();
	mask.release();

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(exposure_type_);
	compensator->feed(corners, images_warped, masks_warped);

	vector<Mat> images_warped_f(input.size());

	for (size_t i = 0; i < input.size(); ++i) {
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
		images_warped[i].release();
	}

	images_warped.clear();

#ifdef DEBUG
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t warping images" << endl;
	t = getTickCount();
#endif

	// Find seam
	seam_finder_->find(images_warped_f, corners, masks_warped);

	// Release unused memory
	images_warped_f.clear();

#ifdef DEBUG
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Seam finder" << endl;
	t = getTickCount();
#endif

	Mat dilated_mask, seam_mask, mask_warped;
	Ptr<Blender> blender;

	// Compute relative scales
	double compose_work_aspect = comp_factor/feat_factor;

	// Update warped image scale
	warped_image_scale *= compose_work_aspect;
	warper = warper_creator->create(warped_image_scale);

	// Update corners and sizes
	for (unsigned int i = 0; i < input.size(); ++i) {
		// Update intrinsics
		cameras[i].focal *= compose_work_aspect;
		cameras[i].ppx   *= compose_work_aspect;
		cameras[i].ppy   *= compose_work_aspect;

		// Update corner and size
		Size sz = imgSize;
		if( comp_res_ != Stitcher::ORIGINAL_RES)
		{
			sz.width  = cvRound(imgSize.width  * comp_factor);
			sz.height = cvRound(imgSize.height * comp_factor);
		}

		Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		Rect roi = warper->warpRoi(sz, K, cameras[i].R);
		corners[i] = roi.tl();
		sizes[i] = roi.size();
	}

	blender = Blender::createDefault(blend_type_, false);
	Size dst_sz = resultRoi(corners, sizes).size();
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength_ / 100.f;
	if (blend_width < 1.f)
		blender = Blender::createDefault(Blender::NO, false);
	else if (blend_type_ == Blender::MULTI_BAND)
	{
		MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
		mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
	}
	else if (blend_type_ == Blender::FEATHER)
	{
		FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
		fb->setSharpness(1.f/blend_width);
	}

	blender->prepare(corners, sizes);

	Mat img_warped, img_warped_s;
	Size sz;
	sz.width  = cvRound(imgSize.width  * comp_factor);
	sz.height = cvRound(imgSize.height * comp_factor);
	mask.create(sz, CV_8U);

	for (size_t i = 0; i < input.size(); ++i) {
		Mat K;
		cameras[i].K().convertTo(K, CV_32F);

		// Additional resize
		if( comp_res_ != Stitcher::ORIGINAL_RES)
			resize( input[i], temp, Size(), comp_factor, comp_factor);
		else
			temp = input[i];

		input[i].release();

		// Warp the current image
		warper->warp(temp, K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		// Warp the current image mask
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		compensator->apply(i, corners[i], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();

		dilate(masks_warped[i], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());

		bitwise_and(seam_mask, mask_warped, mask_warped);

		// Blend the current image
		blender->feed(img_warped_s, mask_warped, corners[i]);
	}

	//
	input.clear();
	input_masks.clear();

	//Mat result, result_mask;
	blender->blend(result, result_mask);

#ifdef DEBUG
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Compositing" << endl;
	cout << "Finished! total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" << endl;
#endif

	return;
}
