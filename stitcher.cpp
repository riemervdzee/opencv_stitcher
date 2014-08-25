/*
 * TODO:
 * - Remove features from mask area, not required so far
 */

#include "stitcher.h"

#include <iostream>

#include "opencv2/stitching/warpers.hpp"

#define DEBUG         1
#define DEBUG_CAMERA  0

using namespace std;
using namespace cv;
using namespace cv::detail;

Stitcher::Stitcher()
{
	set_img_res ( 2.0 * 1e6);
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

Status Stitcher::stitch( std::vector<cv::Mat> &input,
						 std::vector<cv::Mat> &input_masks,
						 cv::Mat &result,
						 cv::Mat &result_mask,
						 const cv::Mat &matching_mask,
						 std::vector<std::vector<cv::Rect>> input_roi,
						 std::vector<cv::detail::CameraParams> cameras)
{
#if DEBUG
	cv::setBreakOnError(true);

	cout << "Finding features:" << endl;
	int64 app_start_time = getTickCount();
	int64 t = getTickCount();
#endif

	// Check if have enough images
	// We want more than 2 images in general. input_masks is either zero or the same as input.size()
	if ( input.size() < 2 || (input_masks.size() != 0 && input.size() != input_masks.size()) )
	{
		cerr << "Incorrect amount of images.";
		cerr << "input.size() = " << input.size() << ", input_masks.size() = " << input_masks.size() << endl;
		return Status::ERR_NEED_MORE_IMGS;
	}

	// Get the factors from input-img->feat, img->seam, feat->seam and img->composition
	double factor_img_feat  = sqrt( feat_res_ /  img_res_);
	double factor_img_seam  = sqrt( seam_res_ /  img_res_);
	double factor_feat_seam = sqrt( seam_res_ / feat_res_);
	double comp_factor = 1;

	if( comp_res_ != Stitcher::ORIGINAL_RES)
		comp_factor = sqrt( comp_res_ / img_res_);

	// Update all ROIs to the img->feat resolution
	for( size_t i = 0; i < input_roi.size(); ++i) {
		for( size_t j = 0; j < input_roi[i].size(); ++j) {
			Rect roi = input_roi[i][j];
			input_roi[i][j] = Rect( roi.x     * factor_img_feat, roi.y      * factor_img_feat,
									roi.width * factor_img_feat, roi.height * factor_img_feat);
		}
	}

	// Loop through all images, resize to find features.
	vector<ImageFeatures> features( input.size());
	vector<Size> full_img_sizes( input.size());

	for (size_t i = 0; i < input.size(); ++i)
	{
		Mat temp;
		resize( input[i], temp, Size(), factor_img_feat, factor_img_feat);

		if( input_roi.size() != 0 && input_roi[i].size() != 0)
			feature_finder_->operator()( temp, features[i], input_roi[i]);
		else
			feature_finder_->operator()( temp, features[i]);

		features[i].img_idx = i;
		full_img_sizes[i]   = input[i].size();
	}
	feature_finder_->collectGarbage();

#if DEBUG
	for (size_t i = 0; i < input.size(); ++i)
		cout << "\tImage #" << (i+1) << ": " << features[i].keypoints.size() << endl;

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t finding features" << endl;
	t = getTickCount();
#endif


	// Now we match the features according the featurematching-mask
	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher( false, conf_featurematching_);

	matcher( features, pairwise_matches, matching_mask);
	matcher.collectGarbage();

#if DEBUG
	for(unsigned int i = 0; i < pairwise_matches.size(); ++i) {
		if( pairwise_matches[i].matches.size() == 0)
			continue;
		cout << "Matched " << pairwise_matches[i].src_img_idx << " with " << pairwise_matches[i].dst_img_idx << ", results: " << pairwise_matches[i].matches.size() << endl;
	}

	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Pairwise matching" << endl;
	t = getTickCount();
#endif


	if( cameras.empty())
	{
		// First attempt to get the images aligned to eachother
		HomographyBasedEstimator estimator;

		estimator( features, pairwise_matches, cameras);

		// Check if we failed (in OpenCV 3 we have better options to check this)
		if( isnan( cameras[0].R.at<float>(0,0)))
			return Status::ERR_HOMOGRAPHY_EST_FAIL;

#if DEBUG_CAMERA
		for (size_t i = 0; i < cameras.size(); ++i)
			cout << "Camera #" << i+1 << ":\n" << cameras[i].K() << "\n" << cameras[i].R << endl;
#endif
#if DEBUG
		cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Homography Estimator" << endl;
		t = getTickCount();
#endif
	}


	// Convert cameras to floats
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}

	// setup refinement mask
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (adjuster_mask_[0] == 'x') refine_mask(0,0) = 1;
	if (adjuster_mask_[1] == 'x') refine_mask(0,1) = 1;
	if (adjuster_mask_[2] == 'x') refine_mask(0,2) = 1;
	if (adjuster_mask_[3] == 'x') refine_mask(1,1) = 1;
	if (adjuster_mask_[4] == 'x') refine_mask(1,2) = 1;

	// Execute adjuster
	adjuster_->setConfThresh( conf_adjustor_);
	adjuster_->setRefinementMask( refine_mask);
	adjuster_->operator()( features, pairwise_matches, cameras);

	// Check if we failed (in OpenCV 3 we have better options to check this)
	if( isnan( cameras[0].R.at<float>(0,0)))
		return Status::ERR_CAMERA_PARAMS_ADJUST_FAIL;

	features.clear();
	pairwise_matches.clear();

#if DEBUG_CAMERA
	for (size_t i = 0; i < cameras.size(); ++i)
		cout << "Camera #" << i+1 << ":\n" << cameras[i].K() << "\n" << cameras[i].R << endl;
#endif
#if DEBUG
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Adjustor" << endl;
	t = getTickCount();
#endif


	// Find median focal length
	vector<double> focals( cameras.size());
	for (size_t i = 0; i < cameras.size(); ++i)
		focals[i] = cameras[i].focal;

	sort(focals.begin(), focals.end());
	double warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = focals[focals.size() / 2];
	else
		warped_image_scale = (focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	// Prepare variables used
	vector<Point> corners       ( input.size());
	vector<Mat>   masks_warped  ( input.size());
	vector<Mat>   images_warped ( input.size());
	vector<Size>  sizes         ( input.size());

	// Warp images and their masks to the adjuster found cameras
	Ptr<WarperCreator>  warper_creator = new cv::PlaneWarper();
	Ptr<RotationWarper> warper         = warper_creator->create( warped_image_scale * factor_feat_seam);

	for (size_t i = 0; i < input.size(); ++i)
	{
		Mat mask, temp;
		resize( input[i], temp, Size(), factor_img_seam, factor_img_seam);

		if( input_masks.size() == 0) {
			mask.create( temp.size(), CV_8U);
			mask.setTo( Scalar::all(255));
		}
		else
			resize( input_masks[i], mask, temp.size());

		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = factor_feat_seam;
		K(0,0) *= swa; K(0,2) *= swa;
		K(1,1) *= swa; K(1,2) *= swa;

		corners[i] = warper->warp(temp, K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes  [i] = images_warped[i].size();

		warper->warp(mask, K, cameras[i].R, INTER_LINEAR, BORDER_CONSTANT, masks_warped[i]);
	}

	// Feed the images to the gain exposure compensator
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(exposure_type_);
	compensator->feed(corners, images_warped, masks_warped);

	// Convert to floats
	vector<Mat> images_warped_f(input.size());
	for (size_t i = 0; i < input.size(); ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	images_warped.clear();

#if DEBUG
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t warping images/masks, gain-feed" << endl;
	t = getTickCount();
#endif


	// Find seam between the images
	seam_finder_->find( images_warped_f, corners, masks_warped);
	images_warped_f.clear();

#if DEBUG
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Seam finder" << endl;
	t = getTickCount();
#endif


	// Compute relative scales and update them
	double compose_work_aspect = comp_factor/factor_img_feat;
	warped_image_scale        *= compose_work_aspect;
	warper                     = warper_creator->create( warped_image_scale);

	// Update corners and sizes
	for (size_t i = 0; i < input.size(); ++i)
	{
		// Update intrinsics
		cameras[i].focal *= compose_work_aspect;
		cameras[i].ppx   *= compose_work_aspect;
		cameras[i].ppy   *= compose_work_aspect;

		// Update corner and size
		Size sz = full_img_sizes[i];
		if( comp_res_ != Stitcher::ORIGINAL_RES)
		{
			sz.width  = cvRound(full_img_sizes[i].width  * comp_factor);
			sz.height = cvRound(full_img_sizes[i].height * comp_factor);
		}

		Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		Rect roi = warper->warpRoi(sz, K, cameras[i].R);
		corners[i] = roi.tl();
		sizes[i] = roi.size();
	}

	// Setup blender
	Size dst_sz = resultRoi(corners, sizes).size();
	Ptr<Blender> blender = Blender::createDefault(blend_type_, false);

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

	// Prepare the blender
	blender->prepare(corners, sizes);

	Mat img_warped, img_warped_s, dilated_mask, seam_mask, mask_warped;

	// Add all images to the blender with fully warped img/masks and found cornerns
	for (size_t i = 0; i < input.size(); ++i)
	{
		Mat K, temp;
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
		Mat mask;
		if( input_masks.size() == 0) {
			mask.create( temp.size(), CV_8U);
			mask.setTo( Scalar::all(255));
		}
		else {
			resize( input_masks[i], mask, temp.size());
			input_masks[i].release();
		}

		warper->warp(mask, K, cameras[i].R, INTER_LINEAR, BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		compensator->apply(i, corners[i], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();

		// Combine masks
		dilate(masks_warped[i], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());

		bitwise_and(seam_mask, mask_warped, mask_warped);

		// Blend the current image
		blender->feed(img_warped_s, mask_warped, corners[i]);
	}

	// Save whether we have masks as input
	bool have_masks = (input_masks.size() != 0);

	// Clear input and blend
	input.clear();
	input_masks.clear();
	blender->blend( result, result_mask);


	// If we have input_masks, there is a chance the end result has too large black areas
	if( have_masks)
	{
		Mat mask_copy = result_mask.clone();
		vector<vector<Point>> v;

		// Find contours
		findContours( mask_copy, v, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
		mask_copy.release();

		// Get the one with the largest area
		unsigned int area = 0;
		unsigned int id;
		for(unsigned int i = 0; i < v.size(); ++i) {
			if( area < v[i].size()) {
				area = v[i].size();
				id = i;
			}
		}

		// Get bounding rect, then ROI
		Rect rect   = boundingRect( v[id]);
		rect        = Rect( rect.x -1, rect.y -1, rect.width + 2, rect.height + 2);
		result      = result      ( rect);
		result_mask = result_mask ( rect);
	}

#if DEBUG
	cout << "Time: " << ((getTickCount() - t) / getTickFrequency()) << " sec,\t Compositing" << endl;
	cout << "Finished! total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" << endl;
#endif

	return Status::OK;
}
