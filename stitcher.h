#ifndef STITCHER_H
#define STITCHER_H

#include <vector>
#include <string>

#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"


enum class Status
{
	OK = 0,
	ERR_NEED_MORE_IMGS = -1,
	ERR_HOMOGRAPHY_EST_FAIL = -2,
	ERR_CAMERA_PARAMS_ADJUST_FAIL = -3,
};

/**
 * @brief The Stitcher class
 *
 *
 */
class Stitcher
{
public:
	enum { ORIGINAL_RES = -1 };

	/**
	 * @brief Stitcher Constructor, preloads the stitcher with the standard options
	 */
	Stitcher();


	/**
	  * Getter/setter functions for all stitch options
	  */
	double feat_res() const { return feat_res_; }
	void set_feat_res(double feat_res) { feat_res_ = feat_res; }

	double seam_res() const { return seam_res_; }
	void set_seam_res(double seam_res) { seam_res_ = seam_res; }

	double comp_res() const { return comp_res_; }
	void set_comp_res(double comp_res) { comp_res_ = comp_res; }

	cv::Ptr<cv::detail::FeaturesFinder> feature_finder() const { return feature_finder_;}
	void set_feature_finder( cv::Ptr<cv::detail::FeaturesFinder> feature_finder){ feature_finder_ = feature_finder; }

	const cv::Mat& matching_mask() const { return matching_mask_; }
	void set_matching_mask(const cv::Mat &mask)
	{
		CV_Assert(mask.type() == CV_8U && mask.cols == mask.rows);
		matching_mask_ = mask.clone();
	}

	float conf_featurematching() const { return conf_featurematching_; }
	void set_conf_featurematching(float conf_featurematching) { conf_featurematching_ = conf_featurematching; }

	float conf_adjustor() const { return conf_adjustor_; }
	void set_conf_adjustor(float conf_adjustor) { conf_adjustor_ = conf_adjustor; }

	cv::Ptr<cv::detail::BundleAdjusterRay> adjuster() const { return adjuster_; }
	void set_adjuster( cv::Ptr<cv::detail::BundleAdjusterRay> adjuster) { adjuster_ = adjuster; }

	std::string adjuster_mask() const { return adjuster_mask_; }
	void set_adjuster_mask( std::string adjuster_mask) { adjuster_mask_ = adjuster_mask; }

	int exposure_type() const { return exposure_type_; }
	void set_exposure_type( int exposure_type) { exposure_type_ = exposure_type; }

	cv::Ptr<cv::detail::GraphCutSeamFinder> seam_finder() const { return seam_finder_; }
	void set_seam_finder( cv::Ptr<cv::detail::GraphCutSeamFinder> seam_finder) { seam_finder_ = seam_finder; }

	int blend_type() const { return blend_type_; }
	void set_blend_type( int blend_type){ blend_type_ = blend_type; }

	float blend_strength() const { return blend_strength_; }
	void set_blend_strength( float blend_strength) { blend_strength_ = blend_strength; }


	/**
	 * @brief stitch
	 * @param input   preloaded images who are about to get stitched
	 * @param result  name of the file the result should be saved to. Don't add extensions
	 * @param imgSize Size of the original photos
	 */
	Status stitch( std::vector<cv::Mat> &input,
				 cv::Mat  &result,
				 cv::Mat  &result_mask,
				 cv::Size &imgSize)
		{ std::vector<cv::Mat> vec; return stitch( input, vec, result, result_mask, imgSize); }

	/**
	 * @brief stitch
	 * @param input       preloaded images who are about to get stitched
	 * @param input_masks Additional masks for input
	 * @param result      name of the file the result should be saved to. Don't add extensions
	 * @param imgSize     Size of the original/non-stitched photos
	 */
	Status stitch( std::vector<cv::Mat> &input,
				 std::vector<cv::Mat> &input_masks,
				 cv::Mat &result,
				 cv::Mat &result_mask,
				 cv::Size &imgSize);

protected:
	/***************************
	 * Options of the stitcher *
	 ***************************/
	// Resolutions used for: feature matching, seam finding, compositioning
	double feat_res_;
	double seam_res_;
	double comp_res_;

	// Options: SurfFeaturesFinder or OrbFeaturesFinder
	cv::Ptr<cv::detail::FeaturesFinder> feature_finder_;

	// Matrix who tells which image to compare to which
	cv::Mat matching_mask_;

	// Confidences
	float conf_featurematching_;
	float conf_adjustor_;

	// options: BundleAdjusterReproj, BundleAdjusterRay
	cv::Ptr<cv::detail::BundleAdjusterRay> adjuster_;
	std::string                            adjuster_mask_;

	// Exposure type
	int exposure_type_ = cv::detail::ExposureCompensator::GAIN;

	// Seamfinder algorithm. Options:
	//		NoSeamFinder, VoronoiSeamFinder, GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR),
	//		GraphCutSeamFinder, (GraphCutSeamFinderBase::COST_COLOR_GRAD), DpSeamFinder(DpSeamFinder::COLOR),
	//		DpSeamFinder(DpSeamFinder::COLOR_GRAD)
	cv::Ptr<cv::detail::GraphCutSeamFinder> seam_finder_;

	// Options:
	//		Blender::NO, Blender::FEATHER, Blender::MULTI_BAND
	int   blend_type_ = cv::detail::Blender::MULTI_BAND;
	float blend_strength_ = 5;


	/*************************************
	 * Direct inputs by stitch functions *
	 *************************************/
	std::vector<cv::Mat> input_;
	std::vector<cv::Mat> input_masks_;
	std::string          result_name_;
	cv::Size             img_size_;


	/*************************************
	 * Direct inputs by stitch functions *
	 *************************************/
};

#endif // STITCHER_H
