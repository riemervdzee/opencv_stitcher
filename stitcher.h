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
	ERR_NEED_MORE_IMGS = 1,
	ERR_HOMOGRAPHY_EST_FAIL = 2,
	ERR_CAMERA_PARAMS_ADJUST_FAIL = 3,
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
	double img_res() const { return img_res_; }
	void set_img_res(double img_res) { img_res_ = img_res; }

	double feat_res() const { return feat_res_; }
	void set_feat_res(double feat_res) { feat_res_ = feat_res; }

	double seam_res() const { return seam_res_; }
	void set_seam_res(double seam_res) { seam_res_ = seam_res; }

	double comp_res() const { return comp_res_; }
	void set_comp_res(double comp_res) { comp_res_ = comp_res; }

	cv::Ptr<cv::detail::FeaturesFinder> feature_finder() const { return feature_finder_;}
	void set_feature_finder( cv::Ptr<cv::detail::FeaturesFinder> feature_finder){ feature_finder_ = feature_finder; }

	float conf_featurematching() const { return conf_featurematching_; }
	void set_conf_featurematching(float conf_featurematching) { conf_featurematching_ = conf_featurematching; }

	float conf_adjustor() const { return conf_adjustor_; }
	void set_conf_adjustor(float conf_adjustor) { conf_adjustor_ = conf_adjustor; }

	cv::Ptr<cv::detail::BundleAdjusterBase> adjuster() const { return adjuster_; }
	void set_adjuster( cv::Ptr<cv::detail::BundleAdjusterBase> adjuster) { adjuster_ = adjuster; }

	std::string adjuster_mask() const { return adjuster_mask_; }
	void set_adjuster_mask( std::string adjuster_mask) { adjuster_mask_ = adjuster_mask; }

	int exposure_type() const { return exposure_type_; }
	void set_exposure_type( int exposure_type) { exposure_type_ = exposure_type; }

	cv::Ptr<cv::detail::SeamFinder> seam_finder() const { return seam_finder_; }
	void set_seam_finder( cv::Ptr<cv::detail::SeamFinder> seam_finder) { seam_finder_ = seam_finder; }

	int blend_type() const { return blend_type_; }
	void set_blend_type( int blend_type){ blend_type_ = blend_type; }

	float blend_strength() const { return blend_strength_; }
	void set_blend_strength( float blend_strength) { blend_strength_ = blend_strength; }


	/**
	 * @brief stitch         Main function of the stitcher. Is able to stitch multiple images and restitch results.
	 *
	 * @param input          Preloaded images who are about to get stitched
	 * @param input_masks    Additional masks for input (pass an empty vector if none available)
	 * @param result         Reference where the result should be stored to
	 * @param result_mask    Reference where the resulting mask should be stored to
	 * @param matching_mask  Tells which image-index to match with which
	 * @param input_roi      Every image can have a ROI where features are attempted to be found
	 * @param cameras        Initial camera setup (no function atm)
	 *
	 * @return               Returns whether the stitcher was succesful or not
	 */
	Status stitch( std::vector<cv::Mat> &input,
				   std::vector<cv::Mat> &input_masks,
				   cv::Mat &result,
				   cv::Mat &result_mask,
				   const cv::Mat &matching_mask,
				   std::vector<std::vector<cv::Rect>> input_roi  = std::vector<std::vector<cv::Rect>>(),
				   std::vector<cv::detail::CameraParams> cameras = std::vector<cv::detail::CameraParams>());

protected:
	/***************************
	 * Options of the stitcher *
	 ***************************/
	// Resolutions used for: feature matching, seam finding, compositioning
	double img_res_;
	double feat_res_;
	double seam_res_;
	double comp_res_;

	// Options: SurfFeaturesFinder or OrbFeaturesFinder
	cv::Ptr<cv::detail::FeaturesFinder> feature_finder_;

	// Confidences
	float conf_featurematching_;
	float conf_adjustor_;

	// options: BundleAdjusterReproj, BundleAdjusterRay
	cv::Ptr<cv::detail::BundleAdjusterBase> adjuster_;
	std::string                             adjuster_mask_;

	// Exposure type
	int exposure_type_;

	// Seamfinder algorithm. Options:
	//		NoSeamFinder, VoronoiSeamFinder, GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR),
	//		GraphCutSeamFinder, (GraphCutSeamFinderBase::COST_COLOR_GRAD), DpSeamFinder(DpSeamFinder::COLOR),
	//		DpSeamFinder(DpSeamFinder::COLOR_GRAD)
	cv::Ptr<cv::detail::SeamFinder> seam_finder_;

	// Options:
	//		Blender::NO, Blender::FEATHER, Blender::MULTI_BAND
	int   blend_type_;
	float blend_strength_;


	/*************************************
	 * Direct inputs by stitch functions *
	 *************************************/
	std::vector<cv::Mat> input_;
	std::vector<cv::Mat> input_masks_;
	std::string          result_name_;
	cv::Size             img_size_;
};

#endif // STITCHER_H
