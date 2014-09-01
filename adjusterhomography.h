#ifndef ADJUSTERHOMOGRAPHY_H
#define ADJUSTERHOMOGRAPHY_H

#include <vector>
#include <string>

#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"

#include <opencv2/stitching/detail/motion_estimators.hpp>

class AdjusterHomography : public cv::detail::BundleAdjusterBase
{
public:
	AdjusterHomography() : BundleAdjusterBase(0, 0) {}

protected:
	void estimate( const std::vector<cv::detail::ImageFeatures> &features,
				   const std::vector<cv::detail::MatchesInfo> &pairwise_matches,
				   std::vector<cv::detail::CameraParams> &cameras);

	void setUpInitialCameraParams(const std::vector<cv::detail::CameraParams> &cameras){}
	void obtainRefinedCameraParams(std::vector<cv::detail::CameraParams> &cameras)const{}
	void calcError(cv::Mat &err){}
	void calcJacobian(cv::Mat &jac){}
};

#endif // ADJUSTERHOMOGRAPHY_H
