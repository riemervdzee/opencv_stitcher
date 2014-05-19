#include "bundleadjusterray2.h"

#include <omp.h>
//#include "precomp.hpp"
#include "opencv2/opencv.hpp"
//#include "opencv2/core/cvdef.h"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::detail;

void calcDeriv(const Mat &err1, const Mat &err2, const double h, Mat res)
{
	for (int i = 0; i < err1.rows; ++i)
		res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
}

//////////////////////////////////////////////////////////////////////////////

void BundleAdjusterRay2::setUpInitialCameraParams(const std::vector<CameraParams> &cameras)
{
	cam_params_.create(num_images_ * 4, 1, CV_64F);
	SVD svd;
	for (int i = 0; i < num_images_; ++i)
	{
		cam_params_.at<double>(i * 4, 0) = cameras[i].focal;

		svd(cameras[i].R, SVD::FULL_UV);
		Mat R = svd.u * svd.vt;
		if (determinant(R) < 0)
			R *= -1;

		Mat rvec;
		Rodrigues(R, rvec);
		CV_Assert(rvec.type() == CV_32F);
		cam_params_.at<double>(i * 4 + 1, 0) = rvec.at<float>(0, 0);
		cam_params_.at<double>(i * 4 + 2, 0) = rvec.at<float>(1, 0);
		cam_params_.at<double>(i * 4 + 3, 0) = rvec.at<float>(2, 0);
	}
}


void BundleAdjusterRay2::obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const
{
	for (int i = 0; i < num_images_; ++i)
	{
		cameras[i].focal = cam_params_.at<double>(i * 4, 0);

		Mat rvec(3, 1, CV_64F);
		rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
		Rodrigues(rvec, cameras[i].R);

		Mat tmp;
		cameras[i].R.convertTo(tmp, CV_32F);
		cameras[i].R = tmp;
	}
}


void BundleAdjusterRay2::calcError(Mat &err)
{
	err.create(total_num_matches_ * 3, 1, CV_64F);

	int match_idx = 0;

	for (size_t edge_idx = 0; edge_idx < edges_.size(); ++edge_idx)
	{
		int i = edges_[edge_idx].first;
		int j = edges_[edge_idx].second;
		double f1 = cam_params_.at<double>(i * 4, 0);
		double f2 = cam_params_.at<double>(j * 4, 0);

		double R1[9];
		Mat R1_(3, 3, CV_64F, R1);
		Mat rvec(3, 1, CV_64F);
		rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
		Rodrigues(rvec, R1_);

		double R2[9];
		Mat R2_(3, 3, CV_64F, R2);
		rvec.at<double>(0, 0) = cam_params_.at<double>(j * 4 + 1, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(j * 4 + 2, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(j * 4 + 3, 0);
		Rodrigues(rvec, R2_);

		const ImageFeatures& features1 = features_[i];
		const ImageFeatures& features2 = features_[j];
		const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];

		Mat_<double> K1 = Mat::eye(3, 3, CV_64F);
		K1(0,0) = f1; K1(0,2) = features1.img_size.width * 0.5;
		K1(1,1) = f1; K1(1,2) = features1.img_size.height * 0.5;

		Mat_<double> K2 = Mat::eye(3, 3, CV_64F);
		K2(0,0) = f2; K2(0,2) = features2.img_size.width * 0.5;
		K2(1,1) = f2; K2(1,2) = features2.img_size.height * 0.5;

		Mat_<double> H1 = R1_ * K1.inv();
		Mat_<double> H2 = R2_ * K2.inv();

		for (size_t k = 0; k < matches_info.matches.size(); ++k)
		{
			if (!matches_info.inliers_mask[k])
				continue;

			const DMatch& m = matches_info.matches[k];

			Point2f p1 = features1.keypoints[m.queryIdx].pt;
			double x1 = H1(0,0)*p1.x + H1(0,1)*p1.y + H1(0,2);
			double y1 = H1(1,0)*p1.x + H1(1,1)*p1.y + H1(1,2);
			double z1 = H1(2,0)*p1.x + H1(2,1)*p1.y + H1(2,2);
			double len = std::sqrt(x1*x1 + y1*y1 + z1*z1);
			x1 /= len; y1 /= len; z1 /= len;

			Point2f p2 = features2.keypoints[m.trainIdx].pt;
			double x2 = H2(0,0)*p2.x + H2(0,1)*p2.y + H2(0,2);
			double y2 = H2(1,0)*p2.x + H2(1,1)*p2.y + H2(1,2);
			double z2 = H2(2,0)*p2.x + H2(2,1)*p2.y + H2(2,2);
			len = std::sqrt(x2*x2 + y2*y2 + z2*z2);
			x2 /= len; y2 /= len; z2 /= len;

			double mult = std::sqrt(f1 * f2);
			err.at<double>(3 * match_idx, 0) = mult * (x1 - x2);
			err.at<double>(3 * match_idx + 1, 0) = mult * (y1 - y2);
			err.at<double>(3 * match_idx + 2, 0) = mult * (z1 - z2);

			match_idx++;
		}
	}
}


void BundleAdjusterRay2::calcJacobian(Mat &jac)
{
	jac.create(total_num_matches_ * 3, num_images_ * 4, CV_64F);

	double val;
	const double step = 1e-3;

	for (int i = 0; i < num_images_; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			val = cam_params_.at<double>(i * 4 + j, 0);
			cam_params_.at<double>(i * 4 + j, 0) = val - step;
			calcError(err1_);
			cam_params_.at<double>(i * 4 + j, 0) = val + step;
			calcError(err2_);
			calcDeriv(err1_, err2_, 2 * step, jac.col(i * 4 + j));
			cam_params_.at<double>(i * 4 + j, 0) = val;
		}
	}
}
