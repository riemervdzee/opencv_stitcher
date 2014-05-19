#ifndef BUNDLEADJUSTERRAY2_H
#define BUNDLEADJUSTERRAY2_H

#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"

namespace cv {
namespace detail {

class BundleAdjusterRay2 : public BundleAdjusterBase
{
public:
	BundleAdjusterRay2() : BundleAdjusterBase(4, 3) {}

private:
	void setUpInitialCameraParams(const std::vector<CameraParams> &cameras);
	void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const;
	void calcError(Mat &err);
	void calcJacobian(Mat &jac);

	Mat err1_, err2_;
};

}
}

#endif // BUNDLEADJUSTERRAY2_H
