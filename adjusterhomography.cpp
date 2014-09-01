#include "adjusterhomography.h"

#include <iostream>
#include <fstream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <vector>

#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

/*
//-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
	//-- Get the keypoints from the good matches
	obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
	scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, CV_RANSAC );

  */

void AdjusterHomography::estimate( const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches,
				std::vector<CameraParams> &cameras)
{
	num_images_ = features.size();
	vector<vector<Mat>> Homographies( num_images_);

	/*for (int i = 0; i < pairwise_matches.size(); i++) {
		MatchesInfo inf = pairwise_matches[i];
		if(inf.matches.size())
			cout << i << ", " << inf.matches.size() << endl;
	}*/

	// Matches in points
	std::vector<Point2f> point1;
	std::vector<Point2f> point2;

	// Calculate homographies
	for(int j = 0; j < num_images_; j++) {
		for(int i = j+1; i < num_images_; i++) {
			if(pairwise_matches[j*num_images_+i].matches.size())
			{
				//
				const MatchesInfo inf = pairwise_matches[j*num_images_+i];
				point1.reserve( inf.matches.size());
				point2.reserve( inf.matches.size());

				for(int k = 0; k < inf.matches.size(); k++) {
					point1.push_back( features[j].keypoints[ inf.matches[k].queryIdx ].pt );
					point2.push_back( features[i].keypoints[ inf.matches[k].trainIdx ].pt );
				}

				Mat H = findHomography( point1, point2, CV_RANSAC );

				Homographies[j].push_back( H);
				point1.clear();
				point2.clear();
			}
		}
	}

	// Cleanup
	point1.clear();
	point2.clear();

	// TODO: Now we have individual homographies but we should multiply them to have global cameras
	// Therefor: Exit, as we do not know how the rest of the stack will behave

	exit(0);

}
