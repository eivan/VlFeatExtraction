#include <iostream>
#include <algorithm>
#include <chrono>

#include <Eigen/Eigen>

#include <opencv2/opencv.hpp>

#include <VlFeatExtraction/Extraction.hpp>
#include <VlFeatExtraction/Matching.hpp>

#include <VLFeat/covdet.h>
#include <VLFeat/sift.h>

using namespace VlFeatExtraction;

int main(int argc, char** argv) {

  if (argc != 3) {
    std::cerr << "Bad parameterization" << std::endl;
    exit(1);
  }

  std::string im_left = argv[1];
  std::string im_right = argv[2];

  FeatureKeypoints fkp_left, fkp_right;
  FeatureDescriptors dsc_left, dsc_right;

  SiftExtractionOptions options_f;
  SiftMatchingOptions options_m;

  options_f.estimate_affine_shape = true;
  options_f.domain_size_pooling = true;

  // Feature extraction from a path-to-an-image
  auto extract_by_filename = [](
    const std::string& filename,
    const SiftExtractionOptions& options,
    FeatureKeypoints* keypoints, 
    FeatureDescriptors* descriptors) {

      // Read Image
      cv::Mat im = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

      if (im.empty()) {
        std::cerr << "Failed to read image " << filename << std::endl;
        return false;
      }

      cv::Mat imF;
      im.convertTo(imF, CV_32F, 1.0 / 255.0);

      // Perform extraction
      return extract(imF.ptr<float>(), imF.cols, imF.rows, options, keypoints, descriptors);
  };

  bool success1 = extract_by_filename(im_left, options_f, &fkp_left, &dsc_left);
  bool success2 = extract_by_filename(im_right, options_f, &fkp_right, &dsc_right);

  if (success1 && success2) {

    FeatureMatches matches;

    // Perform matching
    auto t_start = std::chrono::high_resolution_clock::now();
    MatchSiftFeaturesCPU(options_m, dsc_left, dsc_right, &matches);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count() << " ";

    std::cout << fkp_left.size() << " " << fkp_right.size() << " " << matches.size() << std::endl;

    // Write matches to file

    {
      std::ofstream os("matches.txt");

      os.precision(std::numeric_limits<float>::max_digits10);
      //os << std::scientific;

      for (auto& match : matches) {
        const auto& [left_p, right_p] = std::make_pair(match.point2D_idx1, match.point2D_idx2);

        const auto& left = fkp_left[left_p];
        const auto& right = fkp_right[right_p];

        os
          << left.x << " " << left.y << " " << left.a11 << " " << left.a12 << " " << left.a21 << " " << left.a22 << " "
          << right.x << " " << right.y << " " << right.a11 << " " << right.a12 << " " << right.a21 << " " << right.a22
          << std::endl;
      }

      os.close();
    }

    // Display matches using OpenCV

    std::vector<cv::DMatch> matches_to_draw;
    std::vector< cv::KeyPoint > keypoints_Object, keypoints_Scene; // Keypoints

    auto convertToCV = [](std::vector< cv::KeyPoint >& keypointsCV, const FeatureKeypoints& keypoints) {
      keypointsCV.clear();
      keypointsCV.reserve(keypoints.size());
      for (const auto& kp : keypoints) {
        keypointsCV.emplace_back(kp.x, kp.y, kp.ComputeOrientation());
      }
    };

    convertToCV(keypoints_Object, fkp_right);
    convertToCV(keypoints_Scene, fkp_left);

    matches_to_draw.reserve(matches.size());

    // Iterate through the matches from descriptor
    for (unsigned int i = 0; i < matches.size(); i++)
    {
      cv::DMatch v;
      v.trainIdx = matches[i].point2D_idx1;
      v.queryIdx = matches[i].point2D_idx2;
      // This is for all matches
      matches_to_draw.push_back(v);
    }
    auto image = cv::imread(im_right);
    auto walls = cv::imread(im_left);

    cv::Mat output = cv::Mat::zeros(std::max(image.rows, walls.rows), image.cols + walls.cols, image.type());
    using namespace std;
    cv::drawMatches(image, keypoints_Object, walls, keypoints_Scene, matches_to_draw, output);// , cv::Scalar::all(-1), cv::Scalar::all(-1), vector<vector<char> >(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("OpenCV_view", output);
    cv::waitKey(0);
  }
  else {
    std::cerr << "failed to extract features" << std::endl;
  }

  return 0;
}