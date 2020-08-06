// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)
// Modified by: Ivan Eichhardt (ivan-dot-eichhardt-at-sztaki-dot-hu)
// see: https://github.com/colmap/colmap

#pragma once

#include <Eigen/Eigen>

#include "Options.hpp"
#include "Features.hpp"
#include "Descriptors.hpp"

namespace VlFeatExtraction {

  typedef struct {
    int point2D_idx1;
    int point2D_idx2;
    double score;
  } FeatureMatch;

  typedef std::vector<FeatureMatch> FeatureMatches;

  Eigen::MatrixXi ComputeSiftDistanceMatrix(
    const FeatureKeypoints* keypoints1, const FeatureKeypoints* keypoints2,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    const std::function<bool(float, float, float, float)>& guided_filter);

  size_t FindBestMatchesOneWayBruteForce(const Eigen::MatrixXi& dists,
    const float max_ratio, const float max_distance,
    std::vector<int>* matches);
  void FindBestMatchesBruteForce(const Eigen::MatrixXi& dists, const float max_ratio,
    const float max_distance, const bool cross_check,
    FeatureMatches* matches);
  void MatchSiftFeaturesCPUBruteForce(const SiftMatchingOptions& match_options,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    FeatureMatches* matches);

  void FindBestMatchesOneWayFLANN(
    const FeatureDescriptors& query, const FeatureDescriptors& database,
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
    indices,
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
    distances);
  size_t FindBestMatchesOneWayFLANN(
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
    indices,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
    distances,
    const float max_ratio, const float max_distance,
    std::vector<int>* matches);
  void FindBestMatchesFLANN(
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
    indices_1to2,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
    distances_1to2,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
    indices_2to1,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
    distances_2to1,
    const float max_ratio, const float max_distance, const bool cross_check,
    FeatureMatches* matches);
  void MatchSiftFeaturesCPUFLANN(const SiftMatchingOptions& match_options,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    FeatureMatches* matches);

  // Uses FLANN
  void MatchSiftFeaturesCPU(const SiftMatchingOptions& match_options,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    FeatureMatches* matches,
    const bool sort_matches_by_score = true);

  /*void MatchGuidedSiftFeaturesCPU(const SiftMatchingOptions& match_options,
    const FeatureKeypoints& keypoints1,
    const FeatureKeypoints& keypoints2,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    TwoViewGeometry* two_view_geometry);*/

}