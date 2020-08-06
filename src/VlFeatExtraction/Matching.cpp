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

#include "Matching.hpp"
#include <nanoflann/include/nanoflann.hpp>
#include <array>
#include <iostream>

Eigen::MatrixXi VlFeatExtraction::ComputeSiftDistanceMatrix(
  const FeatureKeypoints* keypoints1, 
  const FeatureKeypoints* keypoints2, 
  const FeatureDescriptors& descriptors1, 
  const FeatureDescriptors& descriptors2,
  const std::function<bool(float, float, float, float)>& guided_filter) {
  if (guided_filter != nullptr) {
    assert(keypoints1);
    assert(keypoints2);
    assert(keypoints1->size() == descriptors1.rows());
    assert(keypoints2->size() == descriptors2.rows());
  }


  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(
    descriptors1.rows(), descriptors2.rows());

  if (guided_filter != nullptr) {
    // TODO: should this be removed and in stead used in dot product?
    const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors1_int =
      descriptors1.cast<int>();
    const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors2_int =
      descriptors2.cast<int>();

    for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
      for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
        if (guided_filter((*keypoints1)[i1].x, (*keypoints1)[i1].y,
            (*keypoints2)[i2].x, (*keypoints2)[i2].y)) {
          dists(i1, i2) = 0;
        }
        else {
          dists(i1, i2) = descriptors1_int.row(i1).dot(descriptors2_int.row(i2));
        }
      }
    }
  }
  else {
    for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
      for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
        //dists(i1, i2) = descriptors1_int.row(i1).dot(descriptors2_int.row(i2));
      }
    }
    dists = descriptors1.cast<int>() * descriptors2.cast<int>().transpose();
  }

  return dists;
}

size_t VlFeatExtraction::FindBestMatchesOneWayBruteForce(const Eigen::MatrixXi& dists, const float max_ratio, const float max_distance, std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
  const float kDistNorm = 1.0f / (512.0f * 512.0f);

  size_t num_matches = 0;
  matches->resize(dists.rows(), -1);

  for (Eigen::Index i1 = 0; i1 < dists.rows(); ++i1) {
    int best_i2 = -1;
    int best_dist = 0;
    int second_best_dist = 0;
    for (Eigen::Index i2 = 0; i2 < dists.cols(); ++i2) {
      const int dist = dists(i1, i2);
      if (dist > best_dist) {
        best_i2 = i2;
        second_best_dist = best_dist;
        best_dist = dist;
      }
      else if (dist > second_best_dist) {
        second_best_dist = dist;
      }
    }

    // Check if any match found.
    if (best_i2 == -1) {
      continue;
    }

    const float best_dist_normed =
      std::acos(std::min(kDistNorm * best_dist, 1.0f));

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    const float second_best_dist_normed =
      std::acos(std::min(kDistNorm * second_best_dist, 1.0f));

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    num_matches += 1;
    (*matches)[i1] = best_i2;
  }

  return num_matches;
}

void VlFeatExtraction::FindBestMatchesBruteForce(const Eigen::MatrixXi& dists, const float max_ratio, const float max_distance, const bool cross_check, FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches12;
  const size_t num_matches12 = FindBestMatchesOneWayBruteForce(
    dists, max_ratio, max_distance, &matches12);

  if (cross_check) {
    std::vector<int> matches21;
    const size_t num_matches21 = FindBestMatchesOneWayBruteForce(
      dists.transpose(), max_ratio, max_distance, &matches21);
    matches->reserve(std::min(num_matches12, num_matches21));
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
        matches21[matches12[i1]] == static_cast<int>(i1)) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
  else {
    matches->reserve(num_matches12);
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
}

void VlFeatExtraction::MatchSiftFeaturesCPUBruteForce(
  const SiftMatchingOptions& match_options,
  const FeatureDescriptors& descriptors1,
  const FeatureDescriptors& descriptors2,
  FeatureMatches* matches) {
  //CHECK(match_options.Check());
  //CHECK_NOTNULL(matches);

  const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
    nullptr, nullptr, descriptors1, descriptors2, nullptr);

  FindBestMatchesBruteForce(dists, match_options.max_ratio, match_options.max_distance,
    match_options.cross_check, matches);
}

// nanoflann otherwise would try to compute difference on the unsigned type
struct metric_L2_intDIST : public nanoflann::Metric {
  template <class T, class DataSource> struct traits {
    typedef nanoflann::L2_Adaptor<T, DataSource, int> distance_t;
  };
};

void VlFeatExtraction::FindBestMatchesOneWayFLANN(
  const FeatureDescriptors& query, const FeatureDescriptors& database,
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
  indices,
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
  distances) {

  std::cout << "rows: " << query.rows() << std::endl;
  std::cout << "dims: " << query.cols() << std::endl;

  constexpr size_t kNumNearestNeighbors = 2;
  constexpr size_t kNumTreesInForest = 4;

  indices->resize(query.rows(), std::min(kNumNearestNeighbors,
    static_cast<size_t>(database.rows())));
  distances->resize(
    query.rows(),
    std::min(kNumNearestNeighbors, static_cast<size_t>(database.rows())));

  using namespace nanoflann;

  typedef KDTreeEigenMatrixAdaptor<FeatureDescriptors, -1, metric_L2_intDIST>
    my_kd_tree_t;

  my_kd_tree_t mat_index(
    database.cols(), std::cref(database),
    kNumTreesInForest /*TODO: max 10 leaves? */);
  mat_index.index->buildIndex();

  std::array<size_t, kNumNearestNeighbors> ret_indexes;
  std::array<int, kNumNearestNeighbors> out_dists_sqr;

  nanoflann::KNNResultSet<int> result_set(kNumNearestNeighbors);

  for (Eigen::Index query_index = 0; query_index < indices->rows(); ++query_index) {

    result_set.init(&ret_indexes[0], &out_dists_sqr[0]);
    mat_index.index->findNeighbors(result_set, query.row(query_index).data(), nanoflann::SearchParams(10));

    for (Eigen::Index k = 0; k < result_set.size(); ++k) {
      const Eigen::Index database_index = ret_indexes[k];

      indices->coeffRef(query_index, k)
        = database_index;

      distances->coeffRef(query_index, k)
        = query.row(query_index).cast<int>()
        .dot(database.row(database_index).cast<int>());
    }
  }
}

size_t VlFeatExtraction::FindBestMatchesOneWayFLANN(
  const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  indices,
  const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  distances,
  const float max_ratio, const float max_distance,
  std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
  const float kDistNorm = 1.0f / (512.0f * 512.0f);

  size_t num_matches = 0;
  matches->resize(indices.rows(), -1);

  for (int d1_idx = 0; d1_idx < indices.rows(); ++d1_idx) {
    int best_i2 = -1;
    int best_dist = 0;
    int second_best_dist = 0;
    for (int n_idx = 0; n_idx < indices.cols(); ++n_idx) {
      const int d2_idx = indices(d1_idx, n_idx);
      const int dist = distances(d1_idx, n_idx);
      if (dist > best_dist) {
        best_i2 = d2_idx;
        second_best_dist = best_dist;
        best_dist = dist;
      }
      else if (dist > second_best_dist) {
        second_best_dist = dist;
      }
    }

    // Check if any match found.
    if (best_i2 == -1) {
      continue;
    }

    const float best_dist_normed =
      std::acos(std::min(kDistNorm * best_dist, 1.0f));

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    const float second_best_dist_normed =
      std::acos(std::min(kDistNorm * second_best_dist, 1.0f));

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    num_matches += 1;
    (*matches)[d1_idx] = best_i2;
  }

  return num_matches;
}

void VlFeatExtraction::FindBestMatchesFLANN(
  const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  indices_1to2,
  const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  distances_1to2,
  const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  indices_2to1,
  const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  distances_2to1,
  const float max_ratio, const float max_distance, const bool cross_check,
  FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches12;
  const size_t num_matches12 = FindBestMatchesOneWayFLANN(
    indices_1to2, distances_1to2, max_ratio, max_distance, &matches12);

  if (cross_check && indices_2to1.rows()) {
    std::vector<int> matches21;
    const size_t num_matches21 = FindBestMatchesOneWayFLANN(
      indices_2to1, distances_2to1, max_ratio, max_distance, &matches21);
    matches->reserve(std::min(num_matches12, num_matches21));
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
        matches21[matches12[i1]] == static_cast<int>(i1)) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
  else {
    matches->reserve(num_matches12);
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
}

void VlFeatExtraction::MatchSiftFeaturesCPUFLANN(const SiftMatchingOptions& match_options,
  const FeatureDescriptors& descriptors1,
  const FeatureDescriptors& descriptors2,
  FeatureMatches* matches) {
  //CHECK(match_options.Check());
  //CHECK_NOTNULL(matches);

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    indices_1to2;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    distances_1to2;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    indices_2to1;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    distances_2to1;

  FindBestMatchesOneWayFLANN(descriptors1, descriptors2, &indices_1to2,
    &distances_1to2);
  if (match_options.cross_check) {
    FindBestMatchesOneWayFLANN(descriptors2, descriptors1, &indices_2to1,
      &distances_2to1);
  }

  FindBestMatchesFLANN(indices_1to2, distances_1to2, indices_2to1,
    distances_2to1, match_options.max_ratio,
    match_options.max_distance, match_options.cross_check,
    matches);
}

void VlFeatExtraction::MatchSiftFeaturesCPU(
  const SiftMatchingOptions& match_options,
  const FeatureDescriptors& descriptors1,
  const FeatureDescriptors& descriptors2,
  FeatureMatches* matches,
  const bool sort_matches_by_score) {

  return MatchSiftFeaturesCPUFLANN(match_options, descriptors1, descriptors2, matches);

  // optionally, sort matches by matching score (useful for PROSAC and the like)
  if (sort_matches_by_score) {
    std::sort(
      matches->begin(), matches->end(),
      [](const FeatureMatch& match1, const FeatureMatch& match2) {
        return match1.score > match2.score;
      });
  }
}

void VlFeatExtraction::MatchGuidedSiftFeaturesCPU(const SiftMatchingOptions& match_options,
  const FeatureKeypoints& keypoints1,
  const FeatureKeypoints& keypoints2,
  const FeatureDescriptors& descriptors1,
  const FeatureDescriptors& descriptors2,
  const std::function<bool(float, float, float, float)>& guided_filter,
  FeatureMatches* matches) {
  //CHECK(match_options.Check());
  //CHECK(guided_filter);

  const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
    &keypoints1, &keypoints2, descriptors1, descriptors2, guided_filter);

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    indices_1to2(dists.rows(), dists.cols());
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    indices_2to1(dists.cols(), dists.rows());
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    distances_1to2 = dists;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    distances_2to1 = dists.transpose();

  for (int i = 0; i < indices_1to2.rows(); ++i) {
    indices_1to2.row(i) = Eigen::VectorXi::LinSpaced(indices_1to2.cols(), 0,
      indices_1to2.cols() - 1);
  }
  for (int i = 0; i < indices_2to1.rows(); ++i) {
    indices_2to1.row(i) = Eigen::VectorXi::LinSpaced(indices_2to1.cols(), 0,
      indices_2to1.cols() - 1);
  }

  FindBestMatchesFLANN(indices_1to2, distances_1to2, indices_2to1,
    distances_2to1, match_options.max_ratio,
    match_options.max_distance, match_options.cross_check,
    matches);
}