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

#include "Descriptors.hpp"
#include <array>

template <typename T1, typename T2>
T2 TruncateCast(const T1 value) {
  return std::min(
    static_cast<T1>(std::numeric_limits<T2>::max()),
    std::max(static_cast<T1>(std::numeric_limits<T2>::min()), value));
}

Eigen::MatrixXf VlFeatExtraction::L2NormalizeFeatureDescriptors(const Eigen::MatrixXf& descriptors) {
  return descriptors.rowwise().normalized();
}

Eigen::MatrixXf VlFeatExtraction::L1RootNormalizeFeatureDescriptors(const Eigen::MatrixXf& descriptors) {
  Eigen::MatrixXf descriptors_normalized(descriptors.rows(),
    descriptors.cols());
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    const float norm = descriptors.row(r).lpNorm<1>();
    descriptors_normalized.row(r) = descriptors.row(r) / norm;
    descriptors_normalized.row(r) =
      descriptors_normalized.row(r).array().sqrt();
  }
  return descriptors_normalized;
}

VlFeatExtraction::FeatureDescriptors VlFeatExtraction::FeatureDescriptorsToUnsignedByte(const Eigen::MatrixXf& descriptors) {
  FeatureDescriptors descriptors_unsigned_byte(descriptors.rows(),
    descriptors.cols());
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::MatrixXf::Index c = 0; c < descriptors.cols(); ++c) {
      const float scaled_value = std::round(512.0f * descriptors(r, c));
      descriptors_unsigned_byte(r, c) =
        TruncateCast<float, uint8_t>(scaled_value);
    }
  }
  return descriptors_unsigned_byte;
}

VlFeatExtraction::FeatureDescriptors VlFeatExtraction::TransformVLFeatToUBCFeatureDescriptors(const FeatureDescriptors& vlfeat_descriptors) {
  FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(),
    vlfeat_descriptors.cols());
  const std::array<int, 8> q{ { 0, 7, 6, 5, 4, 3, 2, 1 } };
  for (FeatureDescriptors::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k) {
          ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
            vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
        }
      }
    }
  }
  return ubc_descriptors;
}
