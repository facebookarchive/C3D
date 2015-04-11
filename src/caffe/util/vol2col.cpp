/*
 *
 *  Copyright (c) 2015, Facebook, Inc. All rights reserved.
 *
 *  Licensed under the Creative Commons Attribution-NonCommercial 3.0
 *  License (the "License"). You may obtain a copy of the License at
 *  https://creativecommons.org/licenses/by-nc/3.0/.
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *
 */

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/vol2col.hpp"

namespace caffe {

template <typename Dtype>
void vol2col_cpu(const Dtype* data_im, const int channels, const int length,
	    const int height, const int width, const int ksize, const int kdepth, const int pad,
	    const int temporal_pad, const int stride, const int temporal_stride, Dtype* data_col) {
  int length_col = (length + 2 * temporal_pad - kdepth) / temporal_stride + 1;
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;

  int channels_col = channels * kdepth * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int l_offset = (c / ksize / ksize) % kdepth;
    int c_im = c / ksize / ksize / kdepth;
    for (int l=0; l < length_col; ++l) {
      for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
          int l_pad = l * temporal_stride - temporal_pad + l_offset;
          int h_pad = h * stride - pad + h_offset;
          int w_pad = w * stride - pad + w_offset;

          if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
        		  && l_pad >=0 && l_pad < length)
            data_col[((c * length_col + l) * height_col + h) * width_col + w] =
              data_im[((c_im * length + l_pad) * height + h_pad) * width + w_pad];
          else
            data_col[((c * length_col + l) * height_col + h) * width_col + w] = 0;
        }
      }
    }
  }
}

// Explicit instantiation
template void vol2col_cpu<float>(const float* data_im, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, float* data_col);
template void vol2col_cpu<double>(const double* data_im, const int channels,const int length,
	    const int height, const int width, const int ksize, const int kdepth, const int pad,
	    const int temporal_pad, const int stride, const int temporal_stride, double* data_col);

template <typename Dtype>
void col2vol_cpu(const Dtype* data_col, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, Dtype* data_im) {
  memset(data_im, 0, sizeof(Dtype) * length * height * width * channels);
  int length_col = (length + 2* temporal_pad - kdepth) / temporal_stride + 1;
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int l_offset = (c / ksize / ksize) % kdepth;
    int c_im = c / ksize / ksize / kdepth;
    for (int l=0; l < length_col; ++l) {
      for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
          int l_pad = l * temporal_stride - temporal_pad + l_offset;
          int h_pad = h * stride - pad + h_offset;
          int w_pad = w * stride - pad + w_offset;
          if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
        		  && l_pad >= 0 && l_pad < length)
            data_im[((c_im * length + l_pad) * height + h_pad) * width + w_pad] +=
                data_col[((c * length_col + l) * height_col + h) * width_col + w];
        }
      }
    }
  }
}

// Explicit instantiation
template void col2vol_cpu<float>(const float* data_col, const int channels, const int length,
	    const int height, const int width, const int ksize, const int kdepth, const int pad,
	    const int temporal_pad, const int stride, const int temporal_stride, float* data_im);
template void col2vol_cpu<double>(const double* data_col, const int channels, const int length,
	    const int height, const int width, const int ksize, const int kdepth, const int pad,
	    const int temporal_pad, const int stride, const int temporal_stride, double* data_im);

}  // namespace caffe
