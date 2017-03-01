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

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/pool3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_depth, const int stride, const int temporal_stride, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pl = (index / pooled_width / pooled_height) % pooled_length;
    int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    int n = index / pooled_width / pooled_height / pooled_length / channels;
    int hstart = ph * stride;
    int hend = min(hstart + kernel_size, height);
    int wstart = pw * stride;
    int wend = min(wstart + kernel_size, width);
    int lstart = pl * temporal_stride;
    int lend = min(lstart + kernel_depth, length);
    Dtype maxval = -FLT_MAX;
    bottom_data += (n * channels + c) * length * height * width;
    for (int l = lstart; l < lend; ++l) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          maxval = max(maxval, bottom_data[(l * height + h) * width + w]);
        }
      }
    }
    top_data[index] = maxval;

  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_depth, const int stride, const int temporal_stride, const int pad, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pl = (index / pooled_width / pooled_height) % pooled_length;
    int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    int n = index / pooled_width / pooled_height / pooled_length / channels;
    int hstart = ph * stride - pad;
    int wstart = pw * stride - pad;
    int lstart = pl * temporal_stride;
    int hend = min(hstart + kernel_size, height + pad);
    int wend = min(wstart + kernel_size, width + pad);
    int lend = min(lstart + kernel_depth, length);
    int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    lend = min(lend, length);
    Dtype aveval = 0;
    bottom_data += (n * channels + c) * length * height * width;
    for (int l = lstart; l < lend; ++l) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          aveval += bottom_data[(l * height + h) * width + w];
        }
      }
    }
    top_data[index] = aveval / pool_size;

  }
}


template <typename Dtype>
Dtype Pooling3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = (*top)[0]->count();
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_, length_,
        height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_size_, kernel_depth_,
        stride_, temporal_stride_, top_data);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_, length_,
        height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_size_, kernel_depth_,
        stride_, temporal_stride_, pad_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOT IMPLEMENTED YET
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
  return Dtype(0.);
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_data, const Dtype* top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_depth, const int stride, const int temporal_stride, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int l = (index / width / height) % length;
    int c = (index / width / height / length) % channels;
    int n = index / width / height / length / channels;
    
    int phstart = (h < kernel_size) ? 0 : (h - kernel_size) / stride + 1;
    int phend = min(h / stride + 1, pooled_height);
    int pwstart = (w < kernel_size) ? 0 : (w - kernel_size) / stride + 1;
    int pwend = min(w / stride + 1, pooled_width);
    int plstart = (l < kernel_depth) ? 0 : (l - kernel_depth) / temporal_stride + 1;
    int plend = min(l / temporal_stride + 1, pooled_length);
    
    Dtype gradient = 0;
    Dtype bottom_datum =
        bottom_data[(((n * channels + c) * length + l) * height + h) * width + w];
    top_data += (n * channels + c) * pooled_length * pooled_height * pooled_width;
    top_diff += (n * channels + c) * pooled_length * pooled_height * pooled_width;
    for (int pl = plstart; pl < plend; ++pl) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          gradient += top_diff[(pl * pooled_height + ph) * pooled_width + pw] *
              (bottom_datum == top_data[(pl * pooled_height + ph) * pooled_width + pw]);
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_depth, const int stride, const int temporal_stride, const int pad,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int l = (index / width / height) % length;
    int c = (index / width / height / length) % channels;
    int n = index / width / height / length / channels;
    int phstart = (h < kernel_size) ? 0 : (h - kernel_size) / stride + 1;
    int phend = min(h / stride + 1, pooled_height);
    int pwstart = (w < kernel_size) ? 0 : (w - kernel_size) / stride + 1;
    int pwend = min(w / stride + 1, pooled_width);
    int plstart = (l < kernel_depth) ? 0 : (l - kernel_depth) / temporal_stride + 1;
    int plend = min(l / temporal_stride + 1, pooled_length);
    
    Dtype gradient = 0;
    top_diff += (n * channels + c) * pooled_length * pooled_height * pooled_width;
    for (int pl = plstart; pl < plend; ++pl) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // figure out the pooling size
          int hstart = ph * stride - pad;
          int wstart = pw * stride - pad;
          int lstart = pl * temporal_stride;
          int hend = min(hstart + kernel_size, height + pad);
          int wend = min(wstart + kernel_size, width + pad);
          int lend = min(lstart + kernel_depth, length);
          int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
          gradient += top_diff[(pl * pooled_height + ph) * pooled_width + pw] / pool_size;
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void Pooling3DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  int count = (*bottom)[0]->count();
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (*bottom)[0]->gpu_data(), top[0]->gpu_data(), top_diff,
        top[0]->num(), channels_, length_, height_, width_, pooled_length_, pooled_height_,
        pooled_width_, kernel_size_, kernel_depth_, stride_, temporal_stride_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_, length_,
        height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_size_, kernel_depth_, 
        stride_, temporal_stride_, pad_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOT IMPLEMENTED YET
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_CLASS(Pooling3DLayer);


}  // namespace caffe
