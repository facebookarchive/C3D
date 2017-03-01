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


#include <vector>
#include "caffe/layer.hpp"
#include "caffe/convolution3d_layer.hpp"
#include "caffe/util/vol2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype Convolution3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();

  int weight_offset = M_ * K_;
  int top_offset = M_ * N_;
  
  for (int n = 0; n < num_; ++n) {
    // First, im2col
    vol2col_gpu(bottom_data + bottom[0]->offset(n), channels_, length_, height_,
                      width_, kernel_size_, kernel_depth_, pad_, temporal_pad_, stride_, temporal_stride_, col_data);
    // Second, innerproduct with groups
    for (int g=0; g<filter_group_; ++g){
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight + g * weight_offset, col_data,
        (Dtype)0., top_data + (*top)[0]->offset(n) + g * top_offset);
    }
    // third, add bias
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    CUDA_CHECK(cudaMemset(bias_diff, 0,
        sizeof(Dtype) * this->blobs_[1]->count()));
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
          1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          1., bias_diff);
    }
  }

  int weight_offset = M_ * K_;
  int top_offset = M_ * N_;
  
  CUDA_CHECK(cudaMemset(weight_diff, 0,
      sizeof(Dtype) * this->blobs_[0]->count()));
  for (int n = 0; n < num_; ++n) {
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    vol2col_gpu(bottom_data + (*bottom)[0]->offset(n), channels_, length_, height_,
                      width_, kernel_size_, kernel_depth_, pad_, temporal_pad_, stride_, temporal_stride_, col_data);
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g=0; g<filter_group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff + top[0]->offset(n) + g * top_offset,
        col_data, (Dtype)1.,
        weight_diff + g * weight_offset);
	}
    // gradient w.r.t. bottom data, if necessary
    if (propagate_down) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
        (Dtype)1., weight,
        top_diff + top[0]->offset(n),
        (Dtype)0., col_diff);
    
      for (int g=1; g<filter_group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + g * weight_offset,
          top_diff + top[0]->offset(n) + g * top_offset,
          (Dtype)1., col_diff);
	  }
      // col2vol back to the data
      col2vol_gpu(col_diff, channels_, length_, height_, width_, kernel_size_, kernel_depth_, pad_,
          temporal_pad_, stride_, temporal_stride_, bottom_diff + (*bottom)[0]->offset(n));
    }
  }
}


INSTANTIATE_CLASS(Convolution3DLayer);

}  // namespace caffe
