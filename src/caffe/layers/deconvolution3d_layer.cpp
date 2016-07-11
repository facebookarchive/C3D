/*
 *
 *  Copyright (c) 2016, Facebook, Inc. All rights reserved.
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
 */






#include "caffe/layer.hpp"
#include "caffe/deconvolution3d_layer.hpp"
#include "caffe/util/vol2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <vector>

namespace caffe {

template <typename Dtype>
void Deconvolution3DLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  CHECK_EQ(bottom.size(), 1) << "Deconv Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Deconv Layer takes a single blob as output.";

  kernel_size_ = this->layer_param_.convolution_param().kernel_size();
  kernel_depth_ = this->layer_param_.convolution_param().kernel_depth();
  stride_ = this->layer_param_.convolution_param().stride();
  temporal_stride_ = this->layer_param_.convolution_param().temporal_stride();
  pad_ = this->layer_param_.convolution_param().pad();
  temporal_pad_ = this->layer_param_.convolution_param().temporal_pad();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  length_ = bottom[0]->length();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);


  // The vol2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.

  length_out_ = (length_ - 1) * temporal_stride_ - 2 * temporal_pad_ + kernel_depth_;
  height_out_ = (height_ - 1) * stride_ - 2 * pad_ + kernel_size_;
  width_out_ = (width_ - 1) * stride_ - 2 * pad_ + kernel_size_;

  // buffer for one image
  col_buffer_.Reshape(
      1, num_output_ * kernel_depth_ * kernel_size_ * kernel_size_, length_, height_, width_);


  bias_term_ = this->layer_param_.convolution_param().bias_term();

  // Figure out the dimensions for individual gemms.
  M_ = channels_;
  K_ = num_output_ * kernel_depth_ * kernel_size_ * kernel_size_;
  N_ = length_ * height_ * width_;
  M0_ = num_output_;
  N0_ = length_out_ * height_out_ * width_out_;

  // output size
  (*top)[0]->Reshape(bottom[0]->num(), num_output_, length_out_, height_out_, width_out_);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    this->blobs_[0].reset(new Blob<Dtype>(
        channels_, num_output_, kernel_depth_, kernel_size_, kernel_size_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }


  }

  // Set up the bias filler
  if (bias_term_) {
    bias_multiplier_.reset(new SyncedMemory(N0_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < N0_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
}


template <typename Dtype>
Dtype Deconvolution3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();


  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
    (Dtype)1., weight, bottom_data + bottom[0]->offset(n),
    (Dtype)0., col_data);

    // col2vol from col_data -> top_data
    col2vol_cpu(col_data, num_output_, length_out_, height_out_, width_out_,
    kernel_size_, kernel_depth_, pad_, temporal_pad_, stride_, temporal_stride_,
    top_data + (*top)[0]->offset(n));

    // third, add bias
    if (bias_term_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      N0_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
      reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
      (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }

  return Dtype(0.);
}

template <typename Dtype>
void Deconvolution3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < num_; ++n) {
    	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N0_,
    	1., top_diff + top[0]->offset(n),
    	reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
    	bias_diff);
    }
  }


  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < num_; ++n) {
	  vol2col_cpu(top_diff + top[0]->offset(n), num_output_, length_out_, height_out_, width_out_,
	  	    kernel_size_, kernel_depth_, pad_, temporal_pad_, stride_, temporal_stride_, col_diff);

	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
				  (Dtype)1., bottom_data + (*bottom)[0]->offset(n),
				  col_diff, (Dtype)1.,
				  weight_diff);

	  // gradient w.r.t. bottom data, if necessary
	  if (propagate_down) {
		  // compute first filter group -> col_diff
		  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
				  (Dtype)1., weight, col_diff,
				  (Dtype)0., bottom_diff + (*bottom)[0]->offset(n));
	  }
  }
}


INSTANTIATE_CLASS(Deconvolution3DLayer);

}  // namespace caffe
