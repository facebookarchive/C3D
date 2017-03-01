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
#include "caffe/layers/convolution3d_layer.hpp"
#include "caffe/util/vol2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
void Convolution3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// already done in LayersetUp
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Conv Layer takes a single blob as output.";

  kernel_size_ = this->layer_param_.convolution3d_param().kernel_size();
  kernel_depth_ = this->layer_param_.convolution3d_param().kernel_depth();
  stride_ = this->layer_param_.convolution3d_param().stride();
  temporal_stride_ = this->layer_param_.convolution3d_param().temporal_stride();
  pad_ = this->layer_param_.convolution3d_param().pad();
  temporal_pad_ = this->layer_param_.convolution3d_param().temporal_pad();
  num_ = bottom[0]->shape(0);
  channels_ = bottom[0]->shape(1);
  length_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);
  num_output_ = this->layer_param_.convolution3d_param().num_output();
  filter_group_ = this->layer_param_.convolution3d_param().filter_group();
  CHECK_GT(num_output_, 0);

  // number of output filters must be divided by filter_group
  CHECK_EQ(num_output_ % filter_group_, 0);

  // The vol2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.

  int height_out = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int width_out = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int length_out = (length_ + 2 * temporal_pad_ - kernel_depth_) / temporal_stride_ + 1;

  vector<int> top_shape(5);
  vector<int> weight_shape(5);
  vector<int> col_shape(5);
  vector<int> bias_shape(5);

  col_shape[0] = 1;
  col_shape[1] = channels_ * kernel_depth_ * kernel_size_ * kernel_size_;
  col_shape[2] = length_out;
  col_shape[3] = height_out;
  col_shape[4] = width_out;


  // buffer for one image
  col_buffer_.Reshape(col_shape);
      //1, channels_ * kernel_depth_ * kernel_size_ * kernel_size_, length_out, height_out, width_out);


  bias_term_ = this->layer_param_.convolution3d_param().bias_term();

  // Figure out the dimensions for individual gemms.
  M_ = num_output_ / filter_group_; // doing convolution filter_group_ times per volume
  K_ = channels_ * kernel_depth_ * kernel_size_ * kernel_size_;
  N_ = length_out * height_out * width_out;

  // output size
  top_shape[0] = bottom[0]->shape(0);
  top_shape[1] = num_output_;
  top_shape[2] = length_out;
  top_shape[3] = height_out;
  top_shape[4] = width_out;

  top[0]->Reshape(top_shape);
  //(*top)[0]->Reshape(bottom[0]->num(), num_output_, length_out, height_out, width_out);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }

    weight_shape[0] = num_output_;
    weight_shape[1] = channels_;
    weight_shape[2] = kernel_depth_;
    weight_shape[3] = kernel_size_;
    weight_shape[4] = kernel_size_;
    // Initialize the weights
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
        //num_output_, channels_, kernel_depth_, kernel_size_, kernel_size_));

    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution3d_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      bias_shape[0] = 1;
      bias_shape[1] = 1;
      bias_shape[2] = 1;
      bias_shape[3] = 1;
      bias_shape[4] = num_output_;
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    		  //1, 1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution3d_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
    // profiling number of params
    std::ofstream out("profile.log", std::ofstream::out | std::ofstream::app);
    out << this->layer_param_.name() << " " << num_output_ * channels_ * kernel_depth_ * kernel_size_ * kernel_size_ + num_output_ << "\n";
    out.close();
  }

  // Set up the bias filler
  if (bias_term_) {
    bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < N_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
}


template <typename Dtype>
void Convolution3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top) {

  CPUTimer timer;
  timer.Start();
  double computation;

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  int weight_offset = M_ * K_;
  int top_offset = M_ * N_;
  static const int zero_array[] = {0, 0, 0, 0, 0};
  vector<int> offset_indices(zero_array, zero_array + 5);

  for (int n = 0; n < num_; ++n) {
	offset_indices[0] = n;

	// First, im2col
    vol2col_cpu(bottom_data + bottom[0]->offset(offset_indices), channels_, length_, height_,
	    		  width_, kernel_size_, kernel_depth_, pad_, temporal_pad_, stride_, temporal_stride_, col_data);

    // Second, inner-product without filter groups
	for (int g=0 ; g < filter_group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
	    (Dtype)1., weight + g * weight_offset, col_data,
	    (Dtype)0., top_data + top[0]->offset(offset_indices) + g * top_offset);
	}
      // third, add bias
	if (bias_term_) {
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
	   N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
	   reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
	   (Dtype)1., top_data + top[0]->offset(offset_indices));
    }

  }
  computation = timer.MilliSeconds();
  std::ofstream out("profile_inference.log", std::ofstream::out | std::ofstream::app);
  out << this->layer_param_.name() << " CPU " << computation << "\n";
  out.close();
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  static const int zero_array[] = {0, 0, 0, 0, 0};
  vector<int> offset_indices(zero_array, zero_array + 5);

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < num_; ++n) {
      offset_indices[0] = n;
	  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
		  1., top_diff + top[0]->offset(offset_indices),
		  reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
		  bias_diff);
    }
  }

  int weight_offset = M_ * K_;
  int top_offset = M_ * N_;

  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < num_; ++n) {
	  offset_indices[0] = n;
	// since we saved memory in the forward pass by not storing all col data,
	// we will need to recompute them.
	vol2col_cpu(bottom_data + bottom[0]->offset(offset_indices), channels_, length_, height_,
					  width_, kernel_size_, kernel_depth_, pad_, temporal_pad_, stride_,
					  temporal_stride_, col_data);

	// gradient w.r.t. weight. Note that we will accumulate diffs.
	for (int g=0; g<filter_group_; ++g){
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
				(Dtype)1., top_diff + top[0]->offset(offset_indices) + g * top_offset,
				col_data, (Dtype)1.,
				weight_diff + g * weight_offset);
	}

	// gradient w.r.t. bottom data, if necessary
	if (propagate_down[0]) {
	  // compute first filter group -> col_diff
	  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
	    (Dtype)1., weight,
		top_diff + top[0]->offset(offset_indices),
		(Dtype)0., col_diff);

	  // accumulate the other filter groups -> col_diff
	  for (int g=1; g<filter_group_; ++g){
	    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
	      (Dtype)1., weight + g * weight_offset,
		  top_diff + top[0]->offset(offset_indices) + g * top_offset,
		  (Dtype)1., col_diff);
	  }

	  // vol2im back to the data
	  col2vol_cpu(col_diff, channels_, length_, height_, width_, kernel_size_, kernel_depth_, pad_,
		  temporal_pad_, stride_, temporal_stride_, bottom_diff + bottom[0]->offset(offset_indices));
	}

  }
}

#ifdef CPU_ONLY
STUB_GPU(Convolution3DLayer);
#endif

INSTANTIATE_CLASS(Convolution3DLayer);
REGISTER_LAYER_CLASS(Convolution3D);

}  // namespace caffe
