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




#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/voxel_wise_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void VoxelWiseSoftmaxWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SoftmaxLoss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
  CHECK_EQ(bottom[0]->length(), bottom[1]->length());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype VoxelWiseSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);

  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* truth = bottom[1]->cpu_data();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int length = bottom[0]->length();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  Dtype loss = 0;

  for (int i = 0; i < num; ++i) {
	  for (int l = 0; l < length; l++)
		  for (int h = 0; h < height; h++)
			  for (int w = 0; w < width; w++) {
				  int label = truth[bottom[1]->offset(i,0,l,h,w)];
				  if (label < channels) {
					  loss += -log(max(prob_data[bottom[0]->offset(i,label,l,h,w)],
		                     Dtype(FLT_MIN)));
				  }
			  }
  }
  return loss / (num * length * height * width);
}

template <typename Dtype>
void VoxelWiseSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());

  const Dtype* truth = (*bottom)[1]->cpu_data();

  const int num = (*bottom)[0]->num();
  const int channels = (*bottom)[0]->channels();
  const int length = (*bottom)[0]->length();
  const int height = (*bottom)[0]->height();
  const int width = (*bottom)[0]->width();

  for (int i = 0; i < num; ++i) {
	  for (int l = 0; l < length; l++)
		  for (int h = 0; h < height; h++)
			  for (int w = 0; w < width; w++) {
				  int label = truth[(*bottom)[1]->offset(i,0,l,h,w)];
				  if (label < channels) {
					  bottom_diff[(*bottom)[0]->offset(i,label,l,h,w)] -= 1;
				  }
			  }
  }

  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / (num * length * height * width), bottom_diff);
}


INSTANTIATE_CLASS(VoxelWiseSoftmaxWithLossLayer);


}  // namespace caffe
