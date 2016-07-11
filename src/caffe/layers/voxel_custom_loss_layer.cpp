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
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/voxel_custom_loss_layer.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void VoxelCustomLossLayer<Dtype>::FurtherSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->length(), bottom[1]->length());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->length(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
Dtype VoxelCustomLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  Dtype loss = Dtype(0);
  const Dtype *truth = bottom[1]->cpu_data();
  Dtype *diff = diff_.mutable_cpu_data();
  for (int i=0; i < count; i++) {
	if (diff[i]<Dtype(-1))
		diff[i] = Dtype(-1);
	if (diff[i]>Dtype(1))
		diff[i] = Dtype(1);
	if (fabs(diff[i])>1)
		loss += fabs(diff[i]);
	else
		loss += diff[i]*diff[i];
  }
  return (loss / count);
}

template <typename Dtype>
void VoxelCustomLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  caffe_cpu_axpby(
      (*bottom)[0]->count(),              // count
      Dtype(1) / (*bottom)[0]->num(),     // alpha
      diff_.cpu_data(),                   // a
      Dtype(0),                           // beta
      (*bottom)[0]->mutable_cpu_diff());  // b
}

INSTANTIATE_CLASS(VoxelCustomLossLayer);

}  // namespace caffe
