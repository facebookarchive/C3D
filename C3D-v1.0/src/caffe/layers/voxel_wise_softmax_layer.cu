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

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/voxel_wise_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {



template <typename Dtype>
Dtype VoxelWiseSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  return Forward_cpu(bottom, top);
}

template <typename Dtype>
void VoxelWiseSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(VoxelWiseSoftmaxLayer);


}  // namespace caffe
