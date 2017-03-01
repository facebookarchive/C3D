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
 *
 */

#ifndef VOXEL_WISE_SOFTMAX_LOSS_LAYER_HPP_
#define VOXEL_WISE_SOFTMAX_LOSS_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/voxel_wise_softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
class VoxelWiseSoftmaxWithLossLayer : public Layer<Dtype> {
 public:
  explicit VoxelWiseSoftmaxWithLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), softmax_layer_(new VoxelWiseSoftmaxLayer<Dtype>(param)) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<VoxelWiseSoftmaxLayer<Dtype> > softmax_layer_;
  // prob stores the output probability of the layer.
  Blob<Dtype> prob_;
  // Vector holders to call the underlying softmax layer forward and backward.
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
};

}

#endif /* VOXEL_WISE_SOFTMAX_LOSS_LAYER_HPP_ */
