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

#ifndef CONVOLUTION3D_LAYER_HPP_
#define CONVOLUTION3D_LAYER_HPP_

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

namespace caffe {



template <typename Dtype>
class Convolution3DLayer : public Layer<Dtype> {
 public:
  explicit Convolution3DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
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

  int kernel_size_;
  int kernel_depth_;
  int stride_;
  int temporal_stride_;
  int num_;
  int channels_;
  int pad_;
  int temporal_pad_;
  int length_;
  int height_;
  int width_;
  int num_output_;
  int filter_group_;
  Blob<Dtype> col_buffer_;
  shared_ptr<SyncedMemory> bias_multiplier_;
  bool bias_term_;
  int M_;
  int K_;
  int N_;
};

}

#endif /* CONVOLUTION3D_LAYER_HPP_ */
