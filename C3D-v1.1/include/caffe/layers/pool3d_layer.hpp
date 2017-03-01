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


#ifndef POOL3D_LAYER_HPP_
#define POOL3D_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Pooling3DLayer : public Layer<Dtype> {
 public:
	explicit Pooling3DLayer(const LayerParameter& param)
	      : Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Pooling3D"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_size_;
  int kernel_depth_;
  int stride_;
  int temporal_stride_;
  int pad_;
  int channels_;
  int length_;
  int height_;
  int width_;
  int pooled_length_;
  int pooled_height_;
  int pooled_width_;
  Blob<Dtype> rand_idx_;
};

}

#endif /* POOL3D_LAYER_HPP_ */
