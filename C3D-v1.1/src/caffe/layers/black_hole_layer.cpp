/*
 * black_hole_layer.cpp
 *
 *  Created on: Oct 11, 2016
 *      Author: trandu
 */




#include <vector>

#include "caffe/layers/black_hole_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BlackHoleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape(5,1);
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void BlackHoleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void BlackHoleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void BlackHoleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(BlackHoleLayer);
#endif

INSTANTIATE_CLASS(BlackHoleLayer);
REGISTER_LAYER_CLASS(BlackHole);

}  // namespace caffe
