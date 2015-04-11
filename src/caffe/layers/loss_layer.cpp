// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void LossLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
  CHECK_LE(top->size(), 1) << "Loss Layer takes no more than one output.";
  if (top->size() == 1) {
   // Layers should copy the loss in the top blob
   (*top)[0]->Reshape(1, 1, 1, 1, 1);
  }
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  FurtherSetUp(bottom, top);
}

INSTANTIATE_CLASS(LossLayer);

}  // namespace caffe
