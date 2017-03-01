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
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/voxel_wise_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void VoxelWiseSoftmaxLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Softmax Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Softmax Layer takes a single blob as output.";
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->length(),
      bottom[0]->height(), bottom[0]->width());
  scale_.Reshape(bottom[0]->num(), 1, bottom[0]->length(),
	      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
Dtype VoxelWiseSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();

  memcpy(top_data, bottom_data, sizeof(Dtype) * bottom[0]->count());
  // we need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int length = bottom[0]->length();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  // max over channels -> scale_data
  for (int i = 0; i < num; ++i) {
	  for (int l = 0; l < length; l++)
		  for (int h = 0; h < height; h++)
			  for (int w = 0; w < width; w++) {
				  scale_data[scale_.offset(i,0,l,h,w)] =
						  bottom_data[bottom[0]->offset(i,0,l,h,w)];
				  for (int c = 1; c < channels; c++)
					  scale_data[scale_.offset(i,0,l,h,w)] =
							  max(scale_data[bottom[0]->offset(i,0,l,h,w)],
									  bottom_data[bottom[0]->offset(i,c,l,h,w)]);
			  }
  }

  // subtraction
  for (int i = 0; i < num; ++i) {
	  for (int l = 0; l < length; l++)
		  for (int h = 0; h < height; h++)
			  for (int w = 0; w < width; w++)
				  for (int c = 0; c < channels; c++)
					  top_data[bottom[0]->offset(i,c,l,h,w)] -=
					  scale_data[scale_.offset(i,0,l,h,w)];
  }

  // Perform exponentiation
  caffe_exp<Dtype>((*top)[0]->count(), top_data, top_data);

  // sum after exp
  for (int i = 0; i < num; ++i) {
	  for (int l = 0; l < length; l++)
		  for (int h = 0; h < height; h++)
			  for (int w = 0; w < width; w++) {
				  scale_data[scale_.offset(i,0,l,h,w)] = 0;
				  for (int c = 0; c < channels; c++)
					  scale_data[scale_.offset(i,0,l,h,w)] +=
									  top_data[bottom[0]->offset(i,c,l,h,w)];
			  }
  }

  // Do division
  for (int i = 0; i < num; ++i) {
	  for (int l = 0; l < length; l++)
		  for (int h = 0; h < height; h++)
			  for (int w = 0; w < width; w++)
				  for (int c = 0; c < channels; c++)
					  top_data[bottom[0]->offset(i,c,l,h,w)] /=
					  scale_data[scale_.offset(i,0,l,h,w)];
  }
  return Dtype(0);
}

template <typename Dtype>
void VoxelWiseSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();


  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int length = top[0]->length();
  const int height = top[0]->height();
  const int width = top[0]->width();

  memcpy(bottom_diff, top_diff, sizeof(Dtype) * top[0]->count());

  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  for (int i = 0; i < num; ++i) {
	  for (int l = 0; l < length; l++)
		  for (int h = 0; h < height; h++)
			  for (int w = 0; w < width; w++) {
				  scale_data[scale_.offset(i,0,l,h,w)] = 0;
				  for (int c = 0; c < channels; c++)
					  scale_data[scale_.offset(i,0,l,h,w)] +=
							  	  	  top_diff[top[0]->offset(i,c,l,h,w)] *
									  top_data[top[0]->offset(i,c,l,h,w)];
			  }
  }

  // subtraction
  for (int i = 0; i < num; ++i) {
	  for (int l = 0; l < length; l++)
		  for (int h = 0; h < height; h++)
			  for (int w = 0; w < width; w++)
				  for (int c = 0; c < channels; c++)
					  bottom_diff[top[0]->offset(i,c,l,h,w)] -=
					  scale_data[scale_.offset(i,0,l,h,w)];
  }

  // elementwise multiplication
  caffe_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


INSTANTIATE_CLASS(VoxelWiseSoftmaxLayer);


}  // namespace caffe
