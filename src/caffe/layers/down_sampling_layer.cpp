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




#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/down_sampling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DownSamplingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) <<
    "DownSamplingLayer takes exactly one blob as input.";
  CHECK_EQ(top->size(), 1) <<
    "DownSamplingLayer takes a single blob as output.";

  spatial_factor_ = this->layer_param_.down_sampling_param().spatial_factor();
  temporal_factor_ = this->layer_param_.down_sampling_param().temporal_factor();
  type_ = this->layer_param_.down_sampling_param().type();

  CHECK_GT(spatial_factor_, 1) <<
    "spatial factor should be >= 1";
  CHECK_GT(temporal_factor_, 1) <<
    "temporal factor should be >= 1";

  // Initialize with the first blob.
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  length_ = bottom[0]->length() / temporal_factor_;
  height_ = bottom[0]->height() / spatial_factor_;
  width_ = bottom[0]->width() / spatial_factor_;

  (*top)[0]->Reshape(num_, channels_, length_, height_, width_);
}

template <typename Dtype>
Dtype DownSamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  if (type_== DownSamplingParameter_DownSamplingType_VOTE) {
	  max_ = bottom_data[0];
	  min_ = bottom_data[0];

	  for (int i = 1; i<bottom[0]->count(); i++) {
		  if (max_ < bottom_data[i])
			  max_ = bottom_data[i];
		  if (min_ > bottom_data[i])
			  min_ = bottom_data[i];
	  }
	  int num_values = int(max_) - int(min_) + 1;
	  Dtype* votes = new Dtype[num_values];
	  for (int n = 0; n < num_; ++n)
		  for (int c = 0; c < channels_; ++c)
			  for (int l = 0; l < length_; ++l)
				  for (int h = 0; h < height_; ++h)
					  for (int w = 0; w < width_; ++w) {
						  caffe::caffe_set(num_values, (Dtype)0, votes);
						  for (int l0 = 0; l0 < temporal_factor_; ++l0)
							  for (int h0 = 0; h0 < spatial_factor_; ++h0)
								  for (int w0 = 0; w0 < spatial_factor_; ++w0) {
									Dtype v =
									bottom_data[bottom[0]->offset(n,c,
											l * temporal_factor_ + l0,
											h * spatial_factor_ + h0,
											w * spatial_factor_ + w0)];
									votes[int(v-min_)]++;
								  }
						  Dtype best_vote = Dtype(0);
						  for (int i = 1; i<num_values; i++)
							  if (votes[i] > votes[int(best_vote)])
								  best_vote = (Dtype)i;
						  top_data[(*top)[0]->offset(n,c,l,h,w)] = Dtype(best_vote+min_);
					  }
	  delete []votes;
  } else if (type_== DownSamplingParameter_DownSamplingType_AVERAGE) {
	  // NOT implement yet
	  CHECK_EQ(1, 0) << "Oops, average down sampling is NOT implement yet!";
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(DownSamplingLayer);

}  // namespace caffe
