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


#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/volume_data_layer.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
Dtype VolumeDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
      prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
      cudaMemcpyHostToDevice));
  if (output_labels_) {
    CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
        prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
        cudaMemcpyHostToDevice));
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(VolumeDataLayer);

}  // namespace caffe