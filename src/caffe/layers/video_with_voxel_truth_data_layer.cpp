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




#include <stdint.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>


#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/video_with_voxel_truth_data_layer.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void* VideoWithVoxelTruthDataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  VideoWithVoxelTruthDataLayer<Dtype>* layer = static_cast<VideoWithVoxelTruthDataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);

  CHECK(layer->prefetch_data_);
  CHECK(layer->prefetch_truth_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_truth = layer->prefetch_truth_->mutable_cpu_data();;

  const Dtype scale = layer->layer_param_.image_data_param().scale();
  const int batch_size = layer->layer_param_.image_data_param().batch_size();
  const int crop_size = layer->layer_param_.image_data_param().crop_size();
  const bool mirror = layer->layer_param_.image_data_param().mirror();
  const bool use_byte_input = layer->layer_param_.image_data_param().use_byte_input();
  const int num_truth_channels = layer->layer_param_.image_data_param().num_truth_channels();
  const bool is_flow = layer->layer_param_.image_data_param().is_flow();
  const Dtype truth_scale = layer->layer_param_.image_data_param().truth_scale();
  const Dtype clip_min = (Dtype)layer->layer_param_.image_data_param().clip_min();
  const Dtype clip_max = (Dtype)layer->layer_param_.image_data_param().clip_max();
  const Dtype clip_diff = clip_max - clip_min;
  const bool has_clip_min = layer->layer_param_.image_data_param().has_clip_min();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }

  const int channels = layer->datum_channels_;
  const int length = layer->datum_length_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  const int size = layer->datum_size_;
  const int chunks_size = layer->shuffle_index_.size();
  const Dtype* mean = layer->data_mean_.cpu_data();
  const int show_data = layer->layer_param_.image_data_param().show_data();
  char *data_buffer;
  if (show_data)
	  data_buffer = new char[size];
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK_GT(chunks_size, layer->lines_id_);
    bool read_status;
    int id = layer->shuffle_index_[layer->lines_id_];
    Blob<Dtype> data_blob;

    if (use_byte_input)
    	read_status = load_blob_from_uint8_binary<Dtype>(layer->file_list_[id].c_str(), &data_blob);
    else
    	read_status = load_blob_from_binary<Dtype>(layer->file_list_[id].c_str(), &data_blob);

    if (layer->phase_ == Caffe::TEST){
    	CHECK(read_status) << "Testing must not miss any example";
    }

    if (!read_status) {
        layer->lines_id_++;
        if (layer->lines_id_ >= chunks_size) {
          // We have reached the end. Restart from the first.
          DLOG(INFO) << "Restarting data prefetching from start.";
          layer->lines_id_ = 0;
          if (layer->layer_param_.image_data_param().shuffle()){
        	  std::random_shuffle(layer->shuffle_index_.begin(), layer->shuffle_index_.end());
          }
        }
        item_id--;
        continue;
    }

    const Dtype* data = data_blob.cpu_data();
    if (crop_size) {
      int h_off, w_off;
      // We only do random crop when we do training.
      if (layer->phase_ == Caffe::TRAIN) {
        h_off = layer->PrefetchRand() % (height - crop_size);
        w_off = layer->PrefetchRand() % (width - crop_size);
      } else {
        h_off = (height - crop_size) / 2;
        w_off = (width - crop_size) / 2;
      }
      if (mirror && layer->PrefetchRand() % 2) {
        // Copy mirrored version
        for (int c = 0; c < channels; ++c) {
          for (int l = 0; l < length; ++l) {
            for (int h = 0; h < crop_size; ++h) {
              for (int w = 0; w < crop_size; ++w) {
                int top_index = (((item_id * (channels-num_truth_channels) + c) * length + l) * crop_size + h)
                              * crop_size + (crop_size - 1 - w);
                int data_index = ((c * length + l) * height + h + h_off) * width + w + w_off;
                int truth_index = (((item_id * num_truth_channels + c-channels+num_truth_channels) * length + l) * crop_size + h)
                                      * crop_size + (crop_size - 1 - w);
                Dtype datum_element = data[data_index];
                if (c < channels-num_truth_channels) {
                	top_data[top_index] = (datum_element - mean[data_index]) * scale;
                } else {
                	if (is_flow) {
                		if (c == channels-num_truth_channels)
                			datum_element = - datum_element;
                	}
                	if (has_clip_min) {
						top_truth[truth_index] = (datum_element - clip_min)/clip_diff;
						if (top_truth[truth_index]<0)
							top_truth[truth_index] = 0;
						if (top_truth[truth_index]>1)
							top_truth[truth_index] = 1;
                	} else {
                		top_truth[truth_index] = datum_element * truth_scale;
                	}
                }
                if (show_data) {
                	if (c < channels - num_truth_channels)
                	data_buffer[((c * length + l) * crop_size + h)
                                * crop_size + (crop_size - 1 - w)] = static_cast<uint8_t>(data[data_index]);
                	else
                		data_buffer[((c * length + l) * crop_size + h)
                		            * crop_size + (crop_size - 1 - w)] =
                		            		static_cast<uint8_t>(255*fabs(top_truth[truth_index]));
                }
              }
            }
          }
        }
      } else {
        // Normal copy
        for (int c = 0; c < channels; ++c) {
          for (int l = 0; l < length; ++l) {
            for (int h = 0; h < crop_size; ++h) {
              for (int w = 0; w < crop_size; ++w) {
                int top_index = (((item_id * (channels-num_truth_channels) + c) * length + l) * crop_size + h)
                              * crop_size + w;
                int data_index = ((c * length + l) * height + h + h_off) * width + w + w_off;
                int truth_index = (((item_id * num_truth_channels + c-channels+num_truth_channels) * length + l) * crop_size + h)
                                      * crop_size + w;
                Dtype datum_element = data[data_index];
                if (c < channels - num_truth_channels) {
                	top_data[top_index] = (datum_element - mean[data_index]) * scale;
                } else {
                	if (has_clip_min) {
                	top_truth[truth_index] = (datum_element - clip_min)/clip_diff;
                	if (top_truth[truth_index]<0)
                		top_truth[truth_index] = 0;
                	if (top_truth[truth_index]>1)
                		top_truth[truth_index] = 1;
                	} else {
                		top_truth[truth_index] = datum_element * truth_scale;
                	}
                }
                if (show_data) {
                	if (c < channels - num_truth_channels)
                	data_buffer[((c * length + l) * crop_size + h)
                                * crop_size + w] = static_cast<uint8_t>(data[data_index]);
                	else
                		data_buffer[((c * length + l) * crop_size + h)
                		        * crop_size + w] =
                		        		static_cast<uint8_t>(255*fabs(top_truth[truth_index]));
                }
              }
            }
          }
        }
      }
    } else {
    	int channel_size = length * height * width;
    	int truth_size = num_truth_channels * channel_size;
    	for (int j = 0; j < size; ++j) {
    		if (j < size - truth_size) {
    			top_data[item_id * size + j] =
    					(data[j] - mean[j]) * scale;
    		} else {
    			top_truth[item_id * truth_size + j - size + truth_size] =
    					data[j];
    		}
    	}
    }

    if (show_data>0){
    	int image_size, channel_size;
    	if (crop_size){
    		image_size = crop_size * crop_size;
    	}else{
    		image_size = height * width;
    	}
    	channel_size = length * image_size;
    	for (int l = 0; l < length; ++l) {
    		for (int c = 0; c < channels; ++c) {
    			cv::Mat img;
    			char ch_name[64];
    			if (crop_size)
    				BufferToGrayImage(data_buffer + c * channel_size + l * image_size, crop_size, crop_size, &img);
    			else
    				BufferToGrayImage(data_buffer + c * channel_size + l * image_size, height, width, &img);
    			sprintf(ch_name, "C%d", c);
    			cv::namedWindow(ch_name, CV_WINDOW_AUTOSIZE);
    			cv::imshow( ch_name, img);
    		}
    		cv::waitKey(100);
    	}
    }

    layer->lines_id_++;
    if (layer->lines_id_ >= chunks_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      layer->lines_id_ = 0;
      if (layer->layer_param_.image_data_param().shuffle()){
    	  std::random_shuffle(layer->shuffle_index_.begin(), layer->shuffle_index_.end());
      }
    }
  }
  if (show_data & data_buffer!=NULL)
	  delete []data_buffer;
  return static_cast<void*>(NULL);
}

template <typename Dtype>
VideoWithVoxelTruthDataLayer<Dtype>::~VideoWithVoxelTruthDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void VideoWithVoxelTruthDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Data Layer takes exactly two blobs as output.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  const bool use_byte_input = this->layer_param_.image_data_param().use_byte_input();
  const int num_truth_channels = this->layer_param_.image_data_param().num_truth_channels();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  int count = 0;
  string filename;

  while (infile >> filename) {
		  file_list_.push_back(filename);
		  shuffle_index_.push_back(count);
		  count++;
  }

  if (count==0){
	  LOG(INFO) << "failed to read chunk list" << std::endl;
  }

  if (this->layer_param_.image_data_param().shuffle()){
	  LOG(INFO) << "Shuffling data";
	  const unsigned int prefetch_rng_seed = caffe_rng_rand();
	  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	  std::random_shuffle(shuffle_index_.begin(), shuffle_index_.end());
  }
  LOG(INFO) << "A total of " << shuffle_index_.size() << " examples.";

  lines_id_ = 0;

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(shuffle_index_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  Blob<Dtype> data_blob;
  if (use_byte_input) {
	  CHECK(load_blob_from_uint8_binary<Dtype>(file_list_[0].c_str(), &data_blob))
  	  << "Cannot load video and segmentation data";
  } else {
	  CHECK(load_blob_from_binary<Dtype>(file_list_[0].c_str(), &data_blob))
  	  << "Cannot load video and segmentation data";
  }

  // image
  int crop_size = this->layer_param_.image_data_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.image_data_param().batch_size(),
                    data_blob.channels()-num_truth_channels, data_blob.length(), crop_size, crop_size);
    (*top)[1]->Reshape(this->layer_param_.image_data_param().batch_size(),
    				num_truth_channels, data_blob.length(), crop_size, crop_size);
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.image_data_param().batch_size(), data_blob.channels()-num_truth_channels,
        	data_blob.length(), crop_size, crop_size));
    prefetch_truth_.reset(new Blob<Dtype>(
            this->layer_param_.image_data_param().batch_size(), num_truth_channels,
            	data_blob.length(), crop_size, crop_size));
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.image_data_param().batch_size(), data_blob.channels() - num_truth_channels,
        data_blob.length(), data_blob.height(), data_blob.width());
    (*top)[1]->Reshape(
            this->layer_param_.image_data_param().batch_size(), num_truth_channels,
            data_blob.channels(), data_blob.height(), data_blob.width());
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.image_data_param().batch_size(), data_blob.channels() - num_truth_channels,
        data_blob.length(), data_blob.height(), data_blob.width()));
    prefetch_truth_.reset(new Blob<Dtype>(
            this->layer_param_.image_data_param().batch_size(), num_truth_channels,
            data_blob.length(), data_blob.height(), data_blob.width()));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->length() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  LOG(INFO) << "output truth size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->length() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();


  // datum size
  datum_channels_ = data_blob.channels();
  datum_length_ = data_blob.length();
  datum_height_ = data_blob.height();
  datum_width_ = data_blob.width();
  datum_size_ = data_blob.channels() * data_blob.length() * data_blob.height() * data_blob.width();
  CHECK_GT(datum_height_, crop_size);
  CHECK_GT(datum_width_, crop_size);
  // check if we want to have mean
  if (this->layer_param_.image_data_param().has_mean_file()) {
    const string& mean_file = this->layer_param_.image_data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_-num_truth_channels);
    CHECK_EQ(data_mean_.length(), datum_length_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_-num_truth_channels, datum_length_, datum_height_, datum_width_);
    if (this->layer_param_.image_data_param().has_mean_value()){
    	LOG(INFO) << "Using mean value of " << this->layer_param_.image_data_param().mean_value();
      caffe::caffe_set(data_mean_.count(), (Dtype)this->layer_param_.image_data_param().mean_value(), (Dtype*)data_mean_.mutable_cpu_data());
    }
  }


  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_truth_->mutable_cpu_data();

  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void VideoWithVoxelTruthDataLayer<Dtype>::CreatePrefetchThread() {
  phase_ = Caffe::phase();
  const bool prefetch_needs_rand = (phase_ == Caffe::TRAIN) &&
      (this->layer_param_.image_data_param().mirror() ||
       this->layer_param_.image_data_param().crop_size());
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, VideoWithVoxelTruthDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void VideoWithVoxelTruthDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int VideoWithVoxelTruthDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype VideoWithVoxelTruthDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_truth_->count(), prefetch_truth_->cpu_data(),
             (*top)[1]->mutable_cpu_data());

  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(VideoWithVoxelTruthDataLayer);

}  // namespace caffe
