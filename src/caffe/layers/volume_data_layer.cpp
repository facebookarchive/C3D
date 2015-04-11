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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>


#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/volume_data_layer.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void* VolumeDataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  VolumeDataLayer<Dtype>* layer = static_cast<VolumeDataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);
  VolumeDatum datum;
  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label;
  if (layer->output_labels_) {
    top_label = layer->prefetch_label_->mutable_cpu_data();
  }
  const Dtype scale = layer->layer_param_.data_param().scale();
  const int batch_size = layer->layer_param_.data_param().batch_size();
  const int crop_size = layer->layer_param_.data_param().crop_size();
  const bool mirror = layer->layer_param_.data_param().mirror();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = layer->datum_channels_;
  const int length = layer->datum_length_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  const int size = layer->datum_size_;
  const Dtype* mean = layer->data_mean_.cpu_data();
  const int show_data = layer->layer_param_.data_param().show_data();
  char *data_buffer;
  if (show_data)
	  data_buffer = new char[size];
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK(layer->iter_);
    CHECK(layer->iter_->Valid());
    datum.ParseFromString(layer->iter_->value().ToString());
    const string& data = datum.data();
    if (crop_size) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
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
                int top_index = (((item_id * channels + c) * length + l) * crop_size + h)
                              * crop_size + (crop_size - 1 - w);
                int data_index = ((c * length + l) * height + h + h_off) * width + w + w_off;
                Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                top_data[top_index] = (datum_element - mean[data_index]) * scale;
                if (show_data)
                	data_buffer[((c * length + l) * crop_size + h)
                                * crop_size + (crop_size - 1 - w)] = static_cast<uint8_t>(data[data_index]);
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
                int top_index = (((item_id * channels + c) * length + l) * crop_size + h)
                              * crop_size + w;
                int data_index = ((c * length + l) * height + h + h_off) * width + w + w_off;
                Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                top_data[top_index] = (datum_element - mean[data_index]) * scale;
                if (show_data)
                	data_buffer[((c * length + l) * crop_size + h)
                                * crop_size + w] = static_cast<uint8_t>(data[data_index]);
              }
            }
          }
        }
      }
    } else {
      // we will prefer to use data() first, and then try float_data()
      if (data.size()) {
        for (int j = 0; j < size; ++j) {
          Dtype datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[j]));
          top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
          if (show_data)
        	  data_buffer[j] = static_cast<uint8_t>(data[j]);
        }
      } else {
        for (int j = 0; j < size; ++j) {
          top_data[item_id * size + j] =
              (datum.float_data(j) - mean[j]) * scale;
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
    			sprintf(ch_name, "Channel %d", c);
    			cv::namedWindow(ch_name, CV_WINDOW_AUTOSIZE);
    			cv::imshow( ch_name, img);
    		}
    		cv::waitKey(100);
    	}
    }
    if (layer->output_labels_) {
      top_label[item_id] = datum.label();
    }
    // go to the next iteration
    layer->iter_->Next();
    if (!layer->iter_->Valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      layer->iter_->SeekToFirst();
    }
  }
  if (show_data & data_buffer!=NULL)
	  delete []data_buffer;
  return static_cast<void*>(NULL);
}

template <typename Dtype>
VolumeDataLayer<Dtype>::~VolumeDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void VolumeDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_GE(top->size(), 1) << "Data Layer takes at least one blob as output.";
  CHECK_LE(top->size(), 2) << "Data Layer takes at most two blobs as output.";
  if (top->size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  // Initialize the leveldb
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.data_param().source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
      << this->layer_param_.data_param().source() << std::endl
      << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      iter_->Next();
      if (!iter_->Valid()) {
        iter_->SeekToFirst();
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  VolumeDatum datum;
  datum.ParseFromString(iter_->value().ToString());
  // image
  int crop_size = this->layer_param_.data_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       datum.channels(), datum.length(), crop_size, crop_size);
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.data_param().batch_size(), datum.channels(), datum.length(),
        crop_size, crop_size));
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(), datum.length(),
        datum.height(), datum.width());
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.data_param().batch_size(), datum.channels(), datum.length(),
        datum.height(), datum.width()));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->length() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1, 1);
    prefetch_label_.reset(
        new Blob<Dtype>(this->layer_param_.data_param().batch_size(), 1, 1, 1, 1));
  }

  // datum size
  datum_channels_ = datum.channels();
  datum_length_ = datum.length();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.length() * datum.height() * datum.width();
  CHECK_GT(datum_height_, crop_size);
  CHECK_GT(datum_width_, crop_size);
  // check if we want to have mean
  if (this->layer_param_.data_param().has_mean_file()) {
    const string& mean_file = this->layer_param_.data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.length(), datum_length_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_length_, datum_height_, datum_width_);
  }


  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  if (output_labels_) {
    prefetch_label_->mutable_cpu_data();
  }
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void VolumeDataLayer<Dtype>::CreatePrefetchThread() {
  phase_ = Caffe::phase();
  const bool prefetch_needs_rand = (phase_ == Caffe::TRAIN) &&
      (this->layer_param_.data_param().mirror() ||
       this->layer_param_.data_param().crop_size());
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, VolumeDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void VolumeDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int VolumeDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype VolumeDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (output_labels_) {
    caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(VolumeDataLayer);

}  // namespace caffe
