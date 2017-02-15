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
#include <pthread.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
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
#include "caffe/video_data_layer.hpp"

using std::string;

namespace caffe {

const string win_name = "Live video";
const int waitKeyDelay = 10;
cv::VideoCapture video_cap_;

string GetStreamName(string path, int start_frm)
{
  string name, ext;
  size_t sep = path.find_last_of("\\/");
  if (sep != std::string::npos)
      path = path.substr(sep + 1, path.size() - sep - 1);
  size_t dot = path.find_last_of(".");
  if (dot != std::string::npos)
  {
      name = path.substr(0, dot);
      ext  = path.substr(dot, path.size() - dot);
  }
  else
  {
      name = path;
      ext  = "";
  }

  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(6) << start_frm;

  return name + '/' + oss.str();
}

template <typename Dtype>
void* VideoDataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  VideoDataLayer<Dtype>* layer = static_cast<VideoDataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);

  VolumeDatum datum;
  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label;
  if (layer->output_labels_) {
    top_label = layer->prefetch_label_->mutable_cpu_data();
  }
  const Dtype scale = layer->layer_param_.image_data_param().scale();
  const int batch_size = layer->layer_param_.image_data_param().batch_size();
  const int crop_size = layer->layer_param_.image_data_param().crop_size();
  const bool mirror = layer->layer_param_.image_data_param().mirror();
  const int new_length  = layer->layer_param_.image_data_param().new_length();
  const int new_height  = layer->layer_param_.image_data_param().new_height();
  const int new_width  = layer->layer_param_.image_data_param().new_width();
  const bool use_image = layer->layer_param_.image_data_param().use_image();
  const bool use_stream = layer->layer_param_.image_data_param().use_stream();
  const int sampling_rate = layer->layer_param_.image_data_param().sampling_rate();
  const bool use_temporal_jitter = layer->layer_param_.image_data_param().use_temporal_jitter();

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
    if (use_stream){
        if (!layer->file_list_[id].compare("CAMERA")){
          int camera_index = layer->start_frm_list_[id];
          if (!video_cap_.isOpened()){
              video_cap_.open(camera_index);
              if (!video_cap_.isOpened()){
                LOG(ERROR) << "Cannot open CAM " << camera_index;
                return static_cast<void*>(NULL);
              }
          }

          vector<cv::Mat> live_imgs;
          for (int i=0; i<new_length; i++){
              cv::Mat img;
              int sampling_count = 0;
              while (sampling_count < sampling_rate){
                  cv::waitKey(waitKeyDelay);
                  video_cap_.read(img);
                  if (!img.data){
                      LOG(ERROR) << "No data at CAM " << camera_index;
                      return static_cast<void*>(NULL);
                  }
                  imshow(win_name, img);
                  sampling_count++;
              }
              live_imgs.push_back(img);
              {
                cv::Mat img_disp = img.clone();
                copyMakeBorder( img_disp, img_disp, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0,255,0) );
                imshow(win_name, img_disp);
              }
          }
          int start_frm = layer->interval_jump_++ * layer->interval_list_[id];
          layer->stream_names_.push_back(GetStreamName("CAMERA", start_frm));
          read_status = ReadImageVectorToVolumeDatum(live_imgs,
                          layer->label_list_[id], new_length, new_height, new_width, &datum);

        } else {
          int start_frm = layer->start_frm_list_[id] + layer->interval_jump_++ * layer->interval_list_[id];
          int end_frm = start_frm + new_length * sampling_rate;
          if (end_frm > layer->num_of_frames_list_[id]) {
            LOG(INFO) << "Done extracting features from " << layer->file_list_[id];
            read_status = false;
          } else {
            layer->stream_names_.push_back(GetStreamName(layer->file_list_[id].c_str(), start_frm));
            read_status = ReadVideoToVolumeDatum(layer->file_list_[id].c_str(), start_frm,
                        layer->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
          }
        }
    }
    else if (!use_image){
    	if (!use_temporal_jitter){
    		read_status = ReadVideoToVolumeDatum(layer->file_list_[id].c_str(), layer->start_frm_list_[id],
    	    		layer->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
    	}else{
    		read_status = ReadVideoToVolumeDatum(layer->file_list_[id].c_str(), -1,
    	    		layer->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
    	}
    }
    else {
    	if (!use_temporal_jitter) {
    		read_status = ReadImageSequenceToVolumeDatum(layer->file_list_[id].c_str(), layer->start_frm_list_[id],
    	    	    		layer->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
    	} else {
    		int num_of_frames = layer->start_frm_list_[id];
    		int use_start_frame;
    		if (num_of_frames<new_length*sampling_rate){
    			LOG(INFO) << "not enough frames; having " << num_of_frames;
    			read_status = false;
    		} else {
    			if (layer->phase_ == Caffe::TRAIN)
    				use_start_frame = layer->PrefetchRand()%(num_of_frames-new_length*sampling_rate+1)+1;
    			else
    				use_start_frame = 0;

    			read_status = ReadImageSequenceToVolumeDatum(layer->file_list_[id].c_str(), use_start_frame,
    			    	    	    		layer->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
    		}
    	}
    }

    if (layer->phase_ == Caffe::TEST && !use_stream){
    	CHECK(read_status) << "Testing must not miss any example";
    }

    if (!read_status) {
    	//LOG(ERROR) << "cannot read " << layer->file_list_[id];
        layer->lines_id_++;
        layer->interval_jump_ = 0;
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
    //LOG(INFO) << "--> " << item_id;
    //LOG(INFO) << "label " << datum.label();
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
      // LOG(INFO) << "fetching label" << datum.label() << std::endl;
    }

    if (!use_stream) {
      layer->lines_id_++;
    }
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
VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_GE(top->size(), 1) << "Data Layer takes at least one blob as output.";
  CHECK_LE(top->size(), 2) << "Data Layer takes at most two blobs as output.";
  if (top->size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  const int new_length  = this->layer_param_.image_data_param().new_length();
  const int new_height  = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const int sampling_rate = this->layer_param_.image_data_param().sampling_rate();
  CHECK(new_length > 0) << "new length need to be positive";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  const bool use_temporal_jitter = this->layer_param_.image_data_param().use_temporal_jitter();
  const bool use_image = this->layer_param_.image_data_param().use_image();
  const bool use_stream = this->layer_param_.image_data_param().use_stream();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  int count = 0;
  string filename;
  int start_frm, num_of_frames, interval, label;


  if (use_stream){
    while (infile >> filename >> start_frm >> interval >> label) {
        num_of_frames = -1;
        if (filename.compare("CAMERA")){
          cv::VideoCapture cap;
          CHECK(cap.open(filename)) << "Cannot open " << filename;
          num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
        }
        file_list_.push_back(filename);
        start_frm_list_.push_back(start_frm);
        num_of_frames_list_.push_back(num_of_frames);
        interval_list_.push_back(interval);
        label_list_.push_back(label);
        shuffle_index_.push_back(count);
        count++;
    }
  } else if ((!use_image) && use_temporal_jitter){
	  while (infile >> filename >> label) {
		  file_list_.push_back(filename);
		  label_list_.push_back(label);
		  shuffle_index_.push_back(count);
		  count++;
	  }
  } else {
	  while (infile >> filename >> start_frm >> label) {
		  file_list_.push_back(filename);
		  start_frm_list_.push_back(start_frm);
		  label_list_.push_back(label);
		  shuffle_index_.push_back(count);
		  count++;
	  }
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
  LOG(INFO) << "A total of " << shuffle_index_.size() << " video chunks.";

  lines_id_ = 0;
  interval_jump_ = 0;

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(shuffle_index_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read a data point, and use it to initialize the top blob.
  VolumeDatum datum;
  int id = shuffle_index_[lines_id_];
  if (use_stream){
      if (!file_list_[id].compare("CAMERA")){
        int camera_index = start_frm_list_[id];
        if (!video_cap_.isOpened()){
            video_cap_.open(camera_index);
            if (!video_cap_.isOpened()){
              LOG(ERROR) << "Cannot open CAM " << camera_index;
              return;
            }
        }

        cv::namedWindow(win_name, CV_WINDOW_AUTOSIZE);
        vector<cv::Mat> live_imgs;
        for (int i=0; i<new_length; i++){
            cv::Mat img;
            int sampling_count = 0;
            while (sampling_count < sampling_rate){
                cv::waitKey(waitKeyDelay);
                video_cap_.read(img);
                if (!img.data){
                    LOG(ERROR) << "No data at CAM " << camera_index;
                    return;
                }
                imshow(win_name, img);
                sampling_count++;
            }
            live_imgs.push_back(img);
            {
              cv::Mat img_disp = img.clone();
              copyMakeBorder( img_disp, img_disp, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0,255,0) );
              imshow(win_name, img_disp);
            }
        }
        CHECK(ReadImageVectorToVolumeDatum(live_imgs, label_list_[id],
                                   new_length, new_height, new_width, &datum));
      }
      else
          CHECK(ReadVideoToVolumeDatum(file_list_[id].c_str(), start_frm_list_[id], label_list_[id],
                                   new_length, new_height, new_width, sampling_rate, &datum));
  }
  else if (!use_image){
      if (use_temporal_jitter){
		  srand (time(NULL));
		  CHECK(ReadVideoToVolumeDatum(file_list_[0].c_str(), 0, label_list_[0],
		                             new_length, new_height, new_width, sampling_rate, &datum));
	  }
	  else
		  CHECK(ReadVideoToVolumeDatum(file_list_[id].c_str(), start_frm_list_[id], label_list_[id],
                           	   	   new_length, new_height, new_width, sampling_rate, &datum));
  }
  else{
	  LOG(INFO) << "read video from " << file_list_[id].c_str();
	  CHECK(ReadImageSequenceToVolumeDatum(file_list_[id].c_str(), start_frm_list_[id], label_list_[id],
	                             new_length, new_height, new_width, sampling_rate, &datum));
  }

  // image
  int crop_size = this->layer_param_.image_data_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.image_data_param().batch_size(),
                       datum.channels(), datum.length(), crop_size, crop_size);
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.image_data_param().batch_size(), datum.channels(), datum.length(),
        crop_size, crop_size));
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.image_data_param().batch_size(), datum.channels(), datum.length(),
        datum.height(), datum.width());
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.image_data_param().batch_size(), datum.channels(), datum.length(),
        datum.height(), datum.width()));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->length() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.image_data_param().batch_size(), 1, 1, 1, 1);
    prefetch_label_.reset(
        new Blob<Dtype>(this->layer_param_.image_data_param().batch_size(), 1, 1, 1, 1));
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
  if (this->layer_param_.image_data_param().has_mean_file()) {
    const string& mean_file = this->layer_param_.image_data_param().mean_file();
    LOG(INFO) << "Loading mean file from " << mean_file;
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
    if (this->layer_param_.image_data_param().has_mean_value()){
    	LOG(INFO) << "Using mean value of " << this->layer_param_.image_data_param().mean_value();
    	caffe::caffe_set(data_mean_.count(), (Dtype)this->layer_param_.image_data_param().mean_value(),
    				(Dtype*)data_mean_.mutable_cpu_data());
    }
  }


  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not do so.
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
void VideoDataLayer<Dtype>::CreatePrefetchThread() {
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
  CHECK(!pthread_create(&thread_, NULL, VideoDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void VideoDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int VideoDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype VideoDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

template <typename Dtype>
vector<string> VideoDataLayer<Dtype>::pop_stream_names(int num) {
  vector<string> popped_names;
  for (int i=0; i<num; i++){
    if (stream_names_.empty())
      break;
    popped_names.push_back(stream_names_.front());
    stream_names_.pop_front();
  }
  return popped_names;
}

INSTANTIATE_CLASS(VideoDataLayer);

}  // namespace caffe
