/*
 *  video_segmentation_data_layer.hpp
 *
 *  Created on: Jul 6, 2015
 *      Author: trandu
 */

#ifndef VIDEO_SEGMENTATION_DATA_LAYER_HPP_
#define VIDEO_SEGMENTATION_DATA_LAYER_HPP_


#include <string>
#include <utility>
#include <vector>

#include "pthread.h"
#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

template <typename Dtype>
void* VideoSegmentationDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class VideoSegmentationDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* VideoSegmentationDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit VideoSegmentationDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~VideoSegmentationDataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<string> file_list_;
  vector<int> shuffle_index_;
  int lines_id_;

  int datum_channels_;
  int datum_length_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_truth_;
  Blob<Dtype> data_mean_;
  Caffe::Phase phase_;
};

}


#endif /* VIDEO_SEGMENTATION_DATA_LAYER_HPP_ */
