#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/image_io.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class VideoDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  VideoDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file using frames.
    MakeTempFilename(&filename_jpg_);
    std::ofstream outfile_jpg(filename_jpg_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_jpg_;
    for (int i = 0; i < 5; ++i) {
      outfile_jpg << EXAMPLES_SOURCE_DIR "videos/jpg/v_ApplyEyeMakeup_g01_c01/ " << 16*i+1 << " " << i << std::endl;
    }
    outfile_jpg.close();

    // Create test input file using frames.
    MakeTempFilename(&filename_avi_);
    std::ofstream outfile_avi(filename_avi_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_avi_;
    for (int i = 0; i < 5; ++i) {
      outfile_avi << EXAMPLES_SOURCE_DIR "videos/avi/v_ApplyEyeMakeup_g01_c01.avi " << 16*i << " " << i << std::endl;
    }
    outfile_avi.close();

    // Create test input file using frames.
    MakeTempFilename(&filename_avi_jitter_);
    std::ofstream outfile_avi_jitter(filename_avi_jitter_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_avi_jitter_;
    for (int i = 0; i < 5; ++i) {
      outfile_avi_jitter << EXAMPLES_SOURCE_DIR "videos/avi/v_ApplyEyeMakeup_g01_c01.avi " << i << std::endl;
    }
    outfile_avi_jitter.close();
  }

  virtual ~VideoDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_jpg_;
  string filename_avi_;
  string filename_avi_jitter_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(VideoDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(VideoDataLayerTest, TestReadJpg) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();
  video_data_param->set_batch_size(5);
  video_data_param->set_source(this->filename_jpg_.c_str());
  video_data_param->set_new_length(16);
  video_data_param->set_use_image(true);
  video_data_param->set_show_data(false);
  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 5);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 240);
  EXPECT_EQ(this->blob_top_data_->shape(4), 320);
  EXPECT_EQ(this->blob_top_label_->shape(0), 5);

  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}


TYPED_TEST(VideoDataLayerTest, TestReadAvi) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();
  video_data_param->set_batch_size(5);
  video_data_param->set_source(this->filename_avi_.c_str());
  video_data_param->set_new_length(16);
  video_data_param->set_use_image(false);
  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 5);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 240);
  EXPECT_EQ(this->blob_top_data_->shape(4), 320);
  EXPECT_EQ(this->blob_top_label_->shape(0), 5);
  EXPECT_EQ(this->blob_top_label_->shape(0), 5);

  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(VideoDataLayerTest, TestReadAviJitter) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();
  video_data_param->set_batch_size(5);
  video_data_param->set_source(this->filename_avi_jitter_.c_str());
  video_data_param->set_new_length(16);
  video_data_param->set_use_image(false);
  video_data_param->set_use_temporal_jitter(true);
  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 5);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 240);
  EXPECT_EQ(this->blob_top_data_->shape(4), 320);
  EXPECT_EQ(this->blob_top_label_->shape(0), 5);

  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}



TYPED_TEST(VideoDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();
  video_data_param->set_batch_size(5);
  video_data_param->set_source(this->filename_jpg_.c_str());
  video_data_param->set_new_length(16);
  video_data_param->set_new_height(128);
  video_data_param->set_new_width(171);
  video_data_param->set_use_image(true);
  video_data_param->set_shuffle(false);
  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 5);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 128);
  EXPECT_EQ(this->blob_top_data_->shape(4), 171);
  EXPECT_EQ(this->blob_top_label_->shape(0), 5);

  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}


TYPED_TEST(VideoDataLayerTest, TestCropSize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  TransformationParameter* transform_param = param.mutable_transform_param();;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();

  video_data_param->set_batch_size(5);
  video_data_param->set_source(this->filename_jpg_.c_str());
  video_data_param->set_new_length(16);
  video_data_param->set_new_height(128);
  video_data_param->set_new_width(171);

  transform_param->set_crop_size(112);
  video_data_param->set_use_image(true);
  video_data_param->set_shuffle(false);
  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 5);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 112);
  EXPECT_EQ(this->blob_top_data_->shape(4), 112);
  EXPECT_EQ(this->blob_top_label_->shape(0), 5);

  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
	  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	  for (int i = 0; i < 5; ++i) {
		  EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
	  }
  }
}



TYPED_TEST(VideoDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();
  video_data_param->set_batch_size(5);
  video_data_param->set_source(this->filename_jpg_.c_str());
  video_data_param->set_shuffle(true);
  video_data_param->set_use_image(true);
  video_data_param->set_new_length(16);
  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 5);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 240);
  EXPECT_EQ(this->blob_top_data_->shape(4), 320);
  EXPECT_EQ(this->blob_top_label_->shape(0), 5);

  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }
}




}  // namespace caffe
#endif  // USE_OPENCV
