/*
 * test_convolution3d_layer.cpp
 *
 *  Created on: Jul 13, 2016
 *      Author: dutran
 */




#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/convolution3d_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class Convolution3DLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  Convolution3DLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
	Caffe::set_random_seed(1701);

    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    static const int shape_values[] = {2, 3, 5, 5, 5};
    vector<int> shape_size(shape_values, shape_values + 5);
    blob_bottom_->Reshape(shape_size);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~Convolution3DLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(Convolution3DLayerTest, TestDtypesAndDevices);

TYPED_TEST(Convolution3DLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Convolution3DParameter* convolution3d_param =
      layer_param.mutable_convolution3d_param();

  convolution3d_param->set_kernel_size(3);
  convolution3d_param->set_kernel_depth(3);
  convolution3d_param->set_stride(2);
  convolution3d_param->set_temporal_stride(2);
  convolution3d_param->set_num_output(6);

  shared_ptr<Convolution3DLayer<Dtype> > layer(
      new Convolution3DLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 6);
  EXPECT_EQ(this->blob_top_->shape(2), 2);
  EXPECT_EQ(this->blob_top_->shape(3), 2);
  EXPECT_EQ(this->blob_top_->shape(4), 2);

}

TYPED_TEST(Convolution3DLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;

  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  Convolution3DParameter* convolution3d_param =
  layer_param.mutable_convolution3d_param();

  convolution3d_param->set_kernel_size(3);
  convolution3d_param->set_kernel_depth(3);
  convolution3d_param->set_stride(2);
  convolution3d_param->set_temporal_stride(2);
  convolution3d_param->set_num_output(6);
  convolution3d_param->mutable_weight_filler()->set_type("constant");
  convolution3d_param->mutable_weight_filler()->set_value(1);
  convolution3d_param->mutable_bias_filler()->set_type("constant");
  convolution3d_param->mutable_bias_filler()->set_value(0.1);

  shared_ptr<Layer<Dtype> > layer(
     new Convolution3DLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // After the convolution, the output should all have output values 81.1
  const Dtype* top_data;
  top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], 81.1, 1e-4);
  }
}

TYPED_TEST(Convolution3DLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  Convolution3DParameter* convolution3d_param =
      layer_param.mutable_convolution3d_param();
  convolution3d_param->set_kernel_size(3);
  convolution3d_param->set_kernel_depth(3);
  convolution3d_param->set_stride(2);
  convolution3d_param->set_temporal_stride(2);
  convolution3d_param->set_num_output(2);
  convolution3d_param->mutable_weight_filler()->set_type("gaussian");
  convolution3d_param->mutable_bias_filler()->set_type("gaussian");

  Convolution3DLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);

}


}  // namespace caffe
