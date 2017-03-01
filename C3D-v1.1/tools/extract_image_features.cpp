/*
 *
 *  Copyright (c) 2017, Facebook, Inc. All rights reserved.
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

#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/image_io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::VolumeDatum;
using caffe::Net;
using std::string;
using std::vector;
namespace db = caffe::db;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 7;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type"
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names separated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }
  char* net_proto = argv[1];
  char* pretrained_model = argv[2];
  int device_id = atoi(argv[3]);
  uint batch_size = atoi(argv[4]);
  uint num_mini_batches = atoi(argv[5]);
  char* fn_feat = argv[6];

  if (device_id >= 0) {
    LOG(INFO)<< "Using GPU";
    LOG(INFO) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  std::string pretrained_binary_proto(pretrained_model);

  std::string feature_extraction_proto(net_proto);
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  for (int i=7; i<argc; i++){
    CHECK(feature_extraction_net->has_blob(string(argv[i])))
      << "Unknown feature blob name " << string(argv[i])
      << " in the network " << string(net_proto);
  }


  LOG(INFO)<< "Extracting features for " << num_mini_batches << " batches";
  std::ifstream infile(fn_feat);
  string feat_prefix;
  std::vector<string> list_prefix;

  int image_index = 0;

  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward();
    list_prefix.clear();
    for (int n=0; n<batch_size; n++){
      if (infile >> feat_prefix)
        list_prefix.push_back(feat_prefix);
      else
        break;
    }

    if (list_prefix.empty())
      break;

    for (int k=7; k<argc; k++){
      const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
        ->blob_by_name(string(argv[k]));
      int num_features = feature_blob->shape(0);

      for (int n = 0; n < num_features; ++n) {
          if (list_prefix.size()>n){
            string fn_feat = list_prefix[n] + string(".") + string(argv[k]);
            save_blob_to_binary(feature_blob.get(), fn_feat, n);
          }
      }
    }
    image_index += list_prefix.size();
    if (batch_index % 100 == 0) {
        LOG(INFO)<< "Extracted features of " << image_index <<
            " images.";
    }
  }
  LOG(INFO)<< "Successfully extracted " << image_index << " features!";
  infile.close();
  return 0;
}
