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

#include <glog/logging.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::max;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 5) {
    LOG(ERROR) << "Usage: compute_image_mean input_list new_height new_width output_file [dropping_rate]";
    return 1;
  }

  char* fn_list = argv[1];
  const int height = atoi(argv[2]);
  const int width = atoi(argv[3]);
  char* fn_output = argv[4];

  int sampling_rate = 1;
  if (argc >= 6){
	  sampling_rate = atoi(argv[5]);
	  LOG(INFO) << "using sampling rate " << sampling_rate;
  }

  Datum datum;
  BlobProto sum_blob;
  int count = 0;

  std::ifstream infile(fn_list);
  string fn_frm;
  int label;
  infile >> fn_frm >> label;

  ReadImageToDatum(fn_frm, label, height, width, &datum);

  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_length(1);
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());

  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());

  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }

  LOG(INFO) << "Starting Iteration";
  int i = 0;
  while (infile >> fn_frm >> label) {
	  i++;
	  if (i % sampling_rate!=0){
		  continue;
	  }
	ReadImageToDatum(fn_frm, label, height, width, &datum);
    const string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(), datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }

  infile.close();

  if (count % 10000 != 0) {
    LOG(ERROR) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  LOG(INFO) << "Write to " << fn_output;
  WriteProtoToBinaryFile(sum_blob, fn_output);

  return 0;
}
