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
#include "caffe/util/image_io.hpp"
#include "caffe/util/io.hpp"

using caffe::VolumeDatum;
using caffe::BlobProto;
using std::max;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 7) {
    LOG(ERROR) << "Usage: compute_volume_mean_from_list_videos input_chunk_list length height width sampling_rate output_file [dropping rate]";
    return 1;
  }

  char* fn_list = argv[1];
  const int length = atoi(argv[2]);
  const int height = atoi(argv[3]);
  const int width = atoi(argv[4]);
  const int sampling_rate = atoi(argv[5]);
  char* fn_output = argv[6];

  int dropping_rate = 1;
  if (argc >= 8){
	  dropping_rate = atoi(argv[7]);
	  LOG(INFO) << "using dropping rate " << dropping_rate;
  }

  VolumeDatum datum;
  BlobProto sum_blob;
  int count = 0;

  std::ifstream infile(fn_list);
  string frm_dir;
  int label, start_frm;
  infile >> frm_dir >> start_frm >> label;

  ReadVideoToVolumeDatum(frm_dir.c_str(), start_frm, label, length, height, 
			 width, sampling_rate, &datum);

  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_length(datum.length());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.length() * datum.height() * datum.width();
  int size_in_datum = datum.data().size();
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }


  LOG(INFO) << "Starting Iteration";
  int c = 0;
  while (infile >> frm_dir >> start_frm >> label) {
	  c++;
	  if (c % dropping_rate!=0){
		  continue;
	  }

	  ReadVideoToVolumeDatum(frm_dir.c_str(), start_frm, label,
	    	                             length, height, width, sampling_rate, &datum);
	    const string& data = datum.data();
	    size_in_datum = datum.data().size();
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
