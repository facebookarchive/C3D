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


#include "caffe/caffe.hpp"
#include <cstring>
#include <cstdlib>
#include <vector>
#include <stdio.h>

/*
Extract all learned filters.
Input:
argv[1]: net proto
argv[2]: pre-trained model
argv[3]: output directory

argv[4] and on, names of the layers to extract

Output:
Result at output_dir, it will contain layer_name_<blob_id>.bin
*/

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int extraction_learned_filters(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);


  std::string proto(argv[1]);
  std::string binary_proto(argv[2]);
  std::string output_dir(argv[3]);

  boost::shared_ptr<Net<Dtype> > pretrained_net(
      new Net<Dtype>(proto, caffe::TEST));
  pretrained_net->CopyTrainedLayersFrom(binary_proto);

  for (int i=4; i<argc; i++) {
    LOG(ERROR) << "Exporting layer " << argv[i] << std::endl;
    shared_ptr<Layer<float> > layer_ptr =
      pretrained_net->layer_by_name(string(argv[i]));

    if (layer_ptr==NULL){
  	   LOG(ERROR) << "layer " << argv[i] << " does not exist" << std::endl;
  	   continue;
    }

    vector<shared_ptr<Blob<float> > > blobs = layer_ptr->blobs();

    if (blobs.size() == 0) {
      LOG(ERROR) << "layer " << argv[i] << " has no parameters" << std::endl;
      continue;
    }
    for (int j=0; j<blobs.size(); j++) {
      FILE *f;
      int header[5];
      header[0] = blobs[j]->shape(0);
      header[1] = blobs[j]->num_axes() > 1 ? blobs[j]->shape(1) : 1;
      header[2] = blobs[j]->num_axes() > 2 ? blobs[j]->shape(2) : 1;
      header[3] = blobs[j]->num_axes() > 3 ? blobs[j]->shape(3) : 1;
      header[4] = blobs[j]->num_axes() > 4 ? blobs[j]->shape(4) : 1;

      char buffer[16];
      sprintf(buffer, "%d", j);

      f = fopen(
        (output_dir +
         string(argv[i]) +
         string("_") +
         string(buffer) +
         string(".bin")
        ).c_str(), "wb");
      fwrite(header, 5, sizeof(int), f);
      fwrite(blobs[j]->mutable_cpu_data(), blobs[j]->count(), sizeof(float), f);
      fclose(f);
    }
  }

  return 0;
}

int main(int argc, char** argv){
  return extraction_learned_filters<float>(argc, argv);
}
