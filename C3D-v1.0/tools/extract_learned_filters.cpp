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

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv){


	Net<float> caffe_test_net(argv[1]);
	caffe_test_net.CopyTrainedLayersFrom(argv[2]);

	shared_ptr<Layer<float> > layer_ptr = caffe_test_net.layer_by_name(string(argv[3]));

	if (layer_ptr==NULL){
		LOG(INFO) << "filer layer not exist" << std::endl;
		return 1;
	}

	vector<shared_ptr<Blob<float> > > blobs = layer_ptr->blobs();

	FILE *f;
	int header[5];
	int size;
	header[0] = blobs[0]->num();
	header[1] = blobs[0]->channels();
	header[2] = blobs[0]->length();
	header[3] = blobs[0]->height();
	header[4] = blobs[0]->width();

	f = fopen(argv[4], "wb");
	fwrite(header, 5, sizeof(int), f);
	fwrite(blobs[0]->mutable_cpu_data(), blobs[0]->count(), sizeof(float), f);
	if (blobs.size()>1)
		fwrite(blobs[1]->mutable_cpu_data(), blobs[1]->count(), sizeof(float), f);
	fclose(f);
	return 0;
}
