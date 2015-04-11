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

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "fcntl.h"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
	BlobProto blob_proto;
	Blob<float> mean_data;
	ReadProtoFromBinaryFile(string(argv[1]), &blob_proto);
	mean_data.FromProto(blob_proto);
	save_blob_to_binary<float>(&mean_data, string(argv[2]), 0);
}
