#include <map>
#include <set>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/image_io.hpp"
#include <sstream>
#define SSTR( x ) static_cast< std::ostringstream & >( \
                ( std::ostringstream() << std::dec << x ) ).str()

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int get_weights(int argc, char** argv);

int main(int argc, char** argv) {
  return get_weights<float>(argc, argv);
}

template<typename Dtype>
int get_weights(int argc, char** argv) {
	const int num_required_args = 3;
	if (argc < num_required_args) {
		LOG(ERROR) << 
		"\nUsage: get_weights.bin <c3d_pre_trained_model> <output_dir>\n";
		return 1;
	}

    //char* net_proto = argv[1];
    char* pretrained_model = argv[1];
    char* output_dir = argv[2];
    //int device_id = atoi(argv[3]);
    //Caffe::set_phase(Caffe::TEST);
    //Caffe::set_mode(Caffe::CPU);
    //LOG(INFO) << "Using CPU";
    
    NetParameter param;
    ReadNetParamsFromBinaryFileOrDie(string(pretrained_model), &param);
    int num_source_layers = param.layers_size();
    for (int i = 0; i < num_source_layers; ++i) {
        const LayerParameter& source_layer = param.layers(i);
        const string& source_layer_name = source_layer.name();
        DLOG(INFO) << "saving source layer " << source_layer_name;
        //vector<shared_ptr<Blob<Dtype> > >& target_blobs
        LOG(INFO) << "blobs_size: " << source_layer.blobs_size();
        for (int j = 0; j < source_layer.blobs_size(); ++j) {
            LOG(INFO) << "n=" << source_layer.blobs(j).num() <<
                        ", c=" << source_layer.blobs(j).channels();
            if (source_layer.blobs(j).length()) //  only check new train model (C3D)
    	        LOG(INFO) << "l=" << source_layer.blobs(j).length() <<
                            ", h=" << source_layer.blobs(j).height() <<
                            ", w=" << source_layer.blobs(j).width();
            Blob<Dtype> blob_data;
            blob_data.FromProto(source_layer.blobs(j));
            string fn_weight = string(output_dir) + string("/") + 
                source_layer_name + string("_") + SSTR( j );
            save_blob_to_binary<Dtype>(&blob_data, fn_weight, -1); // filename, num_index < 0 to use blob_data.num()
        }
    }
    return 0;
}


