#include "caffe/caffe.hpp"
#include <cstring>
#include <cstdlib>
#include <vector>
#include <stdio.h>

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv){


	Net<float> caffe_test_net(argv[1]);
	caffe_test_net.CopyTrainedLayersFrom(argv[2]);
  std::string output_dir(argv[3]);

  for (int i=0; i<caffe_test_net.layers().size(); i++) {

    const string& layer_name = caffe_test_net.layer_names()[i];

    LOG(ERROR) << "Exporting layer " << layer_name << std::endl;
    Layer<float>* layer_ptr = caffe_test_net.layers()[i].get();

	  vector<shared_ptr<Blob<float> > > blobs = layer_ptr->blobs();

    for (int j=0; j<blobs.size(); j++) {
        char buffer[16];
        sprintf(buffer, "%d", j);
        FILE *f;
        // write the shape file
        int header[5];
        long int c;
        header[0] = blobs[j]->num();
        header[1] = blobs[j]->channels();
        header[2] = blobs[j]->length();
        header[3] = blobs[j]->height();
        header[4] = blobs[j]->width();
        c = header[0] * header[1] * header[2] * header[3] * header[4];
        f = fopen(
          (output_dir +
           layer_name +
           string("_") +
           string(buffer) +
           string(".shape")
          ).c_str(), "wb");
        fwrite(header, 5, sizeof(int), f);
        fclose(f);

        CHECK_EQ(blobs[j]->count(), c) << "Wrong size";
        // write the value file
        f = fopen(
          (output_dir +
           layer_name +
           string("_") +
           string(buffer) +
           string(".value")
          ).c_str(), "wb");
        fwrite(blobs[j]->mutable_cpu_data(), blobs[j]->count(), sizeof(float), f);
        fclose(f);
    }
  }
	return 0;
}
