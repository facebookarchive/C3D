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

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <stdio.h>

#include "caffe/common.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/proto/caffe.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;


namespace caffe {


void ImageToBuffer(const cv::Mat* img, char* buffer){
	int idx = 0;
	for (int c = 0; c < 3; ++c) {
	  for (int h = 0; h < img->rows; ++h) {
		for (int w = 0; w < img->cols; ++w) {
			buffer[idx++] = img->at<cv::Vec3b>(h, w)[c];
		}
	  }
	}
}
void ImageChannelToBuffer(const cv::Mat* img, char* buffer, int c){
    int idx = 0;
	for (int h = 0; h < img->rows; ++h) {
	    for (int w = 0; w < img->cols; ++w) {
		    buffer[idx++] = img->at<cv::Vec3b>(h, w)[c];
		}
	}
}

void GrayImageToBuffer(const cv::Mat* img, char* buffer){
	int idx = 0;
    for (int h = 0; h < img->rows; ++h) {
	  for (int w = 0; w < img->cols; ++w) {
		buffer[idx++] = img->at<unsigned char>(h, w);
	  }
	}
}
void BufferToGrayImage(const char* buffer, const int height, const int width, cv::Mat* img){
	int idx = 0;
	img->create(height, width, CV_8U);
    for (int h = 0; h < height; ++h) {
	  for (int w = 0; w < width; ++w) {
		img->at<unsigned char>(h, w) = buffer[idx++];
	  }
	}
}
void BufferToColorImage(const char* buffer, const int height, const int width, cv::Mat* img){
	img->create(height, width, CV_8UC3);
	for (int c=0; c<3; c++) {
		for (int h = 0; h < height; ++h) {
		  for (int w = 0; w < width; ++w) {
			img->at<cv::Vec3b>(h, w)[c] = buffer[c * width * height + h * width + w];
		  }
		}
	}
}


bool ReadVideoToVolumeDatum(const char* filename, const int start_frm, const int label,
		const int length, const int height, const int width, const int sampling_rate, VolumeDatum* datum){
	cv::VideoCapture cap;
	cv::Mat img, img_origin;
	char *buffer;
	unsigned long int offset, channel_size, image_size, data_size;
	int use_start_frm = start_frm;

	cap.open(filename);
	if (!cap.isOpened()){
		LOG(ERROR) << "Cannot open " << filename;
		return false;
	}

	datum->set_channels(3);
	datum->set_length(length);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	if (num_of_frames<length*sampling_rate){
		LOG(INFO) << "not enough frames; having " << num_of_frames;
		return false;
	}
	if (start_frm < 0){
		use_start_frm = rand()%(num_of_frames-length*sampling_rate+1);
	}

	offset = 0;
	CHECK_GE(use_start_frm, 0) << "start frame must be greater or equal to 0";

	cap.set(CV_CAP_PROP_POS_FRAMES, use_start_frm);
	int end_frm = use_start_frm + length * sampling_rate;
	CHECK_LE(end_frm, num_of_frames) << "end frame must less or equal to num of frames";

	for (int i=use_start_frm; i<end_frm; i+=sampling_rate){
		if (sampling_rate > 1)
			cap.set(CV_CAP_PROP_POS_FRAMES, i);
		if (height > 0 && width > 0){
			cap.read(img_origin);
			if (!img_origin.data){
				LOG(INFO) << "No data at frame " << i;
				return false;
			}
			cv::resize(img_origin, img, cv::Size(width, height));
		}
		else
			cap.read(img);
		if (!img.data){
			LOG(ERROR) << "Could not open or find file " << filename;
			return false;
		}

		if (i==use_start_frm){
			datum->set_height(img.rows);
			datum->set_width(img.cols);
			image_size = img.rows * img.cols;
			channel_size = image_size * length;
			data_size = channel_size * 3;
			buffer = new char[data_size];
		}
		for (int c=0; c<3; c++){
			ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
		}
		offset += image_size;
	}
	CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
	cap.release();
 	return true;
}

bool ReadImageSequenceToVolumeDatum(const char* img_dir, const int start_frm, const int label,
		const int length, const int height, const int width, const int sampling_rate, VolumeDatum* datum){
	char fn_im[256];
	cv::Mat img, img_origin;
	char *buffer;
	unsigned long int offset, channel_size, image_size, data_size;

	datum->set_channels(3);
	datum->set_length(length);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	offset = 0;
	int end_frm = start_frm + length * sampling_rate;
	for (int i=start_frm; i<end_frm; i+=sampling_rate){
		sprintf(fn_im, "%s/%06d.jpg", img_dir, i);
		if (height > 0 && width > 0) {
		    img_origin = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
		    if (!img_origin.data)
		    	return false;
		    cv::resize(img_origin, img, cv::Size(width, height));
		    img_origin.release();
		} else {
		  img = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
		}

		if (!img.data){
			LOG(ERROR) << "Could not open or find file " << fn_im;
			return false;
		}

		if (i==start_frm){
			datum->set_height(img.rows);
			datum->set_width(img.cols);
			image_size = img.rows * img.cols;
			channel_size = image_size * length;
			data_size = channel_size * 3;
			buffer = new char[data_size];
		}
		for (int c=0; c<3; c++){
			ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
		}
		offset += image_size;
	}
	CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
 	return true;
}

template <>
bool load_blob_from_binary<float>(const string fn_blob, Blob<float>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	float* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);
	blob->Reshape(n, c, l, h, w);
	buff = blob->mutable_cpu_data();

	fread(buff, sizeof(float), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool load_blob_from_binary<double>(const string fn_blob, Blob<double>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	double* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);
	blob->Reshape(n, c, l, h, w);
	buff = blob->mutable_cpu_data();

	fread(buff, sizeof(double), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool load_blob_from_uint8_binary<float>(const string fn_blob, Blob<float>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	float* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);
	blob->Reshape(n, c, l, h, w);
	buff = blob->mutable_cpu_data();

	int count = n * c * l * h * w;
	unsigned char* temp_buff = new unsigned char[count];

	fread(temp_buff, sizeof(unsigned char), count, f);
	fclose(f);

	for (int i = 0; i < count; i++)
		buff[i] = (float)temp_buff[i];

	delete []temp_buff;
	return true;
}

template <>
bool load_blob_from_uint8_binary<double>(const string fn_blob, Blob<double>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	double* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);
	blob->Reshape(n, c, l, h, w);
	buff = blob->mutable_cpu_data();


	int count = n * c * l * h * w;
	unsigned char* temp_buff = new unsigned char[count];

	fread(temp_buff, sizeof(unsigned char), count, f);
	fclose(f);

	for (int i = 0; i < count; i++)
		buff[i] = (double)temp_buff[i];

	delete []temp_buff;
	return true;
}


template <>
bool save_blob_to_binary<float>(Blob<float>* blob, const string fn_blob, int num_index){
	FILE *f;
	float *buff;
	int n, c, l, w, h;
	f = fopen(fn_blob.c_str(), "wb");
	if (f==NULL)
		return false;

	if (num_index<0){
		n = blob->num();
		buff = blob->mutable_cpu_data();
	}else{
		n = 1;
		buff = blob->mutable_cpu_data() + blob->offset(num_index);
	}
	c = blob->channels();
	l = blob->length();
	h = blob->height();
	w = blob->width();

	fwrite(&n, sizeof(int), 1, f);
	fwrite(&c, sizeof(int), 1, f);
	fwrite(&l, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(&w, sizeof(int), 1, f);
	fwrite(buff, sizeof(float), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool save_blob_to_binary<double>(Blob<double>* blob, const string fn_blob, int num_index){
	FILE *f;
	double *buff;
	int n, c, l, w, h;
	f = fopen(fn_blob.c_str(), "wb");
	if (f==NULL)
		return false;

	if (num_index<0){
		n = blob->num();
		buff = blob->mutable_cpu_data();
	}else{
		n = 1;
		buff = blob->mutable_cpu_data() + blob->offset(num_index);
	}
	c = blob->channels();
	l = blob->length();
	h = blob->height();
	w = blob->width();

	fwrite(&n, sizeof(int), 1, f);
	fwrite(&c, sizeof(int), 1, f);
	fwrite(&l, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(&w, sizeof(int), 1, f);
	fwrite(buff, sizeof(double), n * c * l * h * w, f);
	fclose(f);
	return true;
}


}
