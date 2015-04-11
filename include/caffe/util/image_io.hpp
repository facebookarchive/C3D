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


#ifndef IMAGE_IO_HPP_
#define IMAGE_IO_HPP_


#include <string>

#include "google/protobuf/message.h"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"

using std::string;
using ::google::protobuf::Message;

namespace caffe {


void ImageToBuffer(const cv::Mat* img, char* buffer);
void ImageChannelToBuffer(const cv::Mat* img, char* buffer, int c);
void BufferToGrayImage(const char* buffer, const int h, const int w, cv::Mat* img);
void BufferToColorImage(const char* buffer, const int height, const int width, cv::Mat* img);


bool ReadVideoToVolumeDatum(const char* filename, const int start_frm, const int label,
		const int length, const int height, const int width, const int sampling_rate, VolumeDatum* datum);

inline bool ReadVideoToVolumeDatum(const char* filename, const int start_frm, const int label,
		const int length, const int sampling_rate, VolumeDatum* datum){
	return ReadVideoToVolumeDatum(filename, start_frm, label, length, 0, 0, sampling_rate, datum);
}

bool ReadImageSequenceToVolumeDatum(const char* img_dir, const int start_frm, const int label,
		const int length, const int height, const int width, const int sampling_rate, VolumeDatum* datum);

inline bool ReadImageSequenceToVolumeDatum(const char* img_dir, const int start_frm, const int label,
		const int length, const int sampling_rate, VolumeDatum* datum){
	return ReadImageSequenceToVolumeDatum(img_dir, start_frm, label, length, 0, 0, sampling_rate, datum);
}

template <typename Dtype>
bool load_blob_from_binary(const string fn_blob, Blob<Dtype>* blob);

template <typename Dtype>
bool save_blob_to_binary(Blob<Dtype>* blob, const string fn_blob, int num_index);

template <typename Dtype>
inline bool save_blob_to_binary(Blob<Dtype>* blob, const string fn_blob){
	return save_blob_to_binary(blob, fn_blob, -1);
}


}  // namespace caffe


#endif /* IMAGE_IO_HPP_ */
