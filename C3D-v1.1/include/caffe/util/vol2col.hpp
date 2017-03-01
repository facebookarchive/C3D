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


#ifndef VOL2COL_HPP_
#define VOL2COL_HPP_


namespace caffe {

template <typename Dtype>
void vol2col_cpu(const Dtype* data_im, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, Dtype* data_col);

template <typename Dtype>
void col2vol_cpu(const Dtype* data_col, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, Dtype* data_im);

template <typename Dtype>
void vol2col_gpu(const Dtype* data_im, const int channels, const int length,
	    const int height, const int width, const int ksize, const int kdepth, const int pad,
	    const int temporal_pad, const int stride, const int temporal_stride, Dtype* data_col);

template <typename Dtype>
void col2vol_gpu(const Dtype* data_col, const int channels, const int length,
	    const int height, const int width, const int ksize, const int kdepth, const int pad,
	    const int temporal_pad, const int stride, const int temporal_stride, Dtype* data_im);

}  // namespace caffe


#endif /* VOL2COL_HPP_ */
