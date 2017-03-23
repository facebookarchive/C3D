if [ ! -f c3d_resnet18_sports1m_r2_iter_2800000.caffemodel ];then
	wget https://www.dropbox.com/s/qqfrg6h44d4jb46/c3d_resnet18_sports1m_r2_iter_2800000.caffemodel
fi

GLOG_logtostderr=1 ../../build/tools/extract_image_features.bin c3d_resnet18_ucf101_feature_extraction.prototxt c3d_resnet18_sports1m_r2_iter_2800000.caffemodel 0 30 4970 ucf101_video_frame.prefix prob pool5 res5b
