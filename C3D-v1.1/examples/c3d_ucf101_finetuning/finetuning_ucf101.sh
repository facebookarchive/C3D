if [ ! -f c3d_resnet18_sports1m_r2_iter_2800000.caffemodel ]; then
	wget https://www.dropbox.com/s/qqfrg6h44d4jb46/c3d_resnet18_sports1m_r2_iter_2800000.caffemodel
fi

mkdir -p LOG_TRAIN

GLOG_log_dir="./LOG_TRAIN" ../../build/tools/caffe.bin train --solver=solver_r2.prototxt --weights=c3d_resnet18_sports1m_r2_iter_2800000.caffemodel --gpu=0
