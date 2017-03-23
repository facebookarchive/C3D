mkdir -p LOG_TRAIN
GLOG_log_dir="./LOG_TRAIN/" ../../build/tools/caffe.bin train --solver=solver.prototxt --gpu=0
