GLOG_logtostderr=1 ../../build/tools/caffe.bin test --model=train_resnet18_r2.prototxt --weights=c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel --stage='test-on-val' --iterations=2092 --gpu=0
