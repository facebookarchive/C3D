'''
A sample function to run classification using c3d model.
'''

import os
import numpy as np
import math
import cv2

def c3d_classify(
        vid_name,
        image_mean,
        net,
        start_frame,
        prob_layer='prob',
        multi_crop=False
        ):
    '''
    vid_name: a directory that contains extracted images (image_%05d.jpg)
    image_mean: (3,c3d_depth=16,height,width)-dim image mean
    net: a caffe network object
    start_frame: frame number to run classification (start_frame:start_frame+16)
                 note: this is 0-based whereas the first image file is
                 image_0001.jpg
    multi_crop: use mirroring / 4-corner + 1-center cropping
    '''

    # infer net params
    batch_size = net.blobs['data'].data.shape[0]
    c3d_depth = net.blobs['data'].data.shape[2]
    num_categories = net.blobs['prob'].data.shape[1]

    # init
    dims = (128,171,3,c3d_depth)
    rgb = np.zeros(shape=dims, dtype=np.float32)
    rgb_flip = np.zeros(shape=dims, dtype=np.float32)

    for i in range(c3d_depth):
        img_file = os.path.join(vid_name, 'image_{0:04d}.jpg'.format(start_frame+i))
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dims[1::-1])
        rgb[:,:,:,i] = img
        rgb_flip[:,:,:,i] = img[:,::-1,:]

    # substract mean
    image_mean = np.transpose(np.squeeze(image_mean), (2,3,0,1))
    rgb -= image_mean
    rgb_flip -= image_mean[:,::-1,:,:]

    if multi_crop:
        # crop (112-by-112) in upperleft, upperright, lowerleft, lowerright
        # corners and the center, for both original and flipped images
        rgb_1 = rgb[:112, :112, :,:]
        rgb_2 = rgb[:112, -112:, :,:]
        rgb_3 = rgb[8:120, 30:142, :,:]
        rgb_4 = rgb[-112:, :112, :,:]
        rgb_5 = rgb[-112:, -112:, :,:]
        rgb_f_1 = rgb_flip[:112, :112, :,:]
        rgb_f_2 = rgb_flip[:112, -112:, :,:]
        rgb_f_3 = rgb_flip[8:120, 30:142, :,:]
        rgb_f_4 = rgb_flip[-112:, :112, :,:]
        rgb_f_5 = rgb_flip[-112:, -112:, :,:]
        rgb = np.concatenate((rgb_1[...,np.newaxis],
                              rgb_2[...,np.newaxis],
                              rgb_3[...,np.newaxis],
                              rgb_4[...,np.newaxis],
                              rgb_5[...,np.newaxis],
                              rgb_f_1[...,np.newaxis],
                              rgb_f_2[...,np.newaxis],
                              rgb_f_3[...,np.newaxis],
                              rgb_f_4[...,np.newaxis],
                              rgb_f_5[...,np.newaxis]), axis=4)
    else:
        # crop (112-by-112) for both original/flipped images
        rgb_3 = rgb[8:120, 30:142, :,:]
        rgb_f_3 = rgb_flip[8:120, 30:142, :,:]
        rgb = np.concatenate((rgb_3[...,np.newaxis],
                              rgb_f_3[...,np.newaxis]), axis=4)

    # run classifications on batches
    prediction = np.zeros((num_categories,rgb.shape[4]))
    if rgb.shape[4] < batch_size:
        net.blobs['data'].data[:rgb.shape[4],:,:,:,:] = np.transpose(rgb, (4,2,3,0,1))
        output = net.forward()
        prediction = np.transpose(np.squeeze(output[prob_layer][:rgb.shape[4],:,:,:,:], axis=(2,3,4)))
    else:
        num_batches = int(math.ceil(float(rgb.shape[4])/batch_size))
        for bb in range(num_batches):
            span = range(batch_size*bb, min(rgb.shape[4],batch_size*(bb+1)))
            net.blobs['data'].data[...] = np.transpose(rgb[:,:,:,:,span], (4,2,3,0,1))
            output = net.forward()
            prediction[:, span] = np.transpose(np.squeeze(output[prob_layer], axis=(2,3,4)))
    return prediction
