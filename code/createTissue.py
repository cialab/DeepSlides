from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os
import glob
from scipy.misc import imread
from  models import create_generator
#version 2.0 generated on 3/16/2018

sess = tf.InteractiveSession()
debug=True



def readImage(imgpath):
    img=imread(imgpath)
    img = img / 255
    img = img * 2 - 1
    return img




def analyze(imgpath,model,inputs,patchSize):
    imgname=os.path.basename(imgpath)
    starttime=time.time()
    img = readImage(imgpath)
    img = img[np.newaxis, :]
    print ('prediction ')
    prediction = sess.run(model, feed_dict={inputs: img})
    print('ok!')
    result = prediction[0, :, :, :]
    result = (result + 1) / 2
    plt.imsave('output/' + imgname +'_result.png', np.uint8(result*255))
    endtime = time.time()
    elapsedtime=endtime-starttime
    print ('elapsed time ' +  str(elapsedtime))



def main():
    searchPath="Testdata/*.png"
    #Please use same patchsize for all images.
    patchSize=1280
    inputs = tf.placeholder(tf.float32, [1, patchSize, patchSize, 3])
    with tf.variable_scope("generator") as scope:
        out_channels = 3
        outputs = create_generator(inputs, out_channels)
    saver = tf.train.Saver()
    saver.restore(sess, "artificialKi67/compact.ckpt")
    files = glob.glob(searchPath)
    print (files)
    for myfile in files:
        print (myfile)
        analyze(myfile,outputs,inputs,patchSize)


if __name__ == "__main__":
    main()
