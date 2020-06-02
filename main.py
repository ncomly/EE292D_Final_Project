# coding: utf-8
import os
import time
import random
import logging
import argparse
import numpy as np
import glob

import tensorflow as tf

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Accuracy

from datetime import datetime
from tqdm import tqdm

from utils import *
from tsm_model import *

from PIL import Image

print("Process Number: ",os.getpid())

SEED = 1

tf.random.set_seed(SEED)
np.random.seed(SEED)


def lr_scheduler(epoch, lr):
    sleep_epochs = 10
    half = sleep_epochs 
    if epoch < sleep_epochs:
        learning_rate = lr
    else:
        learning_rate =  lr * tf.math.pow(0.5, (epoch-sleep_epochs+1)/float(half))
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

def check_dataset(dataset, dir_name=None, img_type=None):
    for i,example in enumerate(dataset.take(1)):
        #print(example)
        image, label = example
        print("img shape:", image.numpy().shape)
        #print("img: ", image.numpy())
        print("img label: ", label)
        image_shape = image.numpy().shape
        if dir_name != None:
            for j in range(image_shape[-4]):
                if img_type=='L':
                    img = Image.fromarray(image.numpy()[j,:,:,0], img_type)
                else:
                    img = Image.fromarray(image.numpy()[j,:,:,:], img_type)
                img.save(dir_name+'/test_img_'+str(j)+'.png')

def _parse_function(example):
    n_frames = 29
    num_depth = 3
    height = 96
    width = 96
    image_seq = []
    for image_count in range(n_frames):
        path = 'blob' + '/' + str(image_count)
    
        feature_description = {
            path: tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64)
        }
            
        features = tf.io.parse_single_example(example, feature_description)
        #image_buffer = tf.reshape(features[path], shape=[])
        #image = tf.io.decode_raw(image_buffer, tf.uint8)
        image = tf.reshape(features[path], shape=[])
        image = tf.io.decode_raw(image, tf.uint8)
        image = tf.reshape(image, tf.stack([height, width, num_depth]))
        image = tf.reshape(image, [1, height, width, num_depth])
#        image = image[:,:,:,0]# / tf.constant(255, shape=(1, height, width), dtype=tf.uint8) 
        image_seq.append(image)
    del example
    del image
#    image_seq = tf.reshape(image_seq, [1, n_frames, height, width])
    image_seq = tf.concat(image_seq, 0)
    label = features['label']
    # print("image: ", image_seq)
    # print("label: ", label)
    return image_seq, label

def video_left_right_flip(image):
    image_seq = tf.unstack(image)
    for i in range(len(image_seq)):
        image_seq[i] = tf.image.flip_left_right(image_seq[i])
    return tf.stack(image_seq)

def _train_preprocess_function(image, label):
    # Convert to grayscale
    image = tf.image.rgb_to_grayscale(image)
    # Take a random crop 88x88
    image = tf.image.random_crop(image, size=[29, 88, 88,1])
    # Randomly horizontal flip entire video
    random_sample = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
    condition = tf.less(random_sample, 0.5)
    image = tf.cond(condition, 
            lambda: video_left_right_flip(image),
            lambda: tf.identity(image))
    return image, label

def _test_preprocess_function(image, label):
    # Convert to grayscale
    image = tf.image.rgb_to_grayscale(image)
    # Take a center crop 88x88
    image = tf.image.crop_to_bounding_box(image, 4, 4, 88, 88)
    return image, label

def _normalize_function(image, label, n=130):
    # Normalize to [0,1]
    image = tf.cast(image, tf.float32) * (1./255.) 
    # Subtract mean, divide by std dev
    mean = 0.413621
    std = 0.1700239
    image = (image - mean) * (1./std)
    f, h, w, = image.shape[:-1]
    image = tf.reshape(image[:,:,:,0], [f,h,w,1])
    # print(f'label: {label}')
    y = tf.keras.backend.one_hot(label, n)
    # print(f'one hot: {y}')
    return image, y

def run(args, use_gpu=True):
    
    # saving
    save_path = os.path.join(os.getcwd(),'models')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    model = lipnext(inputDim=256, hiddenDim=512, nClasses=args.nClasses, frameLen=29, alpha=args.alpha)
    # model = tf.keras.Sequential([
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(args.nClasses)
    # ])

    if args.train==True:
        mode = "train"
    else:
        mode = "test"
    
#    train_list = glob.glob("./test_tfrecord_ACTUALLY_color/*.tfrecords") 
#    val_list = glob.glob("./test_tfrecord_ACTUALLY_color/*.tfrecords") 
#    test_list = glob.glob("./test_tfrecord_ACTUALLY_color/*.tfrecords") 
    #train_list = glob.glob("/mnt/disks/data/dataset/lipread_tfrecords/*/train/*.tfrecords")
    #val_list = glob.glob("/mnt/disks/data/dataset/lipread_tfrecords/*/val/*.tfrecords")
    #test_list = glob.glob("/mnt/disks/data/dataset/lipread_tfrecords/*/test/*.tfrecords")

    with open(args.labels) as f:
        labels = f.read().splitlines()
    train_list = []
    val_list = []
    test_list = []
    for word in labels:
        # print(word)
        train_list.extend(glob.glob(args.dataset + word + '/train/*.tfrecords'))
        val_list.extend(glob.glob(args.dataset + word + '/val/*.tfrecords'))
        test_list.extend(glob.glob(args.dataset + word + '/test/*.tfrecords'))
    # randomly shuffle *_list
    np.random.shuffle(train_list)
    np.random.shuffle(val_list)
    np.random.shuffle(test_list)
    
    if mode=="train":
        dataset = tf.data.TFRecordDataset(train_list)
        val_dataset = tf.data.TFRecordDataset(val_list)
    else:
        dataset = tf.data.TFRecordDataset(test_list)
    # print("raw_dataset: ", dataset)

    if mode=="train":
        dataset = dataset.map(_parse_function)
        val_dataset = val_dataset.map(_parse_function)
        #check_dataset(dataset, "train_test_images", 'RGB')
        
        dataset = dataset.map(_train_preprocess_function)
        val_dataset = val_dataset.map(_test_preprocess_function)
        
        dataset = dataset.map(lambda x, y: _normalize_function(x, y, args.nClasses))
        val_dataset = val_dataset.map(lambda x, y: _normalize_function(x, y, args.nClasses))
    
        dataset = dataset.batch(args.batch_size, drop_remainder=True)
        val_dataset = val_dataset.batch(args.batch_size, drop_remainder=True)
        # check_dataset(dataset)

        dataset = dataset.map(lambda x, y: (x[:,:,::2,::2,:], y))
        val_dataset = val_dataset.map(lambda x, y: (x[:,:,::2,::2,:], y))
        # check_dataset(dataset)

#        dataset = dataset.map(lambda x, y: ( tf.reshape(x, [-1, x.shape[2], x.shape[3], x.shape[4]]), y))
#        val_dataset = val_dataset.map(lambda x, y: ( tf.reshape(x, [-1, x.shape[2], x.shape[3], x.shape[4]]), y))
    
    else:
        dataset = dataset.map(_parse_function)
        #check_dataset(dataset, "test_test_images", 'RGB')

        dataset = dataset.map(_test_preprocess_function)
        
        dataset = dataset.map(lambda x, y: _normalize_function(x,y, args.nClasses))
        
        dataset = dataset.batch(args.batch_size, drop_remainder=True)

        dataset = dataset.map(lambda x, y: (x[:,:,::2,::2,:], y))
 
    #check_dataset(dataset, "test_test_images_processed", 'L')
    model.compile(optimizer=Adam(learning_rate = args.lr), 
           loss=CategoricalCrossentropy(from_logits=True),
           metrics=['accuracy', TopKCategoricalAccuracy(3),keras.metrics.CategoricalAccuracy() ]) 

    run_dir = args.save_path + datetime.now().strftime("%Y%m%d-%H%M%S")
    print(run_dir, "_-----------------------------------------------")
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
  	# tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        # Learning rate scheduler
        tf.keras.callbacks.LearningRateScheduler(lambda e: lr_scheduler(e, args.lr)),
        # Save checkpoints
        tf.keras.callbacks.ModelCheckpoint(
            filepath=run_dir+'/runs/{epoch}/checkpoint', 
            save_weights_only=True,
            save_best_only=True,
            monitor='val_categorical_accuracy',
            mode='max'
        ),
  	# Write TensorBoard logs to `./logs` directory
  	tf.keras.callbacks.TensorBoard(log_dir=run_dir+'/logs', profile_batch=0) 
    ]
    file_writer = tf.summary.create_file_writer(run_dir+'/logs/metrics')
    file_writer.set_as_default()

    if args.checkpoint:
        print("Loading model from: ", args.checkpoint)
        #model.load_weights(args.checkpoint)
        model.load_weights(args.checkpoint)
        #model = tf.keras.models.load_model(args.checkpoint)
    else:
        print("Model training from scratch -")

    if mode=="train":
        model.fit(dataset, epochs=args.epochs, callbacks=callbacks, validation_data=val_dataset)
        #model.save_weights(args.save_path +'/final_weights/Conv3D_model')
    else: 
        model.evaluate(dataset)
    

def main():
    # Settings
    args = parse_args()

    #use_gpu = torch.cuda.is_available()
    use_gpu = False
    run(args,use_gpu)

if __name__ == '__main__':
    main()
