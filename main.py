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

#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#from torchvision import datasets
#from torchsummary import summary

#from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
#from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy

from tqdm import tqdm

from utils import *
from model import *
# from dataset import *
# from lr_scheduler import *
from cvtransforms import *

from PIL import Image

print("Process Number: ",os.getpid())

SEED = 1
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)

tf.random.set_seed(SEED)
np.random.seed(SEED)

use_gpu = False

#if torch.cuda.is_available():
#    print("Available")
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')

#torch.cuda.set_device(1)
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

#def data_loader(args):
#    dsets = {x: LRW(x, args.dataset) for x in ['train', 'val', 'test']}
#    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,\
#                       shuffle=True, num_workers=args.workers, pin_memory=use_gpu) \
#                       for x in ['train', 'val', 'test']}
#    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
#    print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
#    # dset_loaders['train'] , dset_loaders['val']
#    return dset_loaders, dset_sizes

#def reload_model(model, path=""):
#    if not bool(path):
#        print('train from scratch')
#        return model
#    else:
#        model_dict = model.state_dict()
#        pretrained_dict = torch.load(path)
#        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#        model_dict.update(pretrained_dict)
#        model.load_state_dict(model_dict)
#        print('*** model has been successfully loaded! ***')
#        return model


#def showLR(optimizer):
#    lr = []
#    for param_group in optimizer.param_groups:
#        lr += [param_group['lr']]
#    return lr

def lr_scheduler(epoch, lr):
    sleep_epochs = 5
    half = sleep_epochs 
    if epoch < sleep_epochs:
        return lr
    else:
        return lr * tf.math.pow(0.5, (epoch-sleep_epochs+1)/float(half))
#def load_file(filename):
#    cap = np.load(filename)
#    cap = tf.io.read_file(filename)
#    # arrays = np.stack([cv2.cvtColor(cap[_], cv2.COLOR_RGB2GRAY) for _ in range(29)], axis=0)
#    arrays = np.stack(cap, axis=0)
#    arrays = arrays / 255.
#    return arrays

def check_dataset(dataset, dir_name, img_type):
    for i,example in enumerate(dataset.take(1)):
        #print(example)
        image, label = example
        print("img shape:", image.numpy().shape)
        #print("img: ", image.numpy())
        print("img label: ", label)
        image_shape = image.numpy().shape
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
        image_buffer = tf.reshape(features[path], shape=[])
        image = tf.io.decode_raw(image_buffer, tf.uint8)
        image = tf.reshape(image, tf.stack([height, width, num_depth]))
        image = tf.reshape(image, [1, height, width, num_depth])
#        image = image[:,:,:,0]# / tf.constant(255, shape=(1, height, width), dtype=tf.uint8) 
        image_seq.append(image)
#    image_seq = tf.reshape(image_seq, [1, n_frames, height, width])
    image_seq = tf.concat(image_seq, 0)
    label = features['label']
    print("image: ", image_seq)
    print("label: ", label)
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
    y = tf.keras.backend.one_hot(label, n)
    return image, y

def run(args, use_gpu=True):
    
    # saving
    save_path = os.path.join(os.getcwd(),'models')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    model = lipnext(inputDim=256, hiddenDim=512, nClasses=args.nClasses, frameLen=29, alpha=args.alpha)
    #model = reload_model(model, args.path) #.to(device)
    #model = tf.keras.Sequential([
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(1)
    #])

    if args.train==True:
        mode = "train"
    else:
        mode = "test"
    
#    train_list = glob.glob("./test_tfrecord_ACTUALLY_color/*.tfrecords") 
#    val_list = glob.glob("./test_tfrecord_ACTUALLY_color/*.tfrecords") 
#    test_list = glob.glob("./test_tfrecord_ACTUALLY_color/*.tfrecords") 
    train_list = glob.glob("/mnt/disks/data/dataset/lipread_tfrecords/*/train/*.tfrecords")
    val_list = glob.glob("/mnt/disks/data/dataset/lipread_tfrecords/*/val/*.tfrecords")
    test_list = glob.glob("/mnt/disks/data/dataset/lipread_tfrecords/*/test/*.tfrecords")
    
    if mode=="train":
        dataset = tf.data.TFRecordDataset(train_list)
        val_dataset = tf.data.TFRecordDataset(val_list)
    else:
        dataset = tf.data.TFRecordDataset(test_list)
    print("raw_dataset: ", dataset)

    if mode=="train":
        dataset = dataset.map(_parse_function)
        val_dataset = val_dataset.map(_parse_function)
        #check_dataset(dataset, "train_test_images", 'RGB')
        
        dataset = dataset.map(_train_preprocess_function)
        val_dataset = val_dataset.map(_test_preprocess_function)
        #check_dataset(dataset, "train_test_images_processed", 'L')
        
        dataset = dataset.map(_normalize_function)
        val_dataset = val_dataset.map(_normalize_function)
    
        dataset = dataset.shuffle(500).batch(16)
        val_dataset = val_dataset.batch(16)
    
    else:
        dataset = dataset.map(_parse_function)
        #check_dataset(dataset, "test_test_images", 'RGB')

        dataset = dataset.map(_test_preprocess_function)
        
        dataset = dataset.map(_normalize_function)
        
        dataset = dataset.batch(16)
 
    #check_dataset(dataset, "test_test_images_processed", 'L')
    model.compile(optimizer=Adam(learning_rate = args.lr), 
           loss=CategoricalCrossentropy(from_logits=True),
           metrics=['accuracy', TopKCategoricalAccuracy(3) ]) 

    run_dir = args.save_path + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
  	tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        # Learning rate scheduler
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
        # Save checkpoints
        tf.keras.callbacks.ModelCheckpoint(
            filepath=run_dir+'/checkpoints/Conv3D_model_{epoch}', 
            #save_weights_only=True,
            save_best_only=True,
            monitor='val_loss'
        ),
  	# Write TensorBoard logs to `./logs` directory
  	tf.keras.callbacks.TensorBoard(log_dir=run_dir+'/logs') 
    ]

    if args.checkpoint:
        print("Loading model from: ", args.checkpoint)
        #model.load_weights(args.checkpoint)
        model = tf.keras.models.load_model(args.checkpoint)
    else:
        print("Model training from scratch")

    if mode=="train":
        model.fit(dataset, epochs=args.epochs, callbacks=callbacks, validation_data=val_dataset)
        #model.save_weights(args.save_path +'/final_weights/Conv3D_model')
    else: 
        model.evaluate(dataset)
    
#    dset_loaders, dset_sizes = data_loader(args)
#    
#    train_loader = dset_loaders['train']
#    val_loader = dset_loaders['test']
#
#    train_size = dset_sizes['train']
#    val_size = dset_sizes['val']
    
    # print(model.parameters()) - add to optimizer TO DO 
    # optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr,
    #                                     decay=0.)
#    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
    # scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=5, half=5, verbose=1)
    # TQDM
#    desc = "ITERATION - loss: {:.2f}"
    # pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0))

 # Ignite trainer
#    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, \
#                                        device=device, prepare_batch=prepare_train_batch)
#    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 
#                                                            'cross_entropy': Loss(F.cross_entropy),
#                                                            'top-3': TopKCategoricalAccuracy(3)
#                                                            }, device=device,\
#                                                            prepare_batch=prepare_val_batch)

#    # call backs
#    @evaluator.on(Events.EPOCH_STARTED)
#    def start_val(engine):
#        tqdm.write(
#            "Evaluation in progress"
#        )
#
#    @trainer.on(Events.ITERATION_COMPLETED)
#    def log_training_loss(engine):
#        iter = (engine.state.iteration - 1) % len(train_loader) + 1
#
#        if iter % args.interval == 0:
#            pbar.desc = desc.format(engine.state.output)
#            pbar.update(args.interval)
#
#
#    @trainer.on(Events.EPOCH_COMPLETED)
#    def log_training_results(engine):
#        pbar.refresh()
#        evaluator.run(train_loader)
#        metrics = evaluator.state.metrics
#        avg_accuracy = metrics['accuracy']
#        avg_loss = metrics['cross_entropy']
#        top_acc = metrics['top-3']
#        tqdm.write(
#            "Training Results - Epoch: {}  Avg accuracy: {:.2f}, Top3: {:.2f} Avg loss: {:.2f}"
#            .format(engine.state.epoch, avg_accuracy, top_acc, avg_loss)
#        )
#
#    @trainer.on(Events.EPOCH_COMPLETED)
#    def log_validation_results(engine):
#
#        # large dataset so saving often
#        tqdm.write("saving model ..")
#        torch.save(model.state_dict(), os.path.join(save_path,'epoch'+str(engine.state.epoch+1)+'.pt'))
#        # saving to ONNX format
#        dummy_input = torch.randn(args.batch_size, 1, 29, 88, 88)
#        torch.onnx.export(model, dummy_input, "lipnext.onnx")
#
#        evaluator.run(val_loader)
#        metrics = evaluator.state.metrics
#        avg_accuracy = metrics['accuracy']
#        top_acc = metrics['top-3']
#        avg_loss = metrics['cross_entropy']
#        tqdm.write(
#            "Validation Results - Epoch: {}  Avg accuracy: {:.2f}, Top3: {:.2f} Avg loss: {:.2f} "
#            .format(engine.state.epoch, avg_accuracy, top_acc, avg_loss)
#        )
#        
#
#        pbar.n = pbar.last_print_n = 0
#
#    @trainer.on(Events.EPOCH_COMPLETED)
#    def update_lr(engine):
#        scheduler.step(engine.state.epoch)
#
#    trainer.run(train_loader, max_epochs=args.epochs)
#    pbar.close()

def main():
    # Settings
    args = parse_args()

    #use_gpu = torch.cuda.is_available()
    use_gpu = False
    run(args,use_gpu)

if __name__ == '__main__':
    main()
