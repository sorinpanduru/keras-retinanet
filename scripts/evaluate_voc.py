#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
import keras.preprocessing.image
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.voc_eval import VOCEvaluator
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.keras_version import check_keras_version

import tensorflow as tf

import argparse
import os

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for COCO object detection.')
    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument('pascal_path', help='Path to Pascal directory (ie. /tmp/VOCDevKit/VOC2007).')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--set', help='Name of the set file to evaluate (defaults to test).', default='test')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold', help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-det', help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-images', help='Save images with annotation ground truth and predictions.', action='store_true')
    parser.add_argument('--save-path', help='Save path for images (defaults to ./images_voc).', default='images_voc', type=str)

    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the model
    print('Loading model, this may take a second...')
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    # create image data generator object
    test_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    # create a generator for testing data
    test_generator = PascalVocGenerator(
        args.pascal_path,
        args.set,
        test_image_data_generator
    )

    voc_evaluator = VOCEvaluator(
        generator=test_generator,
        model=model,
        threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_det,
        save=args.save_images,
        save_path=args.save_path
    )
    voc_evaluator.evaluate()