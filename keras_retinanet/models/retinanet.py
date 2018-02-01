"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from .. import initializers
from .. import layers
from .. import losses

import numpy as np

custom_objects = {
    'UpsampleLike'          : layers.UpsampleLike,
    'PriorProbability'      : initializers.PriorProbability,
    'RegressBoxes'          : layers.RegressBoxes,
    'NonMaximumSuppression' : layers.NonMaximumSuppression,
    'Anchors'               : layers.Anchors,
    '_smooth_l1'            : losses.smooth_l1(),
    '_focal'                : losses.focal(),
}


def default_pyramid_cnn(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # 23, 9, (?, ?, ?, 23 * 9)
    print("Num classes: {}; Num anchors: {}; Outputs shape at default_pyramid_cnn is {}"
            .format(num_classes, num_anchors, outputs.shape))

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def default_featurevector_model(
    num_classes,
    num_anchors,
    pyramid_cnn_output,
    pyramid_feature_size=256,
    name='featurevector_submodel'
):
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs = keras.layers.Input(shape=(None, None, num_anchors * num_classes))
    outputs = inputs

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_cnn_output,
    pyramid_feature_size=256,
    name='classification'
):

    inputs = keras.layers.Input(shape=(None, num_classes))
    outputs = inputs
    # reshape(flatten) output and apply sigmoid
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(
    num_anchors,
    pyramid_feature_size=256,
    regression_feature_size=256,
    name='regression_submodel'
):
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return P3, P4, P5, P6, P7


class AnchorParameters:
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def default_submodels(num_classes, anchor_parameters):
    submodels = [
        ('regression', default_regression_model(anchor_parameters.num_anchors())),
        ('pyramid_cnn', default_pyramid_cnn(num_classes, anchor_parameters.num_anchors()))
    ]
    return submodels

def attach_features_and_classification(num_classes, anchor_parameters, submodels):
    extra_submodels = [
        ('featurevector', default_featurevector_model(num_classes, anchor_parameters.num_anchors(), submodels[-1][1].outputs)),
        ('classification', default_classification_model(num_classes, anchor_parameters.num_anchors(), submodels[-1][1].outputs))
    ]

    return extra_submodels


def __build_model_pyramid(name, model, features):

    pyramid_outputs = []

    for i, f in enumerate(features):
        pyramid_outputs.append(model(f))

    return pyramid_outputs


def __build_pyramid(models, features):
    # return [__build_model_pyramid(n, m, features) for n, m in models]

    pyramid = []
    for n, m in models:
        pyramid_level_built = __build_model_pyramid(n, m, features)
        pyramid.append(pyramid_level_built)

    return pyramid

def __build_anchors(anchor_parameters, features):
    anchors = []
    print("Building anchors....")
    for i, f in enumerate(features):
        anchors.append(layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f))
    return keras.layers.Concatenate(axis=1)(anchors)


def retinanet(
    inputs,
    backbone,
    num_classes,
    anchor_parameters       = AnchorParameters.default,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    name                    = 'retinanet'
):
    if submodels is None:
        submodels = default_submodels(num_classes, anchor_parameters)

    _, C3, C4, C5 = backbone.outputs  # we ignore C2

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    pyramid = __build_pyramid(submodels, features)

    # pyramid[0] = regression output, list of unconcatenated outputs
    # pyramid[1] = pyramid_cnn_output, list of unconcatenated outputs

    extra_submodels = attach_features_and_classification(num_classes, anchor_parameters, submodels)

    class_output = [extra_submodels[1][1](keras.layers.Reshape((-1, num_classes))(class_pyramid_item)) for class_pyramid_item in pyramid[1]]

    classification_submodel_outputs = keras.layers.Concatenate(axis=1, name='classification_final')(class_output)
    regression_submodel_outputs = keras.layers.Concatenate(axis=1, name='regression')(pyramid[0])

    # feature vector output at each pyramid level
    featurevector_output = [extra_submodels[0][1](pyramid_level_output) for pyramid_level_output in pyramid[1]]

    pyramid_outputs = [regression_submodel_outputs, classification_submodel_outputs]

    anchors = __build_anchors(anchor_parameters, features)

    return keras.models.Model(inputs=inputs, outputs=[anchors] + pyramid_outputs + featurevector_output, name=name)


def retinanet_bbox(inputs, num_classes, nms=True, name='retinanet-bbox', *args, **kwargs):
    model = retinanet(inputs=inputs, num_classes=num_classes, *args, **kwargs)

    # we expect the anchors, regression and classification values as first output
    anchors        = model.outputs[0]
    regression     = model.outputs[1]
    classification = model.outputs[2]
    featurevector  = model.outputs[3:]

    # features_concat = [keras.layers.Reshape((-1, num_classes))(pyramid_feature) for pyramid_feature in featurevector]
    # features = keras.layers.Concatenate(axis=1)(features_concat)

    # apply predicted regression to anchors
    boxes      = layers.RegressBoxes(name='boxes')([anchors, regression])
    detections = keras.layers.Concatenate(axis=2)([boxes, classification])

    # additionally apply non maximum suppression
    if nms:
        detections = layers.NonMaximumSuppression(name='nms')([boxes, classification, detections, features])
        # split up NMS outputs
        detections = detections[0]
        features = detections[1]

    # construct the model
    return keras.models.Model(inputs=inputs, outputs=model.outputs[1:3] + [detections] + featurevector, name=name)
