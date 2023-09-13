# load waymo dataset
import torch
import tensorflow as tf
import numpy as np
import os

num_map_samples = 30000


def features_description():
    num_map_samples = 30000

    # Example field definition
    roadgraph_features = {
        'roadgraph_samples/dir': tf.io.FixedLenFeature(
            [num_map_samples, 3], tf.float32, default_value=None
        ),
        'roadgraph_samples/id': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None
        ),
        'roadgraph_samples/type': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None
        ),
        'roadgraph_samples/valid': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None
        ),
        'roadgraph_samples/xyz': tf.io.FixedLenFeature(
            [num_map_samples, 3], tf.float32, default_value=None
        ),
    }
    # Features of other agents.
    state_features = {
        'state/id':
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/type':
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/is_sdc':
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        'state/tracks_to_predict':
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        'state/current/bbox_yaw':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/height':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/length':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/timestamp_micros':
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        'state/current/valid':
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        'state/current/vel_yaw':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/velocity_x':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/velocity_y':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/width':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/x':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/y':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/z':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/future/bbox_yaw':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/height':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/length':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/timestamp_micros':
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        'state/future/valid':
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        'state/future/vel_yaw':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/velocity_x':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/velocity_y':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/width':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/x':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/y':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/z':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/past/bbox_yaw':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/height':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/length':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/timestamp_micros':
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        'state/past/valid':
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        'state/past/vel_yaw':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/velocity_x':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/velocity_y':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/width':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/x':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/y':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/z':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    }

    traffic_light_features = {
        'traffic_light_state/current/state':
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        'traffic_light_state/current/valid':
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        'traffic_light_state/current/x':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/current/y':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/current/z':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/past/state':
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        'traffic_light_state/past/valid':
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        'traffic_light_state/past/x':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        'traffic_light_state/past/y':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        'traffic_light_state/past/z':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    }

    features_description = {}
    features_description.update(roadgraph_features)
    features_description.update(state_features)
    features_description.update(traffic_light_features)
    return features_description


def parse(value, features_description):
    info = {}
    decoded_example = tf.io.parse_single_example(value, features_description)
    info['decoded_example'] = decoded_example
    info['num_time_steps'] = decoded_example['state/past/x'].shape[1] + \
        decoded_example['state/future/x'].shape[1] + \
        decoded_example['state/current/x'].shape[1]
    info['num_of_objects'] = decoded_example['state/past/x'].shape[0]
    x = tf.concat([
        decoded_example['state/past/x'],
        decoded_example['state/current/x'],
        decoded_example['state/future/x']
    ], axis=1)
    x = tf.split(x, info['num_time_steps'], axis=1)
    # shape of x is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    y = tf.concat([
        decoded_example['state/past/y'],
        decoded_example['state/current/y'],
        decoded_example['state/future/y']
    ], axis=1)
    y = tf.split(y, info['num_time_steps'], axis=1)
    # shape of y is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    z = tf.concat([
        decoded_example['state/past/z'],
        decoded_example['state/current/z'],
        decoded_example['state/future/z']
    ], axis=1)
    z = tf.split(z, info['num_time_steps'], axis=1)
    # shape of z is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    heading_angle = tf.concat([
        decoded_example['state/past/bbox_yaw'],
        decoded_example['state/current/bbox_yaw'],
        decoded_example['state/future/bbox_yaw']
    ], axis=1)
    heading_angle = tf.split(heading_angle, info['num_time_steps'], axis=1)
    # shape of heading_angle is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    velocity_x = tf.concat([
        decoded_example['state/past/velocity_x'],
        decoded_example['state/current/velocity_x'],
        decoded_example['state/future/velocity_x']
    ], axis=1)
    velocity_x = tf.split(velocity_x, info['num_time_steps'], axis=1)
    # shape of velocity_x is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    velocity_y = tf.concat([
        decoded_example['state/past/velocity_y'],
        decoded_example['state/current/velocity_y'],
        decoded_example['state/future/velocity_y']
    ], axis=1)
    velocity_y = tf.split(velocity_y, info['num_time_steps'], axis=1)
    # shape of velocity_y is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    vel_yaw = tf.concat([
        decoded_example['state/past/vel_yaw'],
        decoded_example['state/current/vel_yaw'],
        decoded_example['state/future/vel_yaw']
    ], axis=1)
    vel_yaw = tf.split(vel_yaw, info['num_time_steps'], axis=1)
    # shape of vel_yaw is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    valid = tf.concat([
        decoded_example['state/past/valid'],
        decoded_example['state/current/valid'],
        decoded_example['state/future/valid']
    ], axis=1)
    valid = tf.split(valid, info['num_time_steps'], axis=1)
    # shape of valid is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    length = tf.concat([
        decoded_example['state/past/length'],
        decoded_example['state/current/length'],
        decoded_example['state/future/length']
    ], axis=1)
    length = tf.split(length, info['num_time_steps'], axis=1)
    # shape of length is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    width = tf.concat([
        decoded_example['state/past/width'],
        decoded_example['state/current/width'],
        decoded_example['state/future/width']
    ], axis=1)
    width = tf.split(width, info['num_time_steps'], axis=1)
    # shape of width is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    height = tf.concat([
        decoded_example['state/past/height'],
        decoded_example['state/current/height'],
        decoded_example['state/future/height']
    ], axis=1)
    height = tf.split(height, info['num_time_steps'], axis=1)
    # shape of height is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    timestamp_micros = tf.concat([
        decoded_example['state/past/timestamp_micros'],
        decoded_example['state/current/timestamp_micros'],
        decoded_example['state/future/timestamp_micros']
    ], axis=1)
    timestamp_micros = tf.split(
        timestamp_micros, info['num_time_steps'], axis=1)
    # shape of timestamp_micros is a list of tensors of shape (num_of_objects, 1), the length of the list is num_time_steps
    object_type = decoded_example['state/type']
    # shape of object_type is [num_of_objects]
    is_self_driving = decoded_example['state/is_sdc']
    info['x'] = x
    info['y'] = y
    info['z'] = z
    info['heading_angle'] = heading_angle
    info['velocity_x'] = velocity_x
    info['velocity_y'] = velocity_y
    info['vel_yaw'] = vel_yaw
    info['valid'] = valid
    info['length'] = length
    info['width'] = width
    info['height'] = height
    info['timestamp_micros'] = timestamp_micros
    info['object_type'] = object_type
    info['is_self_driving'] = is_self_driving
    return info
# need to know how to map to the whole dataset
