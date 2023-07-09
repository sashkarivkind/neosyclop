"""dataset.py

This module implements functions for reading ImageNet (ILSVRC2012)
dataset in TFRecords format.
"""


import os
from functools import partial

import tensorflow as tf

from utils.config import config
from utils.image_processing import preprocess_image, resize_and_rescale_image,preprocess_image_drc


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    # with tf.name_scope(values=[image_buffer], name=scope,
    #                    default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height
    # and width that is set dynamically by decode_jpeg. In other
    # words, the height and width of image is unknown at compile-i
    # time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).
    # The various adjust_* ops all require this range for dtype
    # float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def _parse_fn(example_serialized, is_training,image_h=224,image_w=224,**kwargs):
    """Helper function for parse_fn_train() and parse_fn_valid()

    Each Example proto (TFRecord) contains the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

    Args:
        example_serialized: scalar Tensor tf.string containing a
                            serialized Example protocol buffer.
        is_training: training (True) or validation (False).

    Returns:
        image_buffer: Tensor tf.string containing the contents of
        a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
    """
    feature_map = {
        'file_name': tf.io.FixedLenFeature([], dtype=tf.string,
                                           default_value=''),
        'image': tf.io.FixedLenFeature([], dtype=tf.string,
                                       default_value=''),
        'label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                       default_value=-1),
    }
    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    image = decode_jpeg(parsed['image'])
    if config.DATA_AUGMENTATION:
        image = preprocess_image(image, image_h, image_w, is_training=is_training,**kwargs)
    else:
        raise NotImplementedError
        #image = resize_and_rescale_image(image, 224, 224,**kwargs)
    # The labedl in the tfrecords is 1~1000 (0 not used).
    # So I think the minus 1 (of class label) is needed below.
    label = tf.one_hot(parsed['label'] , 1000, dtype=tf.float32) ##todo!!!!!!
    return (image, label)

def get_dataset(tfrecords_dir, subset, batch_size, deterministic=False, NUM_DATA_WORKERS=None,  **kwargs):
    """Read TFRecords files and turn them into a TFRecordDataset."""
    if NUM_DATA_WORKERS is not None:
        config.NUM_DATA_WORKERS = NUM_DATA_WORKERS
    files = tf.io.matching_files(os.path.join(tfrecords_dir, '*-%s*' % subset))
    shards = tf.data.Dataset.from_tensor_slices(files)

    if not deterministic:
        shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)

    if not deterministic:
        dataset = dataset.shuffle(buffer_size=8192)

    if subset == 'train':
        parser = partial(_parse_fn, is_training=True ,**kwargs)
    elif subset == 'validation':
        parser = partial(_parse_fn, is_training=False,**kwargs)
    else:
        raise NotImplementedError

    # dataset = dataset.apply(
    #     tf.data.experimental.map_and_batch(
    #         map_func=parser,
    #         batch_size=batch_size,
    #         num_parallel_calls=config.NUM_DATA_WORKERS))


    if deterministic:
        dataset = dataset.map(parser, num_parallel_calls=config.NUM_DATA_WORKERS)
        dataset = dataset.batch(batch_size)
    else:
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                map_func=parser,
                batch_size=batch_size,
                num_parallel_calls=config.NUM_DATA_WORKERS
            )
        )

    dataset = dataset.prefetch(batch_size)

    return dataset