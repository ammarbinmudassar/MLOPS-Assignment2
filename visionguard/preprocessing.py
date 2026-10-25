"""The original Task 2 pixel transformation, shared with local inference."""

import tensorflow as tf

PREPROCESSING_ID = "rgb_letterbox_bilinear_antialias_uint8_v1"


def decode_resize_bytes(contents, image_size):
    image = tf.io.decode_image(contents, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize_with_pad(image, image_size, image_size, antialias=True)
    # Training cached uint8 before casting to float32; keep that quantization.
    return tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
