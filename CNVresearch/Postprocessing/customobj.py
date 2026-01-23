import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import numpy as np

@tf.keras.utils.register_keras_serializable()
class RepeatChannels(Layer):
    def __init__(self, **kwargs):
        super(RepeatChannels, self).__init__(**kwargs)

    def call(self, inputs):
        # Assuming the repeat operation repeats across channels
        return K.repeat_elements(inputs, rep=3, axis=-1)

    def get_config(self):
        config = super(RepeatChannels, self).get_config()
        return config

@tf.keras.utils.register_keras_serializable()
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice coefficient, a measure of overlap between two samples.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)

    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return dice

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    # Binary Cross-Entropy
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Dice Loss
    intersection = tf.reduce_sum(y_true * y_pred)
    dice_loss = 1 - (2. * intersection + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)
    # Combine both losses
    return bce_loss + dice_loss

# Load models with custom objects
from tensorflow.keras.models import load_model

custom_objects = {
    'RepeatChannels': RepeatChannels,
    'iou': iou,
    'dice_coef': dice_coef,
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss
}