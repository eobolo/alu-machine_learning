#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer
"""


import numpy as np
import tensorflow as tf


class NST:
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        # Validate style_image
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Validate content_image
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Validate alpha
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        
        # Validate beta
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        
        # Set eager execution
        tf.executing_eagerly()
        
        # Preprocess and set instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        # Validate input image
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Convert to tensor
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Get dimensions
        h, w = image.shape[0], image.shape[1]
        max_dim = 512
        scale = max_dim / max(h, w)
        
        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize using bicubic interpolation
        image = tf.image.resize(image, [new_h, new_w], method='bicubic')
        
        # Rescale pixel values to [0, 1]
        image = image / 255.0
        
        # Add batch dimension
        image = tf.expand_dims(image, axis=0)
        
        return image