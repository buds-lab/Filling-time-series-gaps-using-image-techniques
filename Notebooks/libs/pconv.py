import tensorflow as tf

class pconv2d(tf.keras.layers.Layer):
  def __init__(self, output_dim, kernel_size=[3,3], strides = 1, padding = 'SAME', use_bias = True):
    super(pconv2d, self).__init__()
    self.output_dim = output_dim
    self.kernel_size = kernel_size
    self.window_area = kernel_size[0]*kernel_size[1]
    self.strides = strides
    self.padding = padding
    self.use_bias = use_bias

  def build(self, input_shape):
    #input_shape = [feature_shape, mask_shape]
    feature_shape = input_shape[0] #[None, h, w, d] a batched 2d feature.
    mask_shape = input_shape[1] #[None, h, w, 1] a batched 2d mask.

    self.feature_kernel = self.add_weight("feature_kernel",
                                          initializer= 'glorot_uniform',
                                          trainable= True,
                                          shape= self.kernel_size+ [feature_shape[-1], self.output_dim])
    
    self.mask_kernel = self.add_weight("mask_kernel",
                                       initializer= 'ones',
                                       trainable= False,
                                       shape= self.kernel_size+ [1, 1])
    if self.use_bias:
      self.bias = self.add_weight("bias",
                                  initializer= 'zeros',
                                  trainable= True,
                                  shape= [1, 1, 1, self.output_dim])

  def call(self, inputs):
    #inputs = [feature, mask]
    feature = inputs[0]
    mask = inputs[1]
    
    #doing masked convolusion
    next_feature = tf.nn.conv2d(
        input= tf.math.multiply(feature, mask),
        filters= self.feature_kernel,
        strides= self.strides,
        padding= self.padding
    )

    next_mask = tf.nn.conv2d(
        input= mask,
        filters= self.mask_kernel,
        strides= self.strides,
        padding= self.padding
    )
    #each pixel ranges from 0 to window_area-1 after convolution.
    #clip the results to 0, 1

    mask_ratio = self.window_area/ (next_mask+ 1e-6)
    next_mask = tf.clip_by_value(next_mask, 0., 1.)

    next_feature = tf.math.multiply(next_feature, tf.math.multiply(mask_ratio, next_mask) )

    if self.use_bias:
      next_feature = next_feature+ self.bias
  
    return [next_feature, next_mask]
