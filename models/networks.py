import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow_addons.layers import GroupNormalization

"""
There is no custom layer here. 
All built models can be saved by model.save('model.h5') and loaded by load_model('model.h5').

build_*   - a ready-to-use built model.
*_module  - a class-level module. The usage is like layers.Add()([a, b])
*_block   - a functional-level block. The usage is like layers.add([a, b])
"""

# For building unet and retinaunet if not giving strides_list
def get_strides_list(layer_number, im_size, mode='symmetric'):
    # mode: symmetric. e.g. [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    # mode: last. e.g. [[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
    s_list = np.zeros([layer_number, len(im_size)], dtype=np.uint8)
    for i in range(layer_number):
        # stop pooling when size is odd && size <= 4
        to_pool = (np.array(im_size) % 2 ** (i + 1) == 0).astype(int)
        to_pool *= ((np.array(im_size) // 2 ** (i + 1)) > 4).astype(int)
        s_list[i] = 1 + to_pool
    if mode == 'symmetric':
        strides_list = np.concatenate([s_list[::-2], s_list[layer_number % 2::2]])
    else:
        strides_list = np.array(s_list)
    return strides_list

@tf.function
def l1_loss(gt, pred, mask=None, paired=None):
    if mask is None:
        loss = tf.reduce_mean(tf.abs(gt - pred))
    else:  # paired target loss
        if paired is None:
            paired = 1.
        else:  # partially paired
            paired = tf.expand_dims(tf.expand_dims(tf.expand_dims(paired, -1), -1), -1)
            paired = tf.tile(paired, [1, tf.shape(gt)[1], tf.shape(gt)[2], tf.shape(gt)[3]])
        gt = tf.multiply(gt, paired)
        pred = tf.multiply(pred, paired)
        loss = tf.reduce_sum(tf.multiply(tf.abs(gt - pred), mask)) / tf.reduce_sum(mask)
    return loss

@tf.function
def l2_loss(gt, pred, mask=None, paired=None):
    if mask is None:
        loss = tf.reduce_mean(tf.math.square(gt - pred))
    else:  # paired target loss
        if paired is None:
            paired = 1.
        else:  # partially paired
            paired = tf.expand_dims(tf.expand_dims(tf.expand_dims(paired, -1), -1), -1)
            paired = tf.tile(paired, [1, tf.shape(gt)[1], tf.shape(gt)[2], tf.shape(gt)[3]])
        gt = tf.multiply(gt, paired)
        pred = tf.multiply(pred, paired)
        loss = tf.reduce_sum(tf.multiply(tf.math.square(gt - pred), mask)) / tf.reduce_sum(mask)
    return loss

@tf.function
def segmentation_loss(labels, logits, loss_weights=None, nclass=2, class_weights=None,
                      loss_type='both', debug_roi_dice=False, debug_loss=False, policy='float32'):
    if loss_weights is None:
        loss_weights = tf.cast(tf.ones_like(logits), tf.float32)
    
    if policy != 'float32':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        loss_weights = tf.cast(loss_weights, tf.float32)

    flat_logits = tf.reshape(logits, [-1, nclass])
    flat_labels = tf.reshape(labels, [-1, nclass])
    flat_weights = tf.reshape(loss_weights, [-1, nclass])

    weights_logits = tf.multiply(flat_logits, flat_weights)
    weights_labels = tf.multiply(flat_labels, flat_weights)

    # Cross-entrpy loss
    if class_weights is not None:
        class_weights = tf.constant(np.asarray(class_weights, dtype=np.float32))
        weight_map = tf.reduce_max(tf.multiply(flat_labels, class_weights), axis=1)
        loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=weights_logits, labels=weights_labels)

        # First multiply by class weights, then by loss weights due to missing of rois
        weighted_loss = tf.multiply(loss_map, weight_map)
    else:
        loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=weights_logits, labels=weights_labels)
        weighted_loss = loss_map
    cross_entropy_loss = tf.reduce_mean(weighted_loss)

    if 'focal' in loss_type:
        # Focal Loss (categorical, multi-class)
        beta = 1.0 * nclass  # scaling factor for categorical focal loss
        gamma = 2.0  # focusing factor

        ce_loss = weighted_loss  # use class-weighted ce loss calculated as above (replace alpha)
        pt = tf.nn.softmax(weights_logits)
        focal_loss_map = beta * tf.pow(1.0 - pt, gamma) * (weights_labels * ce_loss[:, None])
        focal_loss = tf.reduce_mean(focal_loss_map)

    # Dice loss
    probs = tf.nn.softmax(logits)
    predictions = tf.argmax(probs, 3)
    eps = 1.0
    if debug_roi_dice:
        dice_value = []
    else:
        dice_value = 0.
    dice_loss = 0.
    n_rois = 0
    weighted_n_rois = 0

    for i in range(1, nclass):
        if class_weights is not None:
            weights = class_weights[i]
        else:
            weights = 1.0

        slice_weights = loss_weights[..., i]
        slice_prob = probs[..., i]
        slice_pred = tf.cast(tf.equal(predictions, i), tf.float32)
        slice_label = labels[..., i]
        intersection_prob = eps + tf.reduce_sum(tf.multiply(slice_prob, slice_label), axis=[1, 2])
        intersection_pred = eps + tf.reduce_sum(tf.multiply(slice_pred, slice_label), axis=[1, 2])
        union_prob = 2.0 * eps + tf.reduce_sum(slice_prob, axis=[1, 2]) + tf.reduce_sum(slice_label, axis=[1, 2])
        union_pred = 2.0 * eps + tf.reduce_sum(slice_pred, axis=[1, 2]) + tf.reduce_sum(slice_label, axis=[1, 2])

        # Multiply by loss weights
        roi_exist = tf.reduce_mean(slice_weights, axis=[1, 2]) # Either 1 or 0
        if debug_roi_dice:
            dice_value.append(tf.reduce_mean(tf.multiply(tf.truediv(intersection_pred, union_pred), roi_exist)))
        else:
            dice_value += tf.reduce_mean(tf.multiply(tf.truediv(intersection_pred, union_pred), roi_exist))

        n_rois += tf.reduce_mean(roi_exist)
        weighted_n_rois += tf.reduce_mean(roi_exist) * weights
        dice_loss += tf.reduce_mean(tf.multiply(tf.truediv(intersection_prob, union_prob), roi_exist)) * weights

    dice_loss = 1.0 - dice_loss * 2.0 / weighted_n_rois

    if debug_roi_dice:
        roi_dices = tf.multiply(dice_value, 2.0)
        dice = tf.reduce_sum(roi_dices) / n_rois
    else:
        roi_dices = None
        dice = dice_value * 2.0 / n_rois

    if loss_type == 'cross_entropy':
        loss = cross_entropy_loss
    elif loss_type == 'dice':
        loss = dice_loss
    elif loss_type == 'both':
        loss = cross_entropy_loss + dice_loss
    elif loss_type == 'both_focal':
        loss = focal_loss + dice_loss
    else:
        raise ValueError("Unknown cost function: " + loss_type)

    if debug_loss:
        return loss, roi_dices, dice, cross_entropy_loss, dice_loss
    else:
        return loss, roi_dices, dice

@tf.function
def generator_loss(fake_logits):
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
    return fake_loss

@tf.function
def discriminator_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))

    total_loss = real_loss + fake_loss
    return total_loss * 0.5

def soft_erode(img):
    p1 = -tf.nn.max_pool3d(-img, ksize=(1, 3, 3, 1, 1), strides=(1, 1, 1, 1, 1), padding='SAME')
    p2 = -tf.nn.max_pool3d(-img, ksize=(1, 3, 1, 3, 1), strides=(1, 1, 1, 1, 1), padding='SAME')
    p3 = -tf.nn.max_pool3d(-img, ksize=(1, 1, 3, 3, 1), strides=(1, 1, 1, 1, 1), padding='SAME')
    return tf.math.minimum(tf.math.minimum(p1, p2), p3)

def soft_dilate(img):
    return tf.nn.max_pool3d(img, ksize=(1, 3, 3, 3, 1), strides=(1, 1, 1, 1, 1), padding='SAME')

def soft_open(img):
    img = soft_erode(img)
    img = soft_dilate(img)
    return img

def soft_skel(img, iters):
    img1 = soft_open(img)
    skel = tf.nn.relu(img - img1)

    for j in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = tf.nn.relu(img - img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta - intersect)
    return skel

def modify_input_channels(model, in_channels):
    config = model.get_config()
    input_shape = config['layers'][0]['config']['batch_input_shape']
    input_shape = list(input_shape)
    input_shape[-1] = in_channels
    input_shape = tuple(input_shape)
    config['layers'][0]['config']['batch_input_shape'] = input_shape
    
    new_model = tf.keras.models.Model.from_config(config)
    for layer in new_model.layers[1:]:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print('Weights not transfered:', layer.name)
            continue

    return new_model

def build_auxiliary_ds_model(im_size, nclass, layer_number, deep_supervision_scales):
    """
    For pyramid deep supervision loss.
    
    e.g. layer_number=5
    deep_supervision_scales: [(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
    ds_loss_weights: [0.0, 0.06666667, 0.13333333, 0.26666667, 0.53333333]
    """
    
    input_shape = list(im_size) + [nclass,]
    dim = len(im_size)
    if dim == 2:
        myavgpool = layers.AveragePooling2D
    elif dim == 3:
        myavgpool = layers.AveragePooling3D
    
    L = layer_number
    # Start with first layer
    pool_stack = [myavgpool(pool_size=deep_supervision_scales[layer]) for layer in range(1, L)]
    
    inputs = layers.Input(shape=input_shape)
    outputs = [inputs for _ in range(L - 1)]
    for layer, pool in enumerate(pool_stack):
        outputs[layer] = pool(outputs[layer])
    
    model = Model(inputs, outputs)
    return model

def build_natural_image_model(nclass, input_channels=3):
    """
    Only apply to 2D natural images
    """
    myconv = layers.Conv2D
    
    backbone = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(None, None, 3)
    )
    if input_channels != 3:
        backbone = modify_input_channels(backbone, input_channels)
    
    x = backbone.get_layer("conv4_block6_2_relu").output
    x = aspp_block(x, filters=256, dim=2)

    input_a = layers.UpSampling2D(size=4, interpolation="bilinear")(x)

    input_b = backbone.get_layer("conv2_block3_2_relu").output
    input_b = conv_module(48, 1, strides=1, myconv=myconv, use_bias=False, activation='relu', norm='batchnorm')(input_b)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = conv_module(256, 3, strides=1, myconv=myconv, use_bias=False, activation='relu', norm='batchnorm')(x)
    x = conv_module(256, 3, strides=1, myconv=myconv, use_bias=False, activation='relu', norm='batchnorm')(x)

    x = layers.UpSampling2D(size=4, interpolation="bilinear")(x)
    outputs = conv_module(nclass, 1, strides=1, myconv=myconv, plain=True, name='logits_0')(x)
    return Model(inputs=backbone.inputs, outputs=outputs)

def build_unet(
    im_size, nclass, strides_list=None,
    strides_list_mode='symmetric',
    input_channels=1, features_root=32,
    conv_size=3, deconv_size=2,
    layer_number=5, max_filters=320, 
    dilation=False, attention=False,
    deep_supervision=True, 
    use_upsampling=False, 
    use_residual_encoder=False, 
    freeze_input_shape=False,
    classifier_head=False,
    classifier_nclass=1,
    # Time embedding options (for Diffusion models)
    use_temb_encoder=False,    # 是否使用时间嵌入
    max_timesteps=1000,          # 最大时间步数 T
    # time_embedding_dim=None,     # 时间嵌入维度，默认为 features_root * 4
    dropout=0.0,                 # dropout概率
    **kwargs
):
    if freeze_input_shape:
        input_shape = [s for s in im_size] + [input_channels,]
    else:
        input_shape = [None for _ in im_size] + [input_channels,]
    dim = len(im_size)
    
    # inputshape = (bs, d, h, w, ch) bs: batchsize; d: depth; h: height; w: width; ch: channel
    f = features_root
    k = conv_size
    de_k = deconv_size
    L = layer_number
    max_filters = max_filters
    
    # Time embedding setup
    time_emb_model = None
    if use_temb_encoder:
        time_embedding_dim = f * 4  # 默认为 features_root * 4
        time_emb_model = time_embedding_module(max_timesteps, f, time_embedding_dim)
    
    if dim == 3:
        myconv = layers.Conv3D
    elif dim == 2:
        myconv = layers.Conv2D
    else:
        raise ValueError('Dimension must be 2 or 3') 
    if use_upsampling:
        up_mode = 'upsampling'
    else:
        up_mode = 'deconv'
    
    if use_temb_encoder:    # \
        down_mode = 'time_resnet'
    elif use_residual_encoder:
        down_mode = 'resnet'
    else:
        down_mode = None
    
    # --- Auto calculate model structure parameters ---
    # conv/deconv stride by im_size
    # e.g. [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    if strides_list is None:
        s_list = np.array(get_strides_list(L, im_size, strides_list_mode))
    else:
        s_list = np.array(strides_list)
    
    # conv kernel size. If stride is 1, kernel size is 1. When kernel size become >1 value, it will be fixed.
    # e.g. [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]  conv_size=3
    kernel_size_1 = np.stack([np.amax(s_list[:l+1, :], axis=0) for l in range(len(s_list))]) == 1
    k_list = np.ones_like(s_list) * k
    k_list[kernel_size_1] = 1
    
    # deconv kernel size
    # e.g. [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]  deconv_size=2
    de_k_list = np.ones_like(s_list) * de_k
    de_k_list[s_list == 1] = 1
        
    # Regardless of any model, encoder_0 uses stride=1 and bottom uses kernel_size=k.
    # Top Encoding Layer: 0.
    in_ch, out_ch = input_channels, f
    
    encoder_stack = [downsample_module(in_ch, out_ch, kernel_size=k_list[0], strides=1, dim=dim,
                                       mode=down_mode, dilation=dilation, 
                                       time_embedding_dim=time_embedding_dim if use_temb_encoder else None,
                                       name='encoding_0')]
    
    # Encoding Layer: 1 to L-1.
    for layer in range(1, L):
        in_ch, out_ch = out_ch, min(f * 2 ** layer, max_filters)
        encoder_stack.append(downsample_module(in_ch, out_ch, k_list[layer], s_list[layer - 1], dim=dim, 
                                               mode=down_mode, dilation=dilation,
                                               time_embedding_dim=time_embedding_dim if use_temb_encoder else None,
                                               name=f'encoding_{layer}'))
    
    # Bottom Encoding Layer: L.
    in_ch, out_ch = out_ch, min(f * 2 ** L, max_filters)
    encoder_stack.append(downsample_module(in_ch, out_ch, k, s_list[-1], dim=dim, 
                                           mode=down_mode, dilation=dilation,
                                           time_embedding_dim=time_embedding_dim if use_temb_encoder else None,
                                           name=f'bottom'))
    
    # Decoding Layer: L-1 to 0.
    decoder_up_stack = []
    decoder_conv_stack = []
    for layer in range(L - 1, -1, -1):
        in_ch, out_ch = out_ch, min(f * 2 ** layer, max_filters)
        decoder_up_stack.append(upsample_module(in_ch, out_ch, de_k_list[layer], s_list[layer], dim=dim,
                                                mode=up_mode, name=f'decoding_{layer}'))
        # double conv block with strides=1.
        decoder_conv_stack.append(downsample_module(out_ch, out_ch, k, 1, dim=dim, name=f'decoding_{layer}_1'))
        # decoder_conv_stack.append(downsample_module(out_ch, out_ch, k, 1, dim=dim,
        #                                             mode=down_mode if use_temb_encoder else None,       ## decoder也temb residual
        #                                             time_embedding_dim=time_embedding_dim if use_temb_encoder else None,
        #                                             name=f'decoding_{layer}_1'))
        ### 原程序decoder也用了resnet(temb)
    # --- Build connections ---
    if use_temb_encoder:
        # 双输入：图像 + 时间步
        x_input = layers.Input(shape=input_shape, name='x_input')
        t_input = layers.Input(shape=(), dtype=tf.int32, name='t_input')
        inputs = [x_input, t_input]
        # 生成时间嵌入
        time_emb = time_emb_model(t_input)  # [batch_size, time_embedding_dim]
        x = x_input
    else:
        # 单输入：仅图像
        inputs = layers.Input(shape=input_shape)
        x = inputs
        time_emb = None

    # Encoder: Downsampling through the model
    skips = []
    for enc in encoder_stack:
        if use_temb_encoder:
            # 如果是时间残差模块，传入时间嵌入
            x = enc([x, time_emb])
        else:
            # 普通模块
            x = enc(x)
        skips.append(x)

    # Get output from encoder
    skips = reversed(skips[:-1])  # ignore bottom layer L

    outputs = []
    if deep_supervision:
        outputs += [tf.Variable(0.) for _ in range(L - 1)]

    # Add classification head and append to the end of outputs
    if classifier_head:
        if dim == 3:
            mypool = layers.GlobalAveragePooling3D
        elif dim == 2:
            mypool = layers.GlobalAveragePooling2D
        cls_out = mypool()(x)
        cls_out = layers.Flatten()(cls_out)
        cls_out = layers.Dropout(0.2)(cls_out)
        cls_out = layers.Dense(classifier_nclass, activation=None, name='classifier_logits')(cls_out)
        outputs.append(cls_out)
    
    # Decoder: Upsampling and establishing the skip connections
    for dec_up, dec_conv, skip, layer in zip(decoder_up_stack, decoder_conv_stack, skips, range(L - 1, -1, -1)):
        x = dec_up(x)
        # Attention block
        if attention:
            # in_channels: [skip connection output in encoder, up output in decoder (gate)]
            in_ch = [skip.get_shape().as_list()[-1], x.get_shape().as_list()[-1]]
            skip = attention_module(in_ch, in_ch[0], dim=dim, name=dec_up.name + '_attn')([skip, x])
        x = layers.Concatenate(axis=-1)([x, skip])
        x = dec_conv(x)
        if deep_supervision and layer < (L - 1):
            outputs[layer] = conv_module(nclass, 1, strides=1, myconv=myconv, plain=True, name=f'logits_{layer}')(x)

    if not deep_supervision:
        outputs = [conv_module(nclass, 1, strides=1, myconv=myconv, plain=True, name='logits_0')(x)] + outputs
    
    unet = Model(inputs, outputs)
    return unet

def depthwise_conv(x, in_c, out_c, k=3, s=1, p=0):
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(k, k),
        strides=(s, s),
        padding='same',
        depth_multiplier=1,
        use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    x = tf.keras.layers.Conv2D(
        filters=out_c,
        kernel_size=(1, 1),
        padding='valid',
        use_bias=False
    )(x)
    return x

def inverted_residual_block(x, in_c, out_c, stride, expansion_factor=2, deconvolve=False):
    assert stride in [1, 2]
    use_skip_connection = (stride == 1)
    ex_c = int(in_c * expansion_factor)
    
    residual = x  # Save input tensor for skip connection
    
    # Expansion phase
    x = tf.keras.layers.Conv2D(
        filters=ex_c,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    
    # Depthwise convolution
    if deconvolve:
        x = tf.keras.layers.UpSampling2D()(x)
        strides = (1, 1)
    else:
        strides = (stride, stride)
    
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=strides,
        padding='same',
        depth_multiplier=1,
        use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    
    # Projection phase
    x = tf.keras.layers.Conv2D(
        filters=out_c,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Skip connection
    if use_skip_connection:
        if in_c != out_c:
            residual = tf.keras.layers.Conv2D(
                filters=out_c,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='valid',
                use_bias=False
            )(residual)
            residual = tf.keras.layers.BatchNormalization()(residual)
        x = tf.keras.layers.Add()([x, residual])
    
    return x

def irb_bottleneck(x, in_c, out_c, n, s, t, d=False):
    for i in range(n):
        stride = s if i == 0 else 1
        x = inverted_residual_block(
            x,
            in_c=in_c if i == 0 else out_c,
            out_c=out_c,
            stride=stride,
            expansion_factor=t,
            deconvolve=d
        )
    return x

def build_mobileunet(im_size=[224, 224,1],nclass=4, expansion=8):

    input_layer = tf.keras.layers.Input(shape=im_size)
    x = input_layer
    
    # Encoder
    x = depthwise_conv(x, im_size[-1], 32, s=2)
    x = irb_bottleneck(x, 32, 16, n=1, s=1, t=expansion)
    x = irb_bottleneck(x, 16, 24, n=2, s=2, t=expansion)
    x = irb_bottleneck(x, 24, 32, n=3, s=2, t=expansion)
    x = irb_bottleneck(x, 32, 64, n=4, s=2, t=expansion)
    x = irb_bottleneck(x, 64, 96, n=3, s=1, t=expansion)
    x = irb_bottleneck(x, 96, 160, n=3, s=1, t=expansion)
    
    # Decoder
    d2 = irb_bottleneck(x, 160, 32, n=1, s=2, t=expansion, d=True)
    d3 = irb_bottleneck(d2, 32, 24, n=1, s=2, t=expansion, d=True)
    d4 = irb_bottleneck(d3, 24, 16, n=1, s=2, t=expansion, d=True)
    d5 = tf.keras.layers.UpSampling2D()(d4)
    out = tf.keras.layers.Conv2D(
        filters=nclass,
        kernel_size=(1, 1),
        strides=(1, 1)
    )(d5)
    
    conv1x1_d2 = tf.keras.layers.Conv2D(
        filters=nclass,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False
    )(d2)
    conv1x1_d3 = tf.keras.layers.Conv2D(
        filters=nclass,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False
    )(d3)
    conv1x1_d4 = tf.keras.layers.Conv2D(
        filters=nclass,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False
    )(d4)
    
    # Define the model
    model = tf.keras.Model(
        inputs=input_layer,
        outputs=(out, conv1x1_d4, conv1x1_d3, conv1x1_d2)
    )
    
    return model

def build_retinaunet(
    im_size, nclass, strides_list=None, 
    strides_list_mode='last',
    input_channels=1, features_root=32,
    conv_size=3, deconv_size=2,
    layer_number=5, max_filters=320, 
    start_level=1, num_head_levels=4, num_anchors=27,
    dilation=False, 
    feature_pyramid=True, 
    use_upsampling=False, 
    use_residual_encoder=False, 
    freeze_input_shape=False,
    **kwargs
):
    if freeze_input_shape:
        input_shape = [s for s in im_size] + [input_channels,]
    else:
        input_shape = [None for _ in im_size] + [input_channels,]
    dim = len(im_size)
    
    # inputshape = (bs, d, h, w, ch) bs: batchsize; d: depth; h: height; w: width; ch: channel
    f = features_root
    k = conv_size
    de_k = deconv_size
    L = layer_number
    max_filters = max_filters
    num_head_levels = min(num_head_levels, L - start_level)

    if dim == 3:
        myconv = layers.Conv3D
    elif dim == 2:
        myconv = layers.Conv2D
    else:
        raise ValueError('Dimension must be 2 or 3')
    if use_upsampling:
        up_mode = 'upsampling'
    else:
        up_mode = 'deconv'
    
    if use_residual_encoder:
        down_mode = 'resnet'
    else:
        down_mode = None

    # --- Auto calculate model structure parameters ---
    # conv/deconv stride by im_size
    # e.g. [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    if strides_list is None:
        s_list = np.array(get_strides_list(L, im_size, strides_list_mode))
    else:
        s_list = np.array(strides_list)
    
    # conv kernel size. If stride is 1, kernel size is 1. When kernel size become >1 value, it will be fixed.
    # e.g. [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]  conv_size=3
    kernel_size_1 = np.stack([np.amax(s_list[:l+1, :], axis=0) for l in range(len(s_list))]) == 1
    k_list = np.ones_like(s_list) * k
    k_list[kernel_size_1] = 1
    # deconv kernel size
    # e.g. [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]  deconv_size=2
    de_k_list = np.ones_like(s_list) * de_k
    de_k_list[s_list == 1] = 1

    # Encoder: collect from top to bottom. Inlcudes stem and downsampling part.
    # Regardless of any model, encoder_stem uses stride=1 and kernel_size=k.
    # Top Encoding Layer: stem layer.
    in_ch, out_ch = input_channels, f
    encoder_stack = [downsample_module(in_ch, out_ch, kernel_size=k, strides=1, dim=dim,
                                       mode=down_mode, dilation=dilation, name='encoder_stem')]
    
    # Encoding Layer: 0 to L-1.
    for layer in range(L):
        in_ch, out_ch = out_ch, min(f * 2 ** (layer + 1), max_filters)
        encoder_stack.append(downsample_module(in_ch, out_ch, k_list[layer], s_list[layer], dim=dim, 
                                               mode=down_mode, dilation=dilation, name=f'encoder_{layer}'))

    # Decoder: collect from bottom to top. Includes lateral and upsampling part.
    # Skip connection (lateral bridge). Conv to get same number of channels as decoder output.
    lateral_conv_stack = [conv_module(min(f * 2 ** (layer + 1), 128), 1, 1, myconv=myconv, dilation=dilation,
                                      name=f'lateral_{layer}', plain=True) for layer in range(L - 1, -1, -1)]
    lateral_conv_out_stack = [conv_module(min(f * 2 ** (layer + 1), 128), k, 1, myconv=myconv, dilation=dilation,
                              name=f'lateral_{layer}_out', plain=True) for layer in range(L - 1, -1, -1)]
    
    # Decoding Layer: L-1 to 0.
    decoder_up_stack = []
    for layer in range(L - 1, -1, -1):
        in_ch, out_ch = out_ch, min(f * 2 ** layer, 128)
        decoder_up_stack.append(upsample_module(in_ch, out_ch, de_k_list[layer], s_list[layer], dim=dim,
                                                mode=up_mode, name=f'decoder_{layer}_up'))
    
    # # For stem
    # decoder_up_stack += [None] # no upsampling
    # lateral_conv_stack = [conv_module(min(f, 128), 1, 1, myconv=myconv, dilation=dilation,
    #                                   name='lateral_stem', plain=True)] + lateral_conv_stack
    # lateral_conv_out_stack = [conv_module(min(f, 128), k, 1, myconv=myconv, dilation=dilation,
    #                                       name=f'lateral_stem_out', plain=True)] + lateral_conv_out_stack
    
    # --- Build connections ---
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Encoder: Downsampling through the model
    skips = []
    for enc in encoder_stack:
        x = enc(x)
        skips.append(x)
    
    # Get output from encoder and connect to decoder lateral part
    skips = reversed(skips[1:])  # ignore stem layer
    lateral_skips = []
    for skip, lateral in zip(skips, lateral_conv_stack):
        x = lateral(skip)
        lateral_skips.append(x)

    if feature_pyramid:
        outputs = [tf.Variable(0.) for _ in range(L)]
        
    # Decoder: Upsampling and establishing the skip connections
    for lateral_skip, dec_up, lateral_conv_out, layer in zip(lateral_skips,
                                                             decoder_up_stack,
                                                             lateral_conv_out_stack, 
                                                             range(L - 1, -1, -1)):
        x = lateral_skip
        if layer != L - 1:
            x = layers.Add(name=f'add_lateral_{layer}_decoder_up_{layer + 1}')([x, last_dec_up_output])
        # NOTE: upsampled result is not connected to output in current layer (e.g layer 3),
        #       but to upper layer (e.g. layer 2)
        if dec_up is not None:
            last_dec_up_output = dec_up(x)
        x = lateral_conv_out(x)
        if feature_pyramid:
            head_output = []
            # Connect to Classifier Head
            cls_out = conv_module(128, k, 1, myconv=myconv, use_bias=True, norm='groupnorm', 
                                 name=f'head_classifier_{layer}_0')(x)
            cls_out = conv_module(128, k, 1, myconv=myconv, use_bias=True, norm='groupnorm', 
                                 name=f'head_classifier_{layer}_1')(cls_out)
            prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
            cls_out = conv_module(num_anchors * nclass, k, 1, myconv=myconv, use_bias=True, 
                                 bias_init=prior_probability, plain=True, 
                                 name=f'classifer_logits_{layer}')(cls_out)
            head_output.append(layers.Reshape([-1, nclass], name=f'classifier_logits_reshape_{layer}')(cls_out))
            # Connect to Regressor Head
            reg_out = conv_module(128, k, 1, myconv=myconv, use_bias=True, norm='groupnorm', 
                                 name=f'head_regressor_{layer}_0')(x)
            reg_out = conv_module(128, k, 1, myconv=myconv, use_bias=True, norm='groupnorm', 
                                 name=f'head_regressor_{layer}_1')(reg_out)
            reg_out = conv_module(num_anchors * 2 * dim,  # e.g. 2 * 3: x, y, z, x_range, y_range, z_range
                                 k, 1, myconv=myconv, use_bias=True, plain=True, 
                                 name=f'regressor_logits_{layer}')(reg_out)
            head_output.append(layers.Reshape([-1, 2 * dim], name=f'regressor_logits_reshape_{layer}')(reg_out))
            head_output = layers.Concatenate(axis=-1, name=f'head_output_{layer}')(head_output)
            outputs[layer] = head_output
            
    if not feature_pyramid:
        head_output = []
        # Connect to Classifier Head
        cls_out = conv_module(128, k, 1, myconv=myconv, use_bias=True, norm='groupnorm', 
                             name=f'head_classifier_0')(x)
        cls_out = conv_module(128, k, 1, myconv=myconv, use_bias=True, norm='groupnorm', 
                             name=f'head_classifier_1')(cls_out)
        cls_out = conv_module(num_anchors * nclass, 
                             k, 1, myconv=myconv, use_bias=True, plain=True, 
                             name=f'classfier_logits')(cls_out)
        head_output.append(layers.Reshape([-1, nclass], name=f'classifier_logits_reshape')(cls_out))
        # Connect to Regressor Head
        reg_out = conv_module(128, k, 1, myconv=myconv, use_bias=True, norm='groupnorm', 
                             name=f'head_regressor_0')(x)
        reg_out = conv_module(128, k, 1, myconv=myconv, use_bias=True, norm='groupnorm', 
                             name=f'head_regressor_1')(reg_out)
        reg_out = conv_module(num_anchors * 2 * dim,  # e.g. 2 * 3: x, y, z, x_range, y_range, z_range
                             k, 1, myconv=myconv, use_bias=True, plain=True, 
                             name=f'regressor_logits')(reg_out)
        head_output.append(layers.Reshape([-1, 2 * dim], name=f'regressor_logits_reshape')(reg_out))
        head_output = layers.Concatenate(axis=-1, name='outputs')(head_output)
        outputs = head_output
    else:
        # Stack output for different level
        outputs = layers.Concatenate(axis=1, name='outputs')(
            outputs[start_level : start_level + num_head_levels])
    networks = Model(inputs, outputs, name='retinaunet')
    return networks

def conv_module(filters, kernel_size=3, strides=1, groups=1, dilation=False,
               norm='instancenorm', activation='leakyrelu', myconv=layers.Conv3D,
               name=None, use_bias=False, plain=False, **kwargs):        
    initializer = tf.keras.initializers.HeNormal()

    result = Sequential(name=name)
    if isinstance(dilation, bool):
        if dilation:
            dilation_rate = 2
        else:
            dilation_rate = 1
    else: # (int, list, tuple)
        dilation_rate = dilation
    
    result.add(myconv(filters=filters, kernel_size=kernel_size, 
                      strides=strides, dilation_rate=dilation_rate,
                      padding='same', groups=groups,
                      kernel_initializer=initializer, use_bias=use_bias))
    if plain:
        return result
    
    if norm == 'instancenorm':
        result.add(GroupNormalization(groups=-1))
    elif norm == 'batchnorm':
        result.add(layers.BatchNormalization())
    elif norm == 'layernorm':
        result.add(layers.LayerNormalization())
    elif norm == 'groupnorm':
        norm_channels_per_group = 16
        result.add(GroupNormalization(groups=norm_channels_per_group))
    
    if activation == 'leakyrelu':
        result.add(layers.LeakyReLU(0.01))
    elif activation == 'relu':
        result.add(layers.ReLU())
    elif activation == 'swish':
        result.add(layers.Activation('swish'))
    return result

def upsample_module(in_filters=None, out_filters=256, kernel_size=2, strides=2, dim=2,
                    mode='deconv', name=None, **kwargs):
    result = Sequential(name=name)
    if mode == 'deconv':
        # ConvTranspose
        if dim == 2:
            mydeconv = layers.Conv2DTranspose
        elif dim == 3:
            mydeconv = layers.Conv3DTranspose
        result.add(conv_module(out_filters, kernel_size, strides=strides, myconv=mydeconv, plain=True))
    else:
        # Upsampling + Conv
        if dim == 2:
            myup = layers.UpSampling2D
            myconv = layers.Conv2D
        elif dim == 3:
            myup = layers.UpSampling3D
            myconv = layers.Conv3D
        result.add(myup(size=strides, interpolation="bilinear"))
        result.add(conv_module(out_filters, kernel_size, strides=1, myconv=myconv))
    return result

def downsample_module(in_filters=None, out_filters=256, kernel_size=3, strides=2, dim=2, groups=1,
                     dilation=False, mode=None, time_embedding_dim=None, name=None, **kwargs):
    if dim == 2:
        myconv = layers.Conv2D
    elif dim == 3:
        myconv = layers.Conv3D
    result = Sequential(name=name)
    
    if isinstance(mode, str) and 'resnet' in mode:
        se = False

        if mode == 'time_resnet':
            # Time ResNet module - 需要特殊处理，返回支持时间嵌入的模块
            if time_embedding_dim is None:
                time_embedding_dim = out_filters * 4  # 默认值
            return(time_resnet_module(in_filters, out_filters, time_embedding_dim, kernel_size, strides=strides, dim=dim))

        if mode.startswith('se'):
            se = True
        # SEResNeXt module
        if mode.endswith('xt'):
            result.add(seresnext_module(in_filters, out_filters, kernel_size, strides=strides,
                                        dim=dim, reduction=4, groups=groups, se=se))
            result.add(seresnext_module(in_filters, out_filters, kernel_size, strides=1,
                                        dim=dim, reduction=4, groups=groups, se=se))
            result.add(seresnext_module(in_filters, out_filters, kernel_size, strides=1,
                                        dim=dim, reduction=4, groups=groups, se=se))
        else:   ##  普通的 ResNet 模块
            result.add(resnet_module(in_filters, out_filters, kernel_size, strides=strides, dim=dim))
    else:
        result.add(conv_module(out_filters, kernel_size, strides=strides, myconv=myconv, use_bias=True))
        result.add(conv_module(out_filters, kernel_size, strides=1, dilation=dilation, myconv=myconv, use_bias=True))
    
    return result

def resnet_module(in_filters, out_filters, kernel_size=3, strides=1, dim=3, activation='leakyrelu'):
    if dim == 2:
        myconv = layers.Conv2D
        mypool = layers.MaxPooling2D
        norm = 'batchnorm'
    elif dim == 3:
        myconv = layers.Conv3D
        mypool = layers.MaxPooling3D
        norm = 'instancenorm'
    
    inputs = layers.Input([None for _ in range(dim)] + [in_filters,])
    
    x = inputs
    y = mypool(pool_size=strides)(x)
    x = conv_module(out_filters, kernel_size, strides=1, myconv=myconv,
                    use_bias=False, norm=norm, activation=activation)(y)
    x = conv_module(out_filters, kernel_size, strides=1, myconv=myconv, 
                    use_bias=False, norm=norm, activation=None)(x)
    
    y = conv_module(out_filters, 1, strides=1, myconv=myconv, 
                    use_bias=False, norm=norm, activation=None)(y)
    x = layers.Add()([x, y])
    x = layers.ReLU()(x)

    result = Model(inputs, x)
    return result


############# 20250630
def time_resnet_module(in_filters, out_filters, time_embedding_dim, kernel_size=3, strides=1, 
                      dim=2, dropout=0.0):
    """
    time_embedding_dim: [(batch_size, time_embedding_dim)]
    """
    if dim == 2:
        myconv = layers.Conv2D
        mypool = layers.MaxPooling2D
    elif dim == 3:
        myconv = layers.Conv3D
        mypool = layers.MaxPooling3D
    else:
        raise ValueError('Dimension must be 2 or 3')
    
    norm = 'groupnorm'  # 使用GroupNormalization
    activation = 'swish'  # 使用Swish激活函数
    
    # 输入：图像特征和时间嵌入
    x_input = layers.Input([None for _ in range(dim)] + [in_filters,], name='feature_input')
    t_input = layers.Input([time_embedding_dim,], name='time_input')
    
    x = x_input
    y = mypool(pool_size=strides)(x)

    # 第一个卷积块
    x = conv_module(out_filters, kernel_size, strides=1, myconv=myconv, 
                    use_bias=False, norm=norm, activation=activation)(y)

    # time_embedding: SiLU + Linear
    t_emb = layers.Activation('swish')(t_input)
    t_emb = layers.Dense(out_filters, use_bias=True)(t_emb)   # [bs, out_filters]

    # 将时间嵌入添加到特征图 (广播到空间维度)
    if dim == 2:
        t_emb = layers.Reshape([1, 1, out_filters])(t_emb)  # 2D: [bs, 1, 1, out_filters]
    elif dim == 3:
        t_emb = layers.Reshape([1, 1, 1, out_filters])(t_emb) # 3D: [bs, 1, 1, 1, out_filters]
    
    x = layers.Add()([x, t_emb])  # h += time_emb
    
    # 第二个卷积块
    x = conv_module(out_filters, kernel_size, strides=1, myconv=myconv, 
                    use_bias=False, norm=norm, activation=activation)(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
        

    # 跳跃连接 (shortcut) - 可以使用conv_module
    if in_filters != out_filters:
        # 使用conv_module来处理shortcut，plain=True表示只有卷积没有norm和激活
        y = conv_module(out_filters, 1, strides=1, myconv=myconv,       # self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=1, padding=0)
                              use_bias=False, plain=True)(y)
    else:
        y = layers.Lambda(lambda x: x)(y)  # 如果输入输出通道相同，直接使用输入
    
    
    # 残差连接
    output = layers.Add()([x, y])
    
    result = Model(inputs=[x_input, t_input], outputs=output)
    return result


def seresnext_module(in_filters, out_filters, kernel_size=3, strides=1, dim=2,
                    reduction=4, groups=1, se=False):
    if dim == 2:
        myconv = layers.Conv2D
    elif dim == 3:
        myconv = layers.Conv3D
    
    inputs = layers.Input([None for _ in range(dim)] + [in_filters,])
    
    x = inputs
    x = conv_module(out_filters // reduction, 1, strides=1, myconv=myconv, 
                   use_bias=False, norm='batchnorm', activation='relu')(x)
    x = conv_module(out_filters // reduction, kernel_size, strides=strides, myconv=myconv, groups=groups, 
                   use_bias=False, norm='batchnorm', activation='relu')(x)
    x = conv_module(out_filters, kernel_size, strides=1, myconv=myconv, 
                   use_bias=False, norm='batchnorm', activation=None)(x)

    y = inputs
    if in_filters != out_filters or strides != 1:
        y = conv_module(out_filters, 1, strides=strides, myconv=myconv, 
                       use_bias=False, norm='batchnorm', activation=None)(y)
    x = layers.Add()([x, y])
    x = layers.ReLU()(x)

    if se:
        x = squeeze_excite_block(x, dim=dim)

    result = Model(inputs, x)
    return result

def attention_module(xg_in_filters, out_filters, dim=2, name=None):
    """
    x: skip connection output in encoder.
    g: gate, up output in decoder.
    """
    if dim == 2:
        myconv = layers.Conv2D
    elif dim == 3:
        myconv = layers.Conv3D
    
    initializer = tf.keras.initializers.HeNormal()

    inputs = [layers.Input([None for _ in range(dim)] + [xg_in_filters[0],]),
              layers.Input([None for _ in range(dim)] + [xg_in_filters[1],])]
    x, g = inputs

    theta_x = myconv(out_filters, 1, strides=1, kernel_initializer=initializer, use_bias=True)(x)
    phi_g = myconv(out_filters, 1, strides=1, kernel_initializer=initializer, use_bias=True)(g)

    xg = layers.Add()([theta_x, phi_g])
    xg = layers.ReLU()(xg)
    xg = myconv(1, 1, strides=1, kernel_initializer=initializer, use_bias=True)(xg)
    rate = layers.Activation('sigmoid')(xg)

    attn_x = layers.Multiply()([x, rate])

    result = Model(inputs, attn_x, name=name)
    return result

def squeeze_excite_block(x, dim=2, ratio=8):
    if dim == 2:
        mygap = layers.GlobalAveragePooling2D
    elif dim == 3:
        mygap = layers.GlobalAveragePooling3D
    
    initializer = tf.keras.initializers.HeNormal()

    inputs = x
    filters = inputs.shape[-1]
    x = mygap()(inputs)
    x = layers.Dense(filters // ratio, kernel_initializer=initializer, use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(filters, kernel_initializer=initializer, use_bias=False)(x)
    rate = layers.Activation('sigmoid')(x)
    x = layers.Multiply()([rate, inputs])
    return x

def channel_attention_block(x, dim=2, ratio=8):
    if dim == 2:
        mygap = layers.GlobalAveragePooling2D
        mygmp = layers.GlobalMaxPooling2D
    elif dim == 3:
        mygap = layers.GlobalAveragePooling3D
        mygmp = layers.GlobalMaxPooling3D
        
    initializer = tf.keras.initializers.HeNormal()
    
    inputs = x
    filters = x.shape[-1]
    keep_shape = (1, 1, filters)

    gap = mygap()(x)
    gmp = mygmp()(x)
    shared_mlp1 = layers.Dense(filters // ratio, activation='relu', kernel_initializer=initializer, use_bias=False)
    shared_mlp2 = layers.Dense(filters, activation='sigmoid', kernel_initializer=initializer, use_bias=False)
    x = layers.Add()([shared_mlp2(shared_mlp1(gap)), shared_mlp2(shared_mlp1(gmp))])
    x = layers.Reshape(keep_shape)(x)
    x = layers.Multiply()([inputs, x])
    return x

def spacial_attention_block(x, g=None, kernel_size=3, dim=2, ratio=8):
    if dim == 2:
        myconv = layers.Conv2D
    elif dim == 3:
        myconv = layers.Conv3D
    
    initializer = tf.keras.initializers.HeNormal()
    
    inputs = x
    if g is None:
        g = x

    channel_ap = tf.reduce_max(g, axis=-1, keepdims=True)
    channel_mp = tf.reduce_mean(g, axis=-1, keepdims=True)
    channel_pool = layers.Concatenate(axis=-1)([channel_ap, channel_mp])
    
    x = myconv(1, kernel_size=kernel_size, padding="same", kernel_initializer=initializer)(channel_pool)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('sigmoid')(x)
    
    x = layers.Multiply()([inputs, x])
    return x

def aspp_block(x, filters=256, dim=2, rate_scale=1):
    if dim == 2:
        myconv = layers.Conv2D
    elif dim == 3:
        myconv = layers.Conv3D
    
    x0 = squeeze_excite_block(x, dim=dim)

    x1 = conv_module(filters, 3, strides=1, myconv=myconv, dilation=6*rate_scale, norm='batchnorm', activation='relu')(x)

    x2 = conv_module(filters, 3, strides=1, myconv=myconv, dilation=12*rate_scale, norm='batchnorm', activation='relu')(x)

    x3 = conv_module(filters, 3, strides=1, myconv=myconv, dilation=18*rate_scale, norm='batchnorm', activation='relu')(x)

    x4 = conv_module(filters, 3, strides=1, myconv=myconv, dilation=False, norm='batchnorm', activation='relu')(x)

    x = layers.Add()([x0, x1, x2, x3, x4])
    x = conv_module(filters, 1, strides=1, myconv=myconv, norm='batchnorm', activation='relu')(x)
    return x







############# Time Embedding Components (Added for Diffusion Models) #############

def create_sinusoidal_time_embedding(timesteps, dim):
    """
    创建正弦时间嵌入，对应PyTorch版本的TimeEmbedding预计算部分
    
    Args:
        timesteps: 时间步张量 [batch_size] 或标量
        dim: 嵌入维度，必须是偶数
    
    Returns:
        时间嵌入张量 [batch_size, dim]
    """
    assert dim % 2 == 0, "Embedding dimension must be even"
    
    half_dim = dim // 2
    emb = tf.math.log(10000.0) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    
    timesteps = tf.cast(timesteps, tf.float32)
    emb = timesteps[:, None] * emb[None, :]
    
    # 拼接sin和cos
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    
    return emb

def time_embedding_module(max_timesteps, d_model, output_dim):
    """
    时间嵌入模块，对应PyTorch的TimeEmbedding类
    
    Args:
        max_timesteps: 最大时间步数 T
        d_model: 正弦嵌入的维度
        output_dim: 输出维度 (通常是 model_channels * 4)
    
    Returns:
        Keras Model，输入时间步索引，输出时间嵌入
    """
    assert d_model % 2 == 0, "d_model must be even"
    
    # 输入：时间步索引
    t_input = layers.Input(shape=(), dtype=tf.int32, name='timestep')
    
    # 预计算正弦嵌入表
    timesteps = tf.range(max_timesteps, dtype=tf.float32)
    emb_table = create_sinusoidal_time_embedding(timesteps, d_model)
    
    # 嵌入查找（不可训练）
    emb = layers.Embedding(
        max_timesteps, d_model, 
        weights=[emb_table], 
        trainable=False,
        name='sinusoidal_embedding'
    )(t_input)
    
    # MLP变换 (对应PyTorch的nn.Sequential部分)
    emb = layers.Dense(
        output_dim, 
        kernel_initializer='glorot_uniform',
        name='time_dense_1'
    )(emb)
    emb = layers.Activation('swish', name='time_swish')(emb)
    emb = layers.Dense(
        output_dim, 
        kernel_initializer='glorot_uniform',
        name='time_dense_2'
    )(emb)
    
    return Model(t_input, emb, name='time_embedding')

############# Usage Examples for Time Embedding #############
if __name__ == "__main__":
  def create_diffusion_unet_example():
      """
      创建带有时间嵌入的UNet示例，用于Diffusion模型
      """
      # 普通UNet（不使用时间嵌入）
      normal_unet = build_unet(
          im_size=[64, 64],
          nclass=1,
          input_channels=1,
          features_root=64,
          layer_number=4,
          deep_supervision=False, # 不使用深度监督
          use_temb_encoder=False,  # 关闭时间嵌入
          use_residual_encoder=True, # 使用时间残差编码器
      )
      
      # Diffusion UNet（使用时间嵌入）
      diffusion_unet = build_unet(
          im_size=[64, 64],
          nclass=1,
          input_channels=2,
          features_root=64,
          layer_number=4,
          deep_supervision=False, # 不使用深度监督
          use_temb_encoder=True,     # 开启时间嵌入
          max_timesteps=1000,          # 最大时间步数
          time_embedding_dim=256,      # 时间嵌入维度（默认为features_root*4=256）
          dropout=0.1,                 # dropout概率
          use_residual_encoder=False   # 使用时间残差编码器
      )
      
      return normal_unet, diffusion_unet



  # 使用时在外部拼接
  x = tf.random.normal([2, 64, 64, 1])  # 主图像
  c = tf.random.normal([2, 64, 64, 1])  # context图像
  t = tf.constant([100, 200])               # 时间步，批次大小要匹配

  normal_unet, diffusion_unet = create_diffusion_unet_example()

  output1 = normal_unet(x)
  print(output1.shape)  # [2, 64, 64, 1]
  # 在调用模型前手动拼接
  input_concat = tf.concat([x, c], axis=-1)
  output2 = diffusion_unet([input_concat, t]) 
  print(output2.shape)  # [2, 64, 64, 1]