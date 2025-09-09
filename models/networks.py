import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.initializers import GlorotUniform, Zeros, VarianceScaling

def TimeEmbedding(x, embed_dim, scale=30., W=None):
    """
    x: [batch] int32 或 float32
    embed_dim: 嵌入维度，偶数
    scale: 权重缩放
    W: 可选，固定的 shape=[embed_dim//2] 的向量，若为None则自动生成
    返回: [batch, embed_dim]
    """
    if W is None:
        W = tf.random.normal([embed_dim // 2], stddev=scale)
    W = tf.stop_gradient(W)
    x = tf.cast(x, tf.float32)
    x_proj = x[:, None] * W[None, :] * 2 * np.pi  # [batch, embed_dim//2]
    sin = tf.sin(x_proj)
    cos = tf.cos(x_proj)
    emb = tf.concat([sin, cos], axis=-1)  # [batch, embed_dim]
    return emb

def time_resnet_module(in_filters, out_filters, time_embedding_dim, dim=2, dropout=0.1, attn=False, name=None,**kwargs):
    """
    time_embedding_dim: [(batch_size, time_embedding_dim)]
    """
    if dim == 2:
        myconv = layers.Conv2D
    elif dim == 3:
        myconv = layers.Conv3D
    
    x_input = layers.Input([None for _ in range(dim)] + [in_filters,], name='feature_input')
    t_input = layers.Input([time_embedding_dim,], name='time_input')
    
    h = x_input

    # block 1
    h = GroupNormalization(groups=32, axis=-1)(h) 
    h = layers.Activation('swish')(h) 
    h = myconv(out_filters, kernel_size=3, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(h)

    # time_embedding: SiLU + Linear
    t_emb = layers.Activation('swish')(t_input)
    t_emb = layers.Dense(out_filters, kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(t_emb)

    if dim == 2:
        t_emb = layers.Reshape([1, 1, out_filters])(t_emb)  # 2D: [bs, 1, 1, out_filters]
    elif dim == 3:
        t_emb = layers.Reshape([1, 1, 1, out_filters])(t_emb) # 3D: [bs, 1, 1, 1, out_filters]

    h = layers.Add()([h, t_emb])

    # block 2
    h = GroupNormalization(groups=32, axis=-1)(h)
    h = layers.Activation('swish')(h)
    h = layers.Dropout(dropout)(h)
    h = myconv(out_filters, kernel_size=3, strides=1, padding='same',
               kernel_initializer=VarianceScaling(scale=1e-5), bias_initializer=Zeros())(h)

    if in_filters != out_filters:
        shortcut = myconv(out_filters, kernel_size=1, strides=1, padding='valid', 
                   kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(x_input)
    else:
        shortcut = tf.identity(x_input)
    # if attn:
    #     x = attention_module([in_filters, out_filters], out_filters, dim=dim, name='attn')(x)

    h = layers.Add()([h, shortcut])

    result = Model(inputs=[x_input, t_input], outputs=h, name=name)

    return result


def DownSample_Module_diff(in_filters, dim, tdim, name=None, **kwargs):
    if dim == 2:
        myconv = layers.Conv2D
    elif dim == 3:
        myconv = layers.Conv3D
    else:
        raise ValueError('Dimension must be 2 or 3')

    x_input = layers.Input([None for _ in range(dim)] + [in_filters,], name='feature_input')
    t_input = layers.Input([tdim,], name='time_input')

    x = x_input
    x = myconv(in_filters, kernel_size=3, strides=2, padding='same', 
               kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(x)
    
    result = Model(inputs=[x_input, t_input], outputs=x, name=name)
    return result


def Upsample_module_diff(in_filters, dim, tdim, name=None, **kwargs):
    if dim == 2:
        myconv = layers.Conv2D
        myup = lambda: layers.UpSampling2D(size=2, interpolation='nearest')
    elif dim == 3:
        myconv = layers.Conv3D
        myup = lambda: layers.UpSampling3D(size=2)
    else:
        raise ValueError('Dimension must be 2 or 3')

    x_input = layers.Input([None for _ in range(dim)] + [in_filters,], name='feature_input')
    t_input = layers.Input([tdim,], name='time_input')
    x = x_input

    x = myup()(x)
    x = myconv(in_filters, kernel_size=3, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(x)

    result = Model(inputs=[x_input, t_input], outputs=x, name=name)
    return result


def build_diffusion_unet(
    im_size, nclass,
    input_channels=1, features_root=32,
    layer_number=5, max_filters=320,
    attention=False,
    freeze_input_shape=False,
    num_res_blocks = 2,
    dropout=0.1,
    **kwargs
):
    """
    Build a Buffusion UNet model.
    """
    if freeze_input_shape:
        input_shape = [s for s in im_size] + [input_channels,]
    else:
        input_shape = [None for _ in im_size] + [input_channels,]
    dim = len(im_size)
    f = features_root
    L = layer_number
    max_filters = max_filters

    if dim == 3:
        myconv = layers.Conv3D
    elif dim == 2:
        myconv = layers.Conv2D
    else:
        raise ValueError('Dimension must be 2 or 3') 
    
    # Regardless of any model, encoder_0 uses stride=1 and bottom uses kernel_size=k.
    # Top Encoding Layer: 0.
    in_ch, out_ch = input_channels, f
    tdim = f * 4
    
    ch_mult = [2 ** i for i in range(L)]    # len(ch_mult) = L
    
    if attention is False or attention is None:
      attention = []
    assert all([i < len(ch_mult) for i in attention]), 'attn index out of bound'

    chs = [f]  # record output channel when dowmsample for upsample
    now_ch = f
    # head
    head = Sequential([myconv(f, kernel_size=3, strides=1, padding='same',
                  kernel_initializer=GlorotUniform(), bias_initializer=Zeros())], name='head')
    # Encoder
    encoder_stack = []
    for i, mult in enumerate(ch_mult):
        out_ch = min(f * mult, max_filters)
        for j in range(num_res_blocks):
            encoder_stack.append(time_resnet_module(in_filters=now_ch, out_filters=out_ch, time_embedding_dim=tdim,
                                                    dim=dim, dropout=dropout, attn=(i in attention),name=f'encoder_res_{i}_{j}'))
            now_ch = out_ch
            chs.append(now_ch)
        if i != len(ch_mult) - 1:
            encoder_stack.append(DownSample_Module_diff(in_filters=now_ch, dim=dim, tdim=tdim, name=f'encoder_downsample_{i}'))
            chs.append(now_ch)
    
    # Middle
    middle_stack = []
    middle_stack.append(time_resnet_module(in_filters=now_ch, out_filters=now_ch, time_embedding_dim=tdim,
                                           dropout=dropout, dim=dim, attn=False))
    middle_stack.append(time_resnet_module(in_filters=now_ch, out_filters=now_ch, time_embedding_dim=tdim,
                                           dropout=dropout, dim=dim, attn=False))

    # Decoder
    decoder_stack = []
    for i, mult in reversed(list(enumerate(ch_mult))):
        out_ch = min(f * mult, max_filters)
        for j in range(num_res_blocks + 1):
            decoder_stack.append(time_resnet_module(
                in_filters=chs.pop() + now_ch,
                out_filters=out_ch,
                time_embedding_dim=tdim,
                dim=dim,
                dropout=dropout,
                attn=(i in attention),
                name=f'decoder_res_{i}_{j}'
            ))
            now_ch = out_ch
        if i != 0:
            decoder_stack.append(Upsample_module_diff(in_filters=now_ch, dim=dim, tdim=tdim, name=f'decoder_upsample_{i}'))
    assert len(chs) == 0, 'Channel list should be empty after decoder stack'

    # Tail
    tail = Sequential([
        GroupNormalization(groups=32, axis=-1),
        layers.Activation('swish'),
        myconv(nclass, kernel_size=3, strides=1, padding='same',
               kernel_initializer=VarianceScaling(scale=1e-5, mode='fan_avg', distribution='uniform'),
               bias_initializer=Zeros())
    ], name='tail')


   # --- Build connections ---
    x_input = layers.Input(shape=input_shape, name='x_input')
    t_input = layers.Input(shape=(), dtype=tf.int32, name='t_input')
    input = [x_input, t_input]
    temb = TimeEmbedding(x=t_input, embed_dim=tdim)  # [batch_size, tdim]
    # Encoder: Downsampling
    x = x_input
    h = head(x)
    hs = [h]
    for layer in encoder_stack: 
        h = layer([h, temb])
        hs.append(h)
    # Middle
    for layer in middle_stack:
        h = layer([h, temb])
    # Decoder:  Upsampling
    for layer in decoder_stack: 
        if hasattr(layer, "name") and layer.name.startswith("decoder_res"):
            h = tf.concat([h, hs.pop()], axis=-1)
        h = layer([h, temb])
    output = tail(h) 
    assert len(hs) == 0, 'Channel list should be empty after decoder stack'

    unet = Model(input, output)
    return unet
