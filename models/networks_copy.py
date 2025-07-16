import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.initializers import GlorotUniform, Zeros, VarianceScaling

"""
There is no custom layer here. 
All built models can be saved by model.save('model.h5') and loaded by load_model('model.h5').

build_*   - a ready-to-use built model.
*_module  - a class-level module. The usage is like layers.Add()([a, b])
*_block   - a functional-level block. The usage is like layers.add([a, b])
"""

''' reference: https://github.com/QingLyu0828/diffusion_mri_to_ct_conversion/blob/main/DDPM/model.py '''
############# Time Embedding Components  #############
######### 20240714
# import math
# def TimeEmbedding(t, T, d_model,dim):
#     # 构造正弦时间嵌入
#     assert d_model % 2 == 0
#     emb = tf.range(0, d_model, delta=2, dtype=tf.float32) / d_model * math.log(10000)
#     emb = tf.exp(-emb)
#     pos = tf.range(T, dtype=tf.float32)
#     emb = pos[:, None] * emb[None, :]
#     assert list(emb.shape) == [T, d_model // 2]
#     emb = tf.stack([tf.sin(emb), tf.cos(emb)], axis=-1)
#     assert list(emb.shape) == [T, d_model // 2, 2]
#     emb = tf.reshape(emb, [T, d_model])

#     # 查表
#     t = tf.cast(t, tf.int32)
#     # temb = tf.gather(emb, t)  # [batch_size, d_model]
#     temb = tf.keras.layers.Embedding

#     # 两层MLP + Swish
#     temb = tf.keras.layers.Dense(dim, kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(temb)
#     temb = layers.Activation('swish')(temb)
#     temb = tf.keras.layers.Dense(dim, kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(temb)

#     return temb

def TimeEmbedding(x, embed_dim, scale=30., W=None):
    """
    x: [batch] int32 或 float32
    embed_dim: 嵌入维度，偶数
    scale: 权重缩放
    W: 可选，固定的 shape=[embed_dim//2] 的向量，若为None则自动生成
    返回: [batch, embed_dim]
    """
    if W is None:
        # 固定随机权重，和PyTorch一致
        W = tf.random.normal([embed_dim // 2], stddev=scale)
    # 保证W不被训练
    W = tf.stop_gradient(W)
    x = tf.cast(x, tf.float32)
    x_proj = x[:, None] * W[None, :] * 2 * np.pi  # [batch, embed_dim//2]
    sin = tf.sin(x_proj)
    cos = tf.cos(x_proj)
    emb = tf.concat([sin, cos], axis=-1)  # [batch, embed_dim]
    return emb


############# 20250714
def time_resnet_module(in_filters, out_filters, time_embedding_dim, dim=2, dropout=0.1, attn=False, name=None,**kwargs):
    """
    time_embedding_dim: [(batch_size, time_embedding_dim)]
    """
    if dim == 2:
        myconv = layers.Conv2D
    elif dim == 3:
        myconv = layers.Conv3D
    
    # 输入：图像特征和时间嵌入
    x_input = layers.Input([None for _ in range(dim)] + [in_filters,], name='feature_input')
    t_input = layers.Input([time_embedding_dim,], name='time_input')
    
    h = x_input
    # y = mypool(pool_size=strides)(x)

    # block 1
    h = GroupNormalization(groups=32, axis=-1)(h)  # 使用GroupNormalization
    h = layers.Activation('swish')(h)  # 使用Swish激活函数
    h = myconv(out_filters, kernel_size=3, strides=1, padding='same',
               kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(h)  # 卷积层


    # time_embedding: SiLU + Linear
    t_emb = layers.Activation('swish')(t_input)
    t_emb = layers.Dense(out_filters, kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(t_emb)   # [bs, out_filters]

    # 将时间嵌入添加到特征图 (广播到空间维度)
    if dim == 2:
        t_emb = layers.Reshape([1, 1, out_filters])(t_emb)  # 2D: [bs, 1, 1, out_filters]
    elif dim == 3:
        t_emb = layers.Reshape([1, 1, 1, out_filters])(t_emb) # 3D: [bs, 1, 1, 1, out_filters]

    h = layers.Add()([h, t_emb])  # h += time_emb

    # block 2
    h = GroupNormalization(groups=32, axis=-1)(h)  # 使用GroupNormalization
    h = layers.Activation('swish')(h)  # 使用Swish激活
    h = layers.Dropout(dropout)(h)  # Dropout
    h = myconv(out_filters, kernel_size=3, strides=1, padding='same',
               kernel_initializer=VarianceScaling(scale=1e-5), bias_initializer=Zeros())(h)  # 卷积层

    # 跳跃连接
    if in_filters != out_filters:
        shortcut = myconv(out_filters, kernel_size=1, strides=1, padding='valid', 
                   kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(x_input)
    else:
        shortcut = tf.identity(x_input)  # 如果输入输出通道相同，直接用输入
    # if attn:
    #     x = attention_module([in_filters, out_filters], out_filters, dim=dim, name='attn')(x)

    h = layers.Add()([h, shortcut])  # 输出 = h + shortcut

    result = Model(inputs=[x_input, t_input], outputs=h, name=name)

    return result


def DownSample_Module_diff(in_filters, dim, tdim, name=None, **kwargs):
    """
    Downsample module with optional convolution.
    """
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
    """
    Upsample module with optional convolution.
    """
    if dim == 2:
        myconv = layers.Conv2D
        myup = layers.UpSampling2D
    elif dim == 3:
        myconv = layers.Conv3D
        myup = layers.UpSampling3D
    else:
        raise ValueError('Dimension must be 2 or 3')

    x_input = layers.Input([None for _ in range(dim)] + [in_filters,], name='feature_input')
    t_input = layers.Input([tdim,], name='time_input')
    x = x_input

    x = myup(size=2, interpolation='nearest')(x)  # 上采样
    x = myconv(in_filters, kernel_size=3, strides=1, padding='same',    # 卷积
               kernel_initializer=GlorotUniform(), bias_initializer=Zeros())(x)

    result = Model(inputs=[x_input, t_input], outputs=x, name=name)
    return result


def build_diffusion_unet(
    im_size, nclass, strides_list=None,   # nclass=output_channels=1
    strides_list_mode='symmetric',  
    input_channels=1, features_root=32,   # 2, 64
    conv_size=3, deconv_size=2,           # 3, 3 without use_upsampling, deconv_size is better with 2
    layer_number=5, max_filters=320,      # 4, 320
    dilation=False, attention=False,      # False, False   这里可以设置每一层是否使用注意力机制 list
    deep_supervision=True,        # False
    use_upsampling=False,         # True upsample+conv > convT in image transform. Then deconv_size can be 3
    use_residual_encoder=False,   # False
    freeze_input_shape=False,
    classifier_head=False,
    classifier_nclass=1,
    num_res_blocks = 2,  # Number of residual blocks in each layer
    # Time embedding options (for Diffusion models)
    use_temb_encoder=False,    # 是否使用时间嵌入
    max_timesteps=1000,          # 最大时间步数 T
    # time_embedding_dim=None,     # 时间嵌入维度，默认为 features_root * 4
    dropout=0.1,                 # dropout概率
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
    
    # inputshape = (bs, d, h, w, ch) bs: batchsize; d: depth; h: height; w: width; ch: channel
    f = features_root
    # k = conv_size
    # de_k = deconv_size
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

    # head convolution
    head = Sequential([myconv(f, kernel_size=3, strides=1, padding='same',
                  kernel_initializer=GlorotUniform(), bias_initializer=Zeros())], name='head')
    # Encoder
    encoder_stack = []
    for i, mult in enumerate(ch_mult):
        out_ch = min(f * mult, max_filters)   # 不超过 max_filters
        for j in range(num_res_blocks):
            encoder_stack.append(time_resnet_module(in_filters=now_ch, out_filters=out_ch, time_embedding_dim=tdim,
                                                    dim=dim, dropout=dropout, attn=(i in attention),name=f'encoder_res_{i}_{j}'))
            now_ch = out_ch
            chs.append(now_ch)  # 保存每层的输出通道数
        if i != len(ch_mult) - 1:
            # 如果不是最后一层，则添加下采样层
            encoder_stack.append(DownSample_Module_diff(in_filters=now_ch, dim=dim, tdim=tdim, name=f'encoder_downsample_{i}'))
            chs.append(now_ch)
    
    # Middle
    middle_stack = []
    middle_stack.append(time_resnet_module(in_filters=now_ch, out_filters=now_ch, time_embedding_dim=tdim,
                                           dropout=dropout, dim=dim, attn=False))     # 这里可以选择设置True
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

    # Tail: 最后输出层
    tail = Sequential([
        GroupNormalization(groups=32, axis=-1),
        layers.Activation('swish'),
        myconv(nclass, kernel_size=3, strides=1, padding='same',          # nclass=output_channels=1
               kernel_initializer=VarianceScaling(scale=1e-5, mode='fan_avg', distribution='uniform'),  # 等价于xavier_uniform_，scale=1e-5, 
               bias_initializer=Zeros())
    ], name='tail')


   # --- Build connections ---
    # 双输入：图像 + 时间步
    x_input = layers.Input(shape=input_shape, name='x_input')
    t_input = layers.Input(shape=(), dtype=tf.int32, name='t_input')
    input = [x_input, t_input]
    # 生成时间嵌入
    # temb = TimeEmbedding(t_input, T=max_timesteps, d_model=f, dim=tdim)
    temb = TimeEmbedding(x=t_input, embed_dim=tdim)  # [batch_size, tdim]

    # Encoder: Downsampling
    x = x_input
    h = head(x)  # 初始卷积
    hs = [h]  # 保存每层的输出

    for layer in encoder_stack: # [res, res, down; res, res, down; res, res, down; res, res]
        h = layer([h, temb])
        hs.append(h)
    
    # Middle
    for layer in middle_stack:  # [res, res]
        h = layer([h, temb])

    # Decoder:  Upsampling
    for layer in decoder_stack: # [res,res,res; res,res,res,up; res,res,res,up; res,res,res,up]
        # if isinstance(layer, time_resnet_module): # time_resnet_module 需要是class
        if hasattr(layer, "name") and layer.name.startswith("decoder_res"):
            # 如果是时间嵌入模块，直接传入temb
            h = tf.concat([h, hs.pop()], axis=-1)  # 将temb与h拼接，channel维度
        h = layer([h, temb])
    
    # 最后输出层
    output = tail(h)
    
    assert len(hs) == 0, 'Channel list should be empty after decoder stack'

    unet = Model(input, output)
    return unet



############# Usage Examples #############
if __name__ == "__main__":

  diffusion_unet = build_diffusion_unet(
      im_size=[64, 64],
      nclass=1,
      input_channels=2,
      features_root=64,
      layer_number=4,
      # deep_supervision=False,
      num_res_blocks=2,  # 每层的残差块数量
      # use_temb_encoder=True,     # 开启时间嵌入
      max_timesteps=1000,        # T
      dropout=0.1,                 # dropout概率
      # use_residual_encoder=False   # 使用时间残差编码器
  )
        
  # 使用时在外部拼接
  x = tf.random.normal([2, 64, 64, 1])  # 主图像
  c = tf.random.normal([2, 64, 64, 1])  # context图像
  t = tf.constant([100, 200])               # 时间步，批次大小要匹配

  # 在调用模型前手动拼接
  input_concat = tf.concat([x, c], axis=-1)
  print('input_concat.shape:', input_concat.shape)  # [2, 64, 64, 2]
  output2 = diffusion_unet([input_concat, t]) 
  print('output2.shape:', output2.shape)  # [2, 64, 64, 1]