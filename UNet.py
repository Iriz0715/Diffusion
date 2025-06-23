import math
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import Constant


class GroupNormalization(layers.Layer):
    def __init__(self, groups=16, axis=-1, epsilon=1e-5):
        super(GroupNormalization, self).__init__()
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def call(self, x):
        # x shape: [batch_size, ..., channels]
        batch_size, height, width, depth, channels = tf.shape(x)
        G = self.groups
        C = channels
        
        # Reshape to [batch_size, G, -1, channels // G]
        x = tf.reshape(x, (batch_size, height, width, depth, G, C // G))
        mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=[1, 2, 3], keepdims=True)

        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, (batch_size, height, width, depth, channels))
        
        return x
    

# Swish activation function
class Swish(layers.Layer):
    def call(self, x):
        return x * tf.sigmoid(x)

# Time Embedding class
class TimeEmbedding(layers.Layer):
    def __init__(self, T, d_model, dim):
        super().__init__()
        assert d_model % 2 == 0
        emb = tf.range(0, d_model, delta=2, dtype=tf.float32) / d_model * math.log(10000)
        emb = tf.exp(-emb)
        pos = tf.range(T, dtype=tf.float32)
        emb = tf.expand_dims(pos, axis=-1) * emb  # [T, d_model//2]
        emb = tf.stack([tf.sin(emb), tf.cos(emb)], axis=-1)  # [T, d_model//2, 2]
        emb = tf.reshape(emb, (T, d_model))  # [T, d_model]

        self.timembedding = models.Sequential([
            layers.Embedding(input_dim=T, output_dim=d_model, embeddings_initializer=Constant(emb)),
            layers.Dense(dim),
            Swish(),
            layers.Dense(dim),
        ])

    def call(self, t):
        return self.timembedding(t)

# DownSample class
class DownSample(layers.Layer):
    def __init__(self, in_ch):
        super().__init__()
        self.main = layers.Conv3D(in_ch, kernel_size=3, strides=2, padding='same')

    def call(self, x, temb):   # 这个 temb 没用到，应该可以删去
        return self.main(x)

# UpSample class
class UpSample(layers.Layer):
    def __init__(self, in_ch):
        super().__init__()
        self.main = layers.Conv3DTranspose(in_ch, kernel_size=3, strides=2, padding='same')

    def call(self, x, temb):   # 这个 temb 没用到，应该可以删去
        # 使用 Conv3DTranspose 进行上采样
        return self.main(x)

# Attention Block class
class AttnBlock(layers.Layer):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = GroupNormalization(groups=16, axis=-1)
        self.proj_q = layers.Conv3D(in_ch, kernel_size=1)
        self.proj_k = layers.Conv3D(in_ch, kernel_size=1)
        self.proj_v = layers.Conv3D(in_ch, kernel_size=1)
        self.proj = layers.Conv3D(in_ch, kernel_size=1)

    def call(self, x):
        B, H, W, D, C = tf.shape(x)
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = tf.transpose(q, perm=[0, 2, 3, 4, 1])  # [B, H, W, D, C] -> [B, H, W, D, C]
        q = tf.reshape(q, (B, H * W * D, C))  # [B, H*W*D, C]
        k = tf.reshape(k, (B, C, H * W * D))  # [B, C, H*W*D]
        
        w = tf.matmul(q, k) * (C ** -0.5)
        w = tf.nn.softmax(w, axis=-1)

        v = tf.transpose(v, perm=[0, 2, 3, 4, 1])  # [B, H, W, D, C] -> [B, H, W, D, C]
        v = tf.reshape(v, (B, H * W * D, C))  # [B, H*W*D, C]
        
        h = tf.matmul(w, v)  # [B, H*W*D, C]
        h = tf.reshape(h, (B, H, W, D, C))  # [B, H, W, D, C]
        h = tf.transpose(h, perm=[0, 4, 1, 2, 3])  # [B, C, H, W, D]
        h = self.proj(h)

        return x + h

# ResBlock class
class ResBlock(layers.Layer):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = models.Sequential([
            GroupNormalization(groups=16, axis=-1),
            Swish(),
            layers.Conv3D(out_ch, kernel_size=3, padding='same'),
        ])
        self.temb_proj = models.Sequential([
            Swish(),
            layers.Dense(out_ch),
        ])
        self.block2 = models.Sequential([
            GroupNormalization(groups=16, axis=-1),
            Swish(),
            layers.Dropout(dropout),
            layers.Conv3D(out_ch, kernel_size=3, padding='same'),
        ])
        self.shortcut = layers.Conv3D(out_ch, kernel_size=1, padding='same') if in_ch != out_ch else layers.Layer()
        self.attn = AttnBlock(out_ch) if attn else layers.Layer()

    def call(self, x, temb):
        h = self.block1(x)
        temb = self.temb_proj(temb)  # [batch_size, out_ch]
        
        # 确保 temb 的形状为 [batch_size, 1, 1, 1, out_ch]
        temb = tf.expand_dims(temb, axis=1)  # [batch_size, 1, out_ch]
        temb = tf.expand_dims(temb, axis=1)  # [batch_size, 1, 1, out_ch]
        temb = tf.expand_dims(temb, axis=1)  # [batch_size, 1, 1, 1, out_ch]

        h += temb  # 形状匹配
        h = self.block2(h)
        h += self.shortcut(x)
        return h

# UNet class
class UNet(models.Model):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, in_ch=2, out_c=1):    ## in_channel = 2 (x+c)
        super().__init__()
        assert all(i < len(ch_mult) for i in attn), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = layers.Conv3D(ch, kernel_size=3, padding='same', input_shape=(None, None, None, in_ch))
        self.downblocks = []
        chs = [ch]  # record output channel when downsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = [
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ]

        self.upblocks = []
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        self.tail = models.Sequential([
            GroupNormalization(groups=16, axis=-1),
            Swish(),
            layers.Conv3D(out_c, kernel_size=3, padding='same')
        ])

    def call(self, x, t):
        temb = self.time_embedding(t)
        h = self.head(x)

        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        for layer in self.middleblocks:
            h = layer(h, temb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = tf.concat([h, hs.pop()], axis=-1)  # 在通道维度拼接
            h = layer(h, temb)

        h = self.tail(h)

        return h

if __name__ == '__main__':
  batch_size = 1
  in_ch = 2
  out_c = 1
  model = UNet(
      T=1000, ch=32, ch_mult=[1, 2, 2, 2], attn=[1],
      num_res_blocks=1, dropout=0.1, in_ch=2, out_c=1)
  x = tf.random.normal((batch_size, 96, 96,96, in_ch))
  t = tf.random.uniform((1,), minval=0, maxval=100, dtype=tf.int32)
  y = model(x, t)
  print(y.shape)
  print(t.shape)
  model.summary()
