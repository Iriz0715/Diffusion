
import numpy as np
from functools import partial
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import MeanSquaredError

def _extract(a, t, x_shape):
  """
  Extract some coefficients at specified timesteps,
  then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
  """
  bs, = t.shape
  assert x_shape[0] == bs
  out = tf.gather(tf.cast(a, tf.float32), t)
  assert out.shape == [bs]
  return tf.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))


class GaussianDiffusionTrainer(tf.keras.layers.Layer):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.betas = tf.Variable(tf.cast(tf.linspace(beta_1, beta_T, T), tf.float64), trainable=False)
        alphas = 1. - self.betas
        alphas_bar = tf.math.cumprod(alphas, axis=0)

        self.sqrt_alphas_bar = tf.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = tf.sqrt(1. - alphas_bar)

    def sample(self, x_0, t, in_ch=1):
        t = tf.ones((x_0.shape[0],), dtype=tf.int32) * t
        noise = tf.random.normal(tf.shape(x_0))

        x_t = (
            _extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            _extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        return x_t


    ## train时每个batch t 随机，epoch决定train次数
    def forward (self, x_0, context=None):
        t = tf.random.uniform((x_0.shape[0],), minval=0, maxval=self.T, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x_0))

        x_t = (
            _extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            _extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )

        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        if context is not None:
            x_concat = tf.concat([x_t, context], axis=-1)  # [bs, H, W, D, 2]
            model_output = self.model([x_concat,t])
            # 如果模型输出是列表，取第一个元素
            if isinstance(model_output, list):
                model_output = model_output[0]
        else:
            model_output = self.model([x_t, t])
            if isinstance(model_output, list):
                model_output = model_output[0]

        loss = mse(noise, model_output)
        return loss


# class GaussianDiffusionSampler(tf.keras.layers.Layer):
#     def __init__(self, model, beta_1, beta_T, T, infer_T=None, squeue=None):
#         super().__init__()

#         self.model = model
#         self.T = T
#         self.infer_T = T if infer_T is None else infer_T
#         self.squeue = squeue

#         linear_start = beta_1
#         linear_end = beta_T

#         betas = tf.cast(
#             # tf.linspace(linear_start ** 0.5, linear_end ** 0.5, T) ** 2,
#             tf.linspace(linear_start, linear_end, T),
#             tf.float64
#         )
#         betas = betas.numpy()

#         alphas = 1. - betas
#         alphas_cumprod = np.cumprod(alphas, axis=0)
#         alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

#         self.num_timesteps = int(T)
#         self.linear_start = linear_start
#         self.linear_end = linear_end

#         to_tf = partial(tf.Variable, trainable=False, dtype=tf.float32)   # 方便创建不可训练的float32 tensor变量

#         self.betas = to_tf(betas)
#         self.alphas_cumprod = to_tf(alphas_cumprod)
#         self.alphas_cumprod_prev = to_tf(alphas_cumprod_prev)

#         self.sqrt_alphas_cumprod = to_tf(np.sqrt(alphas_cumprod))
#         self.sqrt_one_minus_alphas_cumprod = to_tf(np.sqrt(1. - alphas_cumprod))
#         self.log_one_minus_alphas_cumprod = to_tf(np.log(1. - alphas_cumprod))
#         self.sqrt_recip_alphas_cumprod = to_tf(np.sqrt(1. / alphas_cumprod))
#         self.sqrt_recipm1_alphas_cumprod = to_tf(np.sqrt(1. / alphas_cumprod - 1))

#         self.v_posterior = 0.0  # This is a hyperparameter, can be adjusted
#         posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
#                     1. - alphas_cumprod) + self.v_posterior * betas
#         self.posterior_variance = to_tf(posterior_variance)
#         self.posterior_log_variance_clipped = to_tf(np.log(np.maximum(posterior_variance, 1e-20)))

#         self.posterior_mean_coef1 = to_tf(
#             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
#         self.posterior_mean_coef2 = to_tf(
#             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        

class GaussianDiffusionSampler(tf.keras.layers.Layer):
    def __init__(self, model, beta_1, beta_T, T, infer_T=None, squeue=None):
        super().__init__()

        self.model = model
        self.T = T
        self.infer_T = T if infer_T is None else infer_T
        self.squeue = squeue

        linear_start = beta_1
        linear_end = beta_T

        betas = tf.cast(
            # tf.linspace(linear_start ** 0.5, linear_end ** 0.5, T) ** 2,
            tf.linspace(linear_start, linear_end, T),
            tf.float64
        )
        betas = betas.numpy()

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(T)
        self.linear_start = linear_start
        self.linear_end = linear_end

        self.betas = tf.Variable(betas, trainable=False, dtype=tf.float32)
        self.alphas_cumprod = tf.Variable(alphas_cumprod, trainable=False, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.Variable(alphas_cumprod_prev, trainable=False, dtype=tf.float32)

        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = tf.math.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1. / self.alphas_cumprod - 1)

        self.v_posterior = 0.0  # This is a hyperparameter, can be adjusted
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        self.posterior_variance = tf.Variable(posterior_variance, trainable=False, dtype=tf.float32)
        self.posterior_log_variance_clipped = tf.Variable(np.log(np.maximum(posterior_variance, 1e-20)), trainable=False, dtype=tf.float32)

        self.posterior_mean_coef1 = tf.Variable(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), trainable=False, dtype=tf.float32)
        self.posterior_mean_coef2 = tf.Variable(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod), trainable=False, dtype=tf.float32)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, x_t, t, context=None):
        if context is not None:
            x_concat = tf.concat([x_t, context], axis=-1)  # [bs, H, W, D, 2]
            eps = self.model([x_concat,t])
            if isinstance(eps, list):
                eps = eps[0]
        else:
            eps = self.model([x_t, t])
            if isinstance(eps, list):
                eps = eps[0]

        x_recon = self.predict_start_from_noise(x_t, t=t, noise=eps)
        x_recon = tf.clip_by_value(x_recon, -1, 1)  # 限制到 [-1, 1] 范围
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)
        return model_mean, posterior_log_variance


    ## sample 时每个batch t 相同，逐t反向去噪，T决定采样次数
    def reverse (self, x_T, context=None):
        x_t = x_T
        infer_num = 0

        # x_squeue = tf.zeros_like(x_t) # [batch_size, H, W, D, C]
        x_squeue = x_T # 保存输入的纯噪声
        #print('T', self.T, 'infer_T', self.infer_T)
        for time_step in reversed(range(self.infer_T)):       # T-1 ->0
            t = tf.ones((x_T.shape[0],), dtype=tf.int32) * time_step

            model_mean, model_log_variance = self.p_mean_variance(x_t=x_t, t=t, context=context)
            if time_step > 0:
                noise = tf.random.normal(tf.shape(x_t))
            else:
                noise = 0

            nonzero_mask = tf.cast(1 - tf.cast(t == 0, tf.float32), tf.float32)
            nonzero_mask = tf.reshape(nonzero_mask, [x_T.shape[0]] + [1] * (len(x_T.shape) - 1))

            x_t = model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise

            assert tf.reduce_sum(tf.cast(tf.math.is_nan(x_t), tf.int32)) == 0, "nan in tensor."
            infer_num += 1
            if self.squeue is not None:
              # If in the last 300 steps and step number is divisible by 100
              if time_step <= 300 and time_step % 100 == 0:
                x_squeue = tf.concat([x_squeue, tf.clip_by_value(x_t, -1, 1)], axis=-1)
              # Otherwise use original logic
              elif time_step > 300 and infer_num % int(self.squeue) == 0:
                x_squeue = tf.concat([x_squeue, tf.clip_by_value(x_t, -1, 1)], axis=-1)
        x_0 = x_t
        x0 = tf.clip_by_value(x_0, -1, 1)
        if self.squeue is not None:
            x0 = x_squeue[..., 1:]  # 去掉第一个0数据

        return x0




import h5py
import matplotlib.pyplot as plt

if __name__ =="__main__":
  file_path = '/home/jiayizhang/project/diffusion/DDPM/CBCT2CTTest/synthrad2023_brain_2BA001.hdf5'
  with h5py.File(file_path, 'r') as f_h5:
    input_images = np.asarray(f_h5['input_images'], dtype=np.float32) # cbct
    output_images = np.asarray(f_h5['output_images'], dtype=np.float32) # ct

    image_tensor = tf.convert_to_tensor(input_images, dtype=tf.float32)
    # image_tensor = tf.expand_dims(image_tensor, axis=0)  # 增加批次维度
    # image_tensor = tf.expand_dims(image_tensor, axis=-1)  # 增加通道维度
    print(image_tensor.shape)


    batch_size = 1


    model = None  # 这里不需要实际的模型，因为我们只进行加噪处理
    trainer = GaussianDiffusionTrainer(model, beta_1=0.0001, beta_T=0.02, T=1000)

    # forward 加噪
    noisy_image = trainer.sample(image_tensor, t=50)

    noisy_image = np.clip(noisy_image, -1, 1)  # 限制到 [-1, 1] 范围
    print(noisy_image.shape)  # 输出加噪后图像的形状


    plt.imsave('/home/jiayizhang/project/diffusion/DDPM/zjy/test_forward.png', noisy_image[219//2,:,:], cmap='gray')
    plt.imsave('/home/jiayizhang/project/diffusion/DDPM/zjy/test_orig.png', input_images[219//2,:,:], cmap='gray')