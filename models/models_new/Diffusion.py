import numpy as np
import tensorflow as tf

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

    @tf.function
    def forward (self, x_0, context=None):
        t = tf.random.uniform((x_0.shape[0],), minval=0, maxval=self.T, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x_0))
        x_t = (
            _extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            _extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        if context is not None:
            x_concat = tf.concat([x_t, context], axis=-1)  # [bs, H, W, D, 2]
            model_output = self.model([x_concat,t])
            if isinstance(model_output, list):
                model_output = model_output[0]
        else:
            model_output = self.model([x_t, t])
            if isinstance(model_output, list):
                model_output = model_output[0]
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        loss = mse(noise, model_output)
        return loss
        

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
            eps = self.model([x_concat,t],training=False)
            if isinstance(eps, list):
                eps = eps[0]
        else:
            eps = self.model([x_t, t])
            if isinstance(eps, list):
                eps = eps[0]

        x_recon = self.predict_start_from_noise(x_t, t=t, noise=eps)
        # If preprocessing scales data to [-1, 1], the reconstructed data should be clipped to [-1, 1] here 
        x_recon = tf.clip_by_value(x_recon, 0, 1)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)
        return model_mean, posterior_log_variance

    @tf.function
    def ddpm_reverse (self, x_T, context=None, mask=None):
        """
        Performs the reverse diffusion process (DDPM sampling) to generate data from noise.
        Args:
          x_T (tf.Tensor): The initial noisy input tensor at time T, typically sampled from a standard normal distribution.
          context (tf.Tensor, optional): Optional conditioning information for the diffusion model. Default is None.
          mask (tf.Tensor, optional): Optional mask tensor to apply to the output, allowing selective generation. Default is None.
        Returns:
          tf.Tensor: The generated sample after the reverse diffusion process.
        """
        # np.random.seed(42)
        # tf.random.set_seed(42)
        x_t = x_T
        infer_num = 0
        x_squeue = x_T
        for time_step in tf.range(self.infer_T - 1, -1, -1):
            t = tf.ones((x_T.shape[0],), dtype=tf.int32) * time_step
            model_mean, model_log_variance = self.p_mean_variance(x_t=x_t, t=t, context=context)
            if time_step > 0:
                noise = tf.random.normal(tf.shape(x_t))
            else:
                noise = tf.zeros_like(x_t)
            nonzero_mask = tf.cast(1 - tf.cast(t == 0, tf.float32), tf.float32)
            nonzero_mask = tf.reshape(nonzero_mask, [x_T.shape[0]] + [1] * (len(x_T.shape) - 1))
            x_t = model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise

            tf.debugging.assert_equal(tf.reduce_sum(tf.cast(tf.math.is_nan(x_t), tf.int32)), 0, message="nan in tensor.")
            infer_num += 1
            if self.squeue is not None:
              if time_step % int(self.squeue) == 0:
                x_squeue = tf.concat([x_squeue, tf.clip_by_value(x_t, 0, 1)], axis=-1)  # clip to [0,1] or [-1,1]

        if self.squeue is not None:
            x0 = x_squeue
        else:
            x0 = tf.clip_by_value(x_t, 0, 1)  # clip to [0,1] or [-1,1]
        if mask is not None:
            x0 = x0 * mask
        return x0

    @tf.function
    def ddim_reverse(self, x_T, context=None, mask=None, eta=1.):
        """
        Performs DDIM reverse sampling to accelerate the generation process.
        Args:
          x_T (tf.Tensor): Initial noise tensor, typically sampled from a standard normal distribution.
          context (tf.Tensor, optional): Optional conditioning tensor to guide the generation process. Default is None.
          mask (tf.Tensor, optional): Optional mask tensor to apply to the final output. Default is None.
          eta (float, optional): Controls the stochasticity of the sampling process. 
            - eta = 0: deterministic sampling.
            - eta = 1: fully stochastic sampling.
        Returns:
          tf.Tensor: The predicted clean sample (x0_pred) after DDIM reverse sampling, optionally masked.
        """
        # np.random.seed(42)
        # tf.random.set_seed(42)
        x_t = x_T
        ddim_steps = self.infer_T
        ddim_timesteps = tf.convert_to_tensor(np.linspace(self.T-1, 0, ddim_steps).astype(np.int32))
        
        for i in tf.range(len(ddim_timesteps) - 1):
            t = tf.ones((x_t.shape[0],), dtype=tf.int32) * ddim_timesteps[i]
            t_prev = tf.ones((x_t.shape[0],), dtype=tf.int32) * ddim_timesteps[i + 1]
            if context is not None:
                x_concat = tf.concat([x_t, context], axis=-1)
                eps = self.model([x_concat, t], training=False)
                if isinstance(eps, list):
                    eps = eps[0]
            else:
                eps = self.model([x_t, t],training=False)
                if isinstance(eps, list):
                    eps = eps[0]

            alpha_t = _extract(self.alphas_cumprod, t, x_t.shape)
            alpha_prev = _extract(self.alphas_cumprod, t_prev, x_t.shape)
            sqrt_alpha_t = tf.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = tf.sqrt(1. - alpha_t)
            sigma = eta * tf.sqrt((1 - alpha_prev) / (1 - alpha_t)) * tf.sqrt(1 - alpha_t / alpha_prev)
            c1 = tf.sqrt(alpha_prev)
            c2 = tf.sqrt(1. - alpha_prev - sigma**2)

            x0_pred = (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
            x0_pred = tf.clip_by_value(x0_pred, 0, 1)    ##

            noise = tf.random.normal(tf.shape(x_t)) if eta > 0 else 0.0
            x_t = c1 * x0_pred + c2 * eps + sigma * noise

        t = tf.ones((x_t.shape[0],), dtype=tf.int32) * ddim_timesteps[-1]
        if context is not None:
            x_concat = tf.concat([x_t, context], axis=-1)
            eps = self.model([x_concat, t], training=False)
            if isinstance(eps, list):
                eps = eps[0]
        else:
            eps = self.model([x_t, t], training=False)
            if isinstance(eps, list):
                eps = eps[0]
        alpha_t = _extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_alpha_t = tf.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = tf.sqrt(1. - alpha_t)
        x0_pred = (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
        x0_pred = tf.clip_by_value(x0_pred, 0, 1)

        if mask is not None:
            x0_pred = x0_pred * mask

        return x0_pred
    