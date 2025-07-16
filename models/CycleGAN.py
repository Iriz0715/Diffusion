import os, sys, glob, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras import layers

from base_model import BaseModel
from networks import *

DEFAULT_MODEL_CONFIG = {
    'epoch': 1000,
    'features_root': 64,
    'conv_size': 3,
    'layers': 4,
    'batch_size': 4,
    'attention': False,
    # model dir or file
    'finetune': './cbct/checkpoint_cv/previous-unet-baseline/', 
}

class CycleGAN(BaseModel):
    def __init__(self, checkpoint_dir, log_dir, training_paths, im_size, num_threads, sampling_config, 
                 left_right_swap_config=None, model_config=None):
        
        super(CycleGAN, self).__init__()
        
        self.model_type = 'CycleGAN'
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        
        self.sampling_config = sampling_config
        
        self.left_right_swap_config = left_right_swap_config
        self.flip_augmentation = True
        
        if model_config is None:
            model_config = DEFAULT_MODEL_CONFIG
            
        self.features_root = int(model_config['features_root'])
        self.im_size = im_size
        self.enlarged_im_size = (int(im_size[0] * 1.1875), int(im_size[1] * 1.1875))
        self.batch_size = int(model_config['batch_size'])
        self.conv_size = int(model_config['conv_size'])
        self.layer_number = int(model_config['layers'])
        self.attention = model_config['attention']
        self.dilation = False
        
        self.num_threads = num_threads
        
        self.lambda_cycle = 10.
        self.steps_per_epoch = int(100 / self.batch_size) 
        self.epoch = model_config['epoch']
        self.finetune = model_config['finetune']
        
        _loaded, self.counter = self.load()
        if not _loaded:
            if self.finetune is not None:
                if os.path.isdir(self.finetune):
                    current_ckpt = os.path.join(self.checkpoint_dir, self.model_dir)
                    shutil.copytree(self.finetune, current_ckpt)
                    self.load()
                elif os.path.isfile(self.finetune):
                    self.unet = tf.keras.models.load_model(self.finetune, compile=False)
                    print('Loaded pretrained model for finetuning', self.finetune)
                else:
                    raise ValueError('Pretrained file or dir is not applicable')
            else:
                self.unet = self.build_unet()
            self.generator_yx = self.build_unet()
            self.discriminator_x = build_cnn(self.im_size, layer_number=3, features_root=32)
            self.discriminator_y = build_cnn(self.im_size, layer_number=3, features_root=32)
    
    def compile(
        self,
        unet_optimizer,
        yx_g_optimizer,
        xy_g_optimizer,
        x_d_optimizer,
        y_d_optimizer,
        g_loss_fn,
        d_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGAN, self).compile()
        self.unet_optimizer = unet_optimizer
        self.yx_g_optimizer = yx_g_optimizer
        self.xy_g_optimizer = xy_g_optimizer
        self.x_d_optimizer = x_d_optimizer
        self.y_d_optimizer = y_d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
    
    def compile_it(self):
        opt0 = self.get_optimizer(1e-4, 0, adam=True)
        opt1 = self.get_optimizer(1e-4, 0, adam=True)
        opt2 = self.get_optimizer(1e-4, 0, adam=True)
        opt3 = self.get_optimizer(1e-4, 0, adam=True)
        opt4 = self.get_optimizer(1e-4, 0, adam=True)
        self.compile(unet_optimizer=opt0, 
                     xy_g_optimizer=opt1, yx_g_optimizer=opt2, x_d_optimizer=opt3, y_d_optimizer=opt4,
                     g_loss_fn=generator_loss, d_loss_fn=discriminator_loss,
                     cycle_loss_fn=l1_loss, identity_loss_fn=l1_loss)
    
    @tf.function
    def train_step(self, batch_data):
        real_x, real_y, mask, paired = batch_data
        
        unet_loss = 0.
        if tf.reduce_mean(paired) > 0.5:
            with tf.GradientTape() as tape:
                output = self.unet(real_x, training=True)
                unet_loss = self.identity_loss_fn(output, real_y, mask, paired)
            
            # Get the gradients
            gradient = tape.gradient(unet_loss, self.unet.trainable_variables)
            # Update the weights
            self.unet_optimizer.apply_gradients(zip(gradient, self.unet.trainable_variables))

        with tf.GradientTape(persistent=True) as tape:
            # y to x back to y
            fake_x = self.generator_yx(real_y, training=True)
            cycled_y = self.unet(fake_x, training=True)

            # x to y back to x
            fake_y = self.unet(real_x, training=True)
            cycled_x = self.generator_yx(fake_y, training=True)

            # generating itself
#             same_x = self.generator_yx(real_x, training=True)
#             same_y = self.unet(real_y, training=True)

            # discriminator used to check, inputing real images
            d_real_x = self.discriminator_x(real_x, training=True)
            d_real_y = self.discriminator_y(real_y, training=True)

            # discriminator used to check, inputing fake images
            d_fake_x = self.discriminator_x(fake_x, training=True)
            d_fake_y = self.discriminator_y(fake_y, training=True)

            # evaluates generator loss + # there exists target images for generated images
            x_target_loss = self.identity_loss_fn(fake_x, real_x, mask, paired)
            y_target_loss = self.identity_loss_fn(fake_y, real_y, mask, paired)

            x_g_loss = self.g_loss_fn(d_fake_x) + self.lambda_cycle * x_target_loss
            y_g_loss = self.g_loss_fn(d_fake_y) + self.lambda_cycle * y_target_loss

            # evaluates total cycle consistency loss
            total_cycle_loss = self.lambda_cycle * self.cycle_loss_fn(real_x, cycled_x) 
            total_cycle_loss += self.lambda_cycle * self.cycle_loss_fn(real_y, cycled_y)

            # evaluates total generator loss
            total_x_g_loss = x_g_loss + total_cycle_loss #+ 0.5 * self.lambda_cycle * self.identity_loss_fn(real_x, same_x)
            total_y_g_loss = y_g_loss + total_cycle_loss #+ 0.5 * self.lambda_cycle * self.identity_loss_fn(real_y, same_y)

            # evaluates discriminator loss # Add the gradient penalty to the original discriminator loss
            x_d_loss = self.d_loss_fn(d_real_x, d_fake_x)
            y_d_loss = self.d_loss_fn(d_real_y, d_fake_y)

        # Calculate the gradients for generator and discriminator
        x_g_gradients = tape.gradient(total_x_g_loss, self.generator_yx.trainable_variables)
        y_g_gradients = tape.gradient(total_y_g_loss, self.unet.trainable_variables)

        x_d_gradients = tape.gradient(x_d_loss, self.discriminator_x.trainable_variables)
        y_d_gradients = tape.gradient(y_d_loss, self.discriminator_y.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.yx_g_optimizer.apply_gradients(zip(x_g_gradients, self.generator_yx.trainable_variables))

        self.xy_g_optimizer.apply_gradients(zip(y_g_gradients, self.unet.trainable_variables))

        self.x_d_optimizer.apply_gradients(zip(x_d_gradients, self.discriminator_x.trainable_variables))

        self.y_d_optimizer.apply_gradients(zip(y_d_gradients, self.discriminator_y.trainable_variables))

        return {
            'loss': unet_loss,
            "y2x_g_loss": total_x_g_loss,
            "x2y_g_loss": total_y_g_loss,
            "x_d_loss": x_d_loss,
            "y_d_loss": y_d_loss
        }

    def build_unet(self):
        input_shape = [None for _ in self.im_size] + [1,]  # FCN actually does not need specific im_size
        
        # inputshape = (bs, h, w, ch) bs: batchsize; h: height; w: width; ch: channel
        f = self.features_root
        c = self.conv_size
        L = self.layer_number
        myconv = layers.Conv2D
        
        # calculate the conv/deconv stride by dims. s: first conv/deconv stride
        s_maxlayer = np.log2(np.array(self.im_size) / 4).astype(int)
        
        down_stack = [downsample(f, c, 1, self.dilation, model_name='encoding_0')]
        down_stack += [downsample(min(f * 2 ** layer, 320), c, 1 + (s_maxlayer >= layer).astype(int),
                                  self.dilation, model_name=f'encoding_{layer}') for layer in range(1, L)]
        down_stack += [downsample(min(f * 2 ** L, 320), c, 1 + (s_maxlayer >= L).astype(int), model_name='bottom')]
        
        up_deconv_stack = [upsample(min(f * 2 ** (L - 1), 320), c, 1 + (s_maxlayer >= L).astype(int), 
                                    model_name=f'decoding_{L - 1}')]
        up_deconv_stack += [upsample(min(f * 2 ** layer, 320), c, 1 + (s_maxlayer >= layer).astype(int), 
                                     model_name=f'decoding_{layer}') for layer in range(L - 2, -1, -1)]
        
        # NOTE - downsample: double conv block. Here the stride == 1.
        up_conv_stack = [downsample(min(f * 2 ** (L - 1), 320), c, 1, model_name=f'decoding_{L - 1}_1')]
        up_conv_stack += [downsample(min(f * 2 ** layer, 320), c, 1, model_name=f'decoding_{layer}_1') 
                          for layer in range(L - 2, -1, -1)]
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Encoder: Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        # get output from encoder
        skips = reversed(skips[:-1])

        # Decoder: Upsampling and establishing the skip connections
        for up_deconv, up_conv, skip, layer in zip(up_deconv_stack, up_conv_stack, skips, range(L - 1, -1, -1)):
            x = up_deconv(x)
            in_channel = skip.get_shape().as_list()[-1]
            if self.attention:
                # in_channel, reduction: for intermediate filters inside attention net
                reduction = 2  # it could be 4 ...
                skip = attention_block(skip, x, in_channel // reduction, myconv,
                                       name=up_deconv.name + '_attn')([skip, x])
            x = layers.Concatenate(axis=-1)([x, skip])
            x = up_conv(x)

        x = conv_block(1, 1, 1, myconv, plain=True, use_bias=True, name='final')(x)  # conv_1by1
        outputs = layers.Activation('sigmoid')(x)
        unet = Model(inputs, outputs)
        return unet
    
    @property
    def model_dir(self):
        return 'finetune-on-previous-baseline-data-checked'