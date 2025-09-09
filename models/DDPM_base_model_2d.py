import os, h5py, glob, re
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
import time
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

class DDPMBaseModel2D(tf.keras.models.Model):
    def call(self, x):
        return self.unet(x)

    def save(self, step, max_to_keep=1):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.h5')))
        if len(ckpt_files) >= max_to_keep:
            os.remove(ckpt_files[0])
        self.unet.save(os.path.join(checkpoint_dir, 'model_epoch%06d.h5' % step))

    def load(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            print('No model is found, please train first')
            return False, 0
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch*.h5')))
        if ckpt_files:
            ckpt_file = ckpt_files[-1]
            ckpt_name = os.path.basename(ckpt_file)
            self.counter = int(re.findall(r'epoch\d+', ckpt_name)[0][5:])
            self.unet = tf.keras.models.load_model(ckpt_file, compile=False)
            print('Loaded model checkpoint:', os.path.basename(os.path.dirname(ckpt_file)), ckpt_name)
            return True, self.counter
        else:
            print('Failed to find a checkpoint')
            return False, 0

    @staticmethod
    def crop_pad2D(x, target_size, shift=[0, 0]):
        'crop or zero-pad the 2D volume to the target size'
        x = np.asarray(x)
        small = 0
        y = np.ones(target_size, dtype=np.float32) * small
        current_size = x.shape
        pad_size = [0, 0]
        # print('current_size:',current_size)
        # print('pad_size:',target_size)
        for dim in range(2):
            if current_size[dim] > target_size[dim]:
                pad_size[dim] = 0
            else:
                pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim])/2.0))
        # pad first
        x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]]], 'constant', constant_values=small)
        # crop on x1
        start_pos = np.ceil((np.asarray(x1.shape) - np.asarray(target_size))/2.0)
        start_pos = start_pos.astype(int)
        y = x1[(shift[0]+start_pos[0]):(shift[0]+start_pos[0]+target_size[0]),
              (shift[1]+start_pos[1]):(shift[1]+start_pos[1]+target_size[1])]
        return y

    def get_training_patch(self, images, full_size, im_size, enlarged_im_size):
        full_size = list(full_size)
        # Pad
        pad = np.array(enlarged_im_size) + 1 - np.array(full_size)
        pad = np.clip(pad, 0, None)
        pad_with = tuple(zip(pad // 2, pad - pad // 2)) + ((0, 0),)
        if pad_with != ((0, 0), (0, 0), (0, 0)):
            images = np.pad(images, pad_with, mode='constant')
        
        full_size = list(np.maximum(full_size, enlarged_im_size))
        
        x_range = max(full_size[0] - enlarged_im_size[0], 1)
        y_range = max(full_size[1] - enlarged_im_size[1], 1)

        x_offset = int(enlarged_im_size[0] / 2)
        y_offset = int(enlarged_im_size[1] / 2)
        
        if self.sampling_config is not None and 'fg_range' in self.sampling_config:
            fg_low, fg_high = self.sampling_config['fg_range']
            labels = np.logical_and(images[..., 0] >= fg_low, images[..., 0] <= fg_high)
        else:
            labels = np.ones(full_size)
        la = labels[x_offset : x_offset + x_range, y_offset : y_offset + y_range]
        
        # Get sampling prob
        # Random sampling ? of the time and always choose fg the rest of the time
        if np.random.random() > self.fg_sampling_ratio:
            # choose random
            o = np.random.choice(x_range * y_range)
        else:
            p = np.zeros((x_range, y_range), dtype=np.float32)
            if np.amax(la) > 0:
                p[la > 0] = 1
            else:
                # if foreground is not present (gives NaN value for p)
                p = np.ones((x_range, y_range), dtype=np.float32)
            p = p.flatten() / np.sum(p)
            o = np.random.choice(x_range * y_range, p=p)    
        x_start, y_start = np.unravel_index(o, (x_range, y_range))
        
        images_extracted = images[x_start : x_start + enlarged_im_size[0], y_start : y_start + enlarged_im_size[1], ...]
        
        # if augmentation:
        #     images_extracted = self.perform_augmentation(images_extracted, enlarged_im_size)

        x_border_width = int((enlarged_im_size[0] - im_size[0]) / 2)
        y_border_width = int((enlarged_im_size[1] - im_size[1]) / 2)

        images_extracted = images_extracted[x_border_width : x_border_width + im_size[0], 
                                            y_border_width : y_border_width + im_size[1], ...]

        return images_extracted
    
    def read_training_inputs(self, file_path, target_size, patch):
        file_path = file_path.numpy().decode('utf-8') if isinstance(file_path, bytes) or isinstance(file_path, np.bytes_) else file_path.numpy().decode()
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        with h5py.File(file_path, 'r') as f_h5:
            input_img = np.asarray(f_h5['input_images'], dtype=np.float32)
            input_images_mask = np.asarray(f_h5['input_images_mask'], dtype=np.uint8) if 'input_images_mask' in f_h5.keys() else np.zeros_like(input_img, dtype=np.uint8)
            output_img = np.asarray(f_h5['output_images'], dtype=np.float32)
            output_images_mask = np.asarray(f_h5['output_images_mask'], dtype=np.uint8) if 'output_images_mask' in f_h5.keys() else np.zeros_like(output_img, dtype=np.uint8)
            # input_img = input_img * 2. - 1.  ############ Scale to [-1, 1]
            # output_img = output_img * 2. - 1. ############ Scale to [-1, 1]
            full_size = input_img.shape[1:]
            results = []
            for i in range(input_img.shape[0]):
              if patch==True: # 随机 patch
                input_img_ch = np.expand_dims(input_img[i], axis=-1)
                output_img_ch = np.expand_dims(output_img[i], axis=-1)
                input_images_mask_ch = np.expand_dims(input_images_mask[i], axis=-1)
                output_images_mask_ch = np.expand_dims(output_images_mask[i], axis=-1)
                images = np.concatenate([input_img_ch, output_img_ch, input_images_mask_ch, output_images_mask_ch], axis=-1)
                images_extracted = self.get_training_patch(images, full_size, im_size=target_size, enlarged_im_size=self.enlarged_im_size)
                input_images_extracted = images_extracted[..., 0:1]
                output_images_extracted = images_extracted[..., 1:2]
                input_images_mask_extracted = (images_extracted[..., 2:3]> 0.5).astype(np.float32)
                output_images_mask_extracted = (images_extracted[..., 3:4]> 0.5).astype(np.float32)
                results.append((input_images_extracted, output_images_extracted, input_images_mask_extracted, output_images_mask_extracted, file_name.encode('utf-8')))
              else: # center crop
                input_img_crop = self.crop_pad2D(x=input_img[i], target_size=target_size)
                output_img_crop = self.crop_pad2D(x=output_img[i], target_size=target_size)
                input_images_mask_crop = self.crop_pad2D(x=input_images_mask[i], target_size=target_size)
                output_images_mask_crop = self.crop_pad2D(x=output_images_mask[i], target_size=target_size)
                input_img_crop = np.expand_dims(input_img_crop, axis=-1)
                output_img_crop = np.expand_dims(output_img_crop, axis=-1)
                input_images_mask_crop = np.expand_dims(input_images_mask_crop, axis=-1)
                output_images_mask_crop = np.expand_dims(output_images_mask_crop, axis=-1)
                results.append((input_img_crop, output_img_crop, input_images_mask_crop, output_images_mask_crop, file_name.encode('utf-8')))
            arrs = list(zip(*results))
            return (
                np.stack(arrs[0], axis=0),
                np.stack(arrs[1], axis=0),
                np.stack(arrs[2], axis=0),
                np.stack(arrs[3], axis=0),
                np.array(arrs[4], dtype=np.bytes_)
            )

    def get_trainning_dataset(self, folder_path, target_size, patch=True, batch_size=8, shuffle=True):
        all_files = []
        # list or tuple
        if isinstance(folder_path, (list, tuple)):
            for single_folder in folder_path:
                if isinstance(single_folder, (list, tuple)):
                    for sub_folder in single_folder:
                        all_files.extend(sorted(glob.glob(os.path.join(str(sub_folder)))))
                else:
                    all_files.extend(sorted(glob.glob(os.path.join(str(single_folder)))))
        else:
            # file folder
            if os.path.isdir(folder_path):
                all_files.extend(sorted(glob.glob(os.path.join(folder_path, "*.hdf5"))))
            # hdf5 file
            elif os.path.isfile(folder_path) and folder_path.endswith('.hdf5'):
                all_files.append(folder_path)

        files_ds = tf.data.Dataset.from_tensor_slices(all_files)
        def map_fn(file_path):
            elems = tf.py_function(
                func=lambda f: self.read_training_inputs(f, target_size, patch),
                inp=[file_path],
                Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.string],
            )
            input_img, output_img, input_mask, output_mask, file_name = elems
            file_name = tf.reshape(file_name, [-1, 1])
            return tf.data.Dataset.from_tensor_slices((input_img, output_img, input_mask, output_mask, file_name))

        ds = files_ds.interleave(
            map_fn,
            cycle_length=8,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # crop background based on mask
    @staticmethod
    def get_bbox_from_mask(mask, threshold=0, margin=5):
        # mask: [d, w, h]
        full_size = mask.shape
        pos = np.where(mask > threshold)
        z_min, y_min, x_min = np.min(pos, axis=1)
        z_max, y_max, x_max = np.max(pos, axis=1) + 1
        z_min = max(z_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_min = max(x_min - margin, 0)
        z_max = min(z_max + margin, mask.shape[0])
        y_max = min(y_max + margin, mask.shape[1])
        x_max = min(x_max + margin, mask.shape[2])
        return (z_min, z_max, y_min, y_max, x_min, x_max),full_size
    
    @staticmethod
    def crop_with_bbox(arr, bbox):
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        return arr[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Restore to original size
    @staticmethod
    def restore_with_bbox(cropped, bbox, shape):
        arr = np.zeros(shape, dtype=cropped.dtype)
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        arr[z_min:z_max, y_min:y_max, x_min:x_max] = cropped
        return arr

    # pad to multiples of layers
    def read_testing_inputs(self, images):
        #images = None
        #targets = None

        # Our 2D image dim standard is [batch(always=1 / maybe missing), h, w, channel(maybe missing)]
        # In testing, we keep the batch dim
        if len(images.shape) == 4:
            full_size = list(images.shape)[1:-1]
        elif len(images.shape) == 3:
            if self.input_channels == images.shape[-1]:
                full_size = list(images.shape)[:-1]
                images = np.expand_dims(images, axis=0)
            else:
                full_size = list(images.shape[1:])
                images = np.expand_dims(images, axis=-1)
        elif len(images.shape) == 2:
            full_size = list(images.shape)
            images = np.expand_dims(np.expand_dims(images, axis=0), axis=-1)
        
        dividable_by = 2 ** self.layer_number
        pad_size = (int(np.ceil(full_size[0] / dividable_by) * dividable_by),
                    int(np.ceil(full_size[1] / dividable_by) * dividable_by))
        
        # Pad if full size is smaller than patch size
        # symetric padding is always better than one-side padding
        pads = [max(0, pad_size[ax] - full_size[ax]) for ax in range(len(full_size))]
        
        if np.any(np.array(pads) > 0):
            if len(images.shape) == 4:
                pad_with = ((0, 0), (pads[0] // 2, pads[0] - pads[0] // 2), 
                            (pads[1] // 2, pads[1] - pads[1] // 2), (0, 0))
            else:
                pad_with = ((0, 0), (pads[0] // 2, pads[0] - pads[0] // 2), 
                            (pads[1] // 2, pads[1] - pads[1] // 2))
            images = np.pad(images, pad_with, mode='constant')
        
        info = {
            'full_size': full_size,
            'pad_size': pad_size
        }

        all_images = images.copy()
        #all_targets = targets.copy()
        for sli in range(all_images.shape[0]):
            images = all_images[sli]
            #targets = all_targets[sli]
            # Normalize images
            if self.norm_config['norm']:
                eps = 1e-7
                if len(images.shape) == 3:
                    if self.norm_config['norm_channels'] == 'rgb_channels':
                        rgb_mean = np.array(self.norm_config.get('norm_mean', [0.485, 0.456, 0.406]))
                        rgb_std = np.array(self.norm_config.get('norm_std', [0.229, 0.224, 0.225]))
                        images = (images - rgb_mean) / rgb_std
                    elif self.norm_config['norm_channels'] == 'all_channels':
                        images = (images - np.mean(images, axis=(0, 1))) / np.clip(np.std(images, axis=(0, 1)), eps, None)
                    else:
                        for channel in self.norm_config['norm_channels']:
                            m = np.mean(images[..., channel])
                            s = np.clip(np.std(images[..., channel]), eps, None)
                            images[..., channel] = (images[..., channel] - m) / s
                else:
                    images = (images - np.mean(images)) / np.clip(np.std(images), eps, None)
          

            all_images[sli] = images
            #all_targets[sli] = targets
        return all_images, info    # [d, h, w, c], [d, h, w]


    ''' >>>>>>>>>>>>>>>>>>> Train <<<<<<<<<<<<<<<<<<<<<<<<'''
    def diffusion_train(self, run_validation=False, validation_paths=None, **kwargs):
        print("Start Diffusion Training...")
        # Compile model for training
        self.compile_it()
        log_dir = os.path.join(self.log_dir, self.model_dir)
        validation_config = None
        if run_validation:
            validation_config = {
                'validation_paths': validation_paths,
                'validation_fn': self.validate_diffusion,
                'log_dir': log_dir,
            }
        saver_config = {
            'period': self.save_period,
            'save_fn': self.save,
            'log_dir': log_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'model_dir': self.model_dir,
        }
        saver_callback = self.DiffusionModelSaver(saver_config=saver_config, 
                                                  validation_config=validation_config, 
                                                  unet=self.unet)
        
        print(f'Steps per epoch: {self.steps_per_epoch}')
        print(f'Training from epoch {self.counter} to {self.epoch}')
        total_iters = self.steps_per_epoch * (self.epoch - self.counter)

        print('Loading training data...')
        train_data = self.get_training_dataset(self.training_paths, target_size=self.im_size,
                                           batch_size=self.batch_size, shuffle=True,
                                           patch=True)
        print('Training data loaded successfully!!')
        num_samples = len(self.training_paths)
        num_slices = 0
        for file in self.training_paths:
            with h5py.File(file, 'r') as f:
                num_slices += f['input_images'].shape[0]
        print('Running on complete dataset with total training samples:', num_samples)
        print(f'Train slices: {num_slices}')

        tr_ls = []
        ## load loss history
        if self.resume == True:        
            readmat = sio.loadmat(log_dir + '/loss.mat')
            load_tr_ls = readmat['loss']
            for i in range(self.counter):
                tr_ls.append(load_tr_ls[0][i])
            print('Finish loading losses!')
        
        ####################### train
        if total_iters > 0:
            callbacks = [saver_callback]

            print(f">>>>>>>>>>> Start training: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} <<<<<<<<<<<<")
            start_time_all = time.time()
            for epoch in range(self.counter, self.epoch):
                train_loss_metric = tf.keras.metrics.Mean()
                iter_times = []
                with tqdm(total=self.steps_per_epoch, ncols=100, desc=f"Epoch {epoch+1}", leave=False, disable=True) as pbar:
                    for step_count, (input_batch, target_batch, _,_, _) in enumerate(train_data):
                        start_time = time.time()
                        if step_count >= self.steps_per_epoch:  # max training step for per epoch
                            break

                        ## diffusion model
                        step_loss = self._diffusion_train_step(input_batch, target_batch, mask_batch=None)
                        train_loss_metric.update_state(step_loss)     

                        if hasattr(self.optimizer.learning_rate, '__call__'):
                            current_step = epoch * self.steps_per_epoch + step_count
                            current_lr = float(self.optimizer.learning_rate(current_step))
                        else:
                            current_lr = float(self.optimizer.learning_rate)
                        batch_logs = {
                            'loss': float(step_loss),
                            'lr': current_lr
                        }
                        for callback in callbacks:
                            callback.on_batch_end(step_count, batch_logs)
                        pbar.set_postfix({
                            'loss': f"{train_loss_metric.result().numpy():.4f}",
                            'lr': f"{current_lr:.2e}"
                        })
                        pbar.update(1)
                        iter_time = time.time() - start_time
                        iter_times.append(iter_time)

                # Epoch loss
                epoch_loss = train_loss_metric.result()              
                epoch_logs = {
                    'loss': float(epoch_loss),
                    'lr': current_lr
                }
                # log & valid & save
                for callback in callbacks:
                    callback.on_epoch_end(epoch, epoch_logs)

                # save loss history
                tr_ls.append(epoch_loss)
                sio.savemat(log_dir + '/loss.mat', {'loss': tr_ls})
                # save latest model
                self.unet.save(os.path.join(self.checkpoint_dir, self.model_dir, 'model_latest.h5'))
                if iter_times:
                   avg_iter_time = sum(iter_times) / len(iter_times)
                print(f"Epoch {epoch + 1}/{self.epoch} - Loss: {epoch_loss:.6f}, LR: {current_lr:.8f}, avg_iter_time: {avg_iter_time:.4f}s")

                train_loss_metric.reset_states()

            for callback in callbacks:
                callback.on_train_end()

            self.save(epoch+1, max_to_keep=10)  # save last checkpoint = model_latest

            # training time
            end_time_all = time.time()
            print(f">>>>>>>>>>> Finish training: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_all))} <<<<<<<<<<<<")
            print(f"Training time: {(end_time_all - start_time_all)/60:.2f} min")

    @tf.function
    def _diffusion_train_step(self, input_batch, target_batch, mask_batch):
        with tf.GradientTape() as tape:
            loss = self.diffusion_trainer.forward(x_0=target_batch, context=input_batch)
            if mask_batch is not None:
                loss = tf.reduce_sum(tf.multiply(loss, mask_batch[...,0])) / tf.reduce_sum(mask_batch[...,0])
            else:
                loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.unet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))
        return loss

    ''' >>>>>>>>>>>>>>>>>>> Valid <<<<<<<<<<<<<<<<<<<<<<<<<'''
    def validate_diffusion(self, validation_paths):
        self.diffusion_infer_global(validation_paths, output_path=None, sampler='ddim', mode='valid')  # valid
        # self.diffusion_test_global('/mnt/newdisk/mri2ct/data_training/train_2_brain', output_path=None, sampler='ddim') # 2 train set if need

    ''' >>>>>>>>>>>>>>>>>>> Model Saver <<<<<<<<<<<<<<<<<<<<<<<<< '''
    class DiffusionModelSaver(tf.keras.callbacks.Callback):
        def __init__(self, saver_config, validation_config=None, custom_log_file=None, unet=None):
            self.counter = 0
            self.period = saver_config['period']
            self.save = saver_config['save_fn']
            self.validation_config = validation_config
            self.logs = {}
            self.custom_log_file = custom_log_file
            self.best_mae = float('inf')
            self.best_ssim = -float('inf')
            self.best_mae_epoch = -1
            self.best_mae_model_path = None
            self.best_ssim_epoch = -1
            self.best_ssim_model_path = None
            self.checkpoint_dir = saver_config.get('checkpoint_dir', None)
            self.model_dir = saver_config.get('model_dir', None)
            self.unet = unet

            log_dir = saver_config.get('log_dir', 'logs')
            self.train_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'avg'))
            
            if validation_config is not None:
                self.validation_paths = validation_config['validation_paths']
                self.validation_fn = validation_config['validation_fn']
                self.test_writer = tf.summary.create_file_writer(os.path.join(validation_config['log_dir'], 'test'))
            else:
                self.test_writer = None
                
            print(f"DiffusionModelSaver initialized - save period: {self.period} epochs")
            print(f"TensorBoard logs will be saved to: {log_dir}/avg")

        def on_batch_end(self, batch, logs):
            if len(self.logs) == 0:
                for key in logs.keys():
                    self.logs[key] = [logs[key]]
            else:
                for key in logs.keys():
                    self.logs[key].append(logs[key])

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            self.counter += 1
            with self.train_writer.as_default():
                if 'loss' in logs:
                    tf.summary.scalar('loss', logs['loss'], step=epoch)
                if 'lr' in logs:
                    tf.summary.scalar('learning_rate', logs['lr'], step=epoch)
                if self.logs:
                    for key in self.logs.keys():
                        if 'loss' in key:
                            non_zero_values = np.array(self.logs[key])[np.nonzero(self.logs[key])]
                            if len(non_zero_values) > 0:
                                batch_avg = non_zero_values.mean()
                                tf.summary.scalar(f'batch_avg_{key}', batch_avg, step=epoch)
                        else:
                            batch_avg = np.nanmean(self.logs[key])
                            tf.summary.scalar(f'batch_avg_{key}', batch_avg, step=epoch)
                self.train_writer.flush()

            if self.custom_log_file is not None:
                import csv
                import datetime
                with open(self.custom_log_file, 'a') as f:
                    writer = csv.writer(f, delimiter=';')
                    loss_val = logs.get('loss', 0.0)
                    lr_val = logs.get('lr', 0.0)
                    writer.writerow([datetime.datetime.now().strftime('%d/%b/%Y %H:%M:%S'),
                                   f'epoch: {epoch + 1}, loss: {loss_val:.6f}, lr: {lr_val:.8f}'])
            
            self.logs = {}

            # period save
            if (epoch + 1) % self.period == 0:
                self.save(epoch + 1, max_to_keep=10)
                print(f"Checkpoint saved at epoch {epoch + 1}")
            # Valid
            if (epoch + 1) % self.period == 0:
              if self.validation_config is not None:
                  try:
                      val_scores = self.validation_fn(self.validation_paths)   
                      mae_val = val_scores.get('mae', None)
                      ssim_val = val_scores.get('ssim', None)
                      if hasattr(self, 'test_writer') and self.test_writer is not None:
                          with self.test_writer.as_default():
                              for metric in val_scores:
                                  tf.summary.scalar(metric, val_scores[metric], step=epoch + 1)
                              self.test_writer.flush()
                          print(f"Validation results: {val_scores}")
                       # save best valid model
                      if mae_val is not None and mae_val < self.best_mae:
                          self.best_mae = mae_val
                          self.best_mae_epoch = epoch + 1
                          best_mae_model_path = os.path.join(self.checkpoint_dir, self.model_dir, 
                                                             'model_epoch%06d_best_mae.h5' % self.best_mae_epoch)
                          # Remove old best_mae models
                          for f in glob.glob(os.path.join(self.checkpoint_dir, self.model_dir, '*best_mae.h5')):
                              if f != best_mae_model_path and os.path.exists(f):
                                  os.remove(f)
                          self.unet.save(best_mae_model_path)
                          self.best_mae_model_path = best_mae_model_path
                          print(f"Best MAE improved to {mae_val:.6f} at epoch {epoch+1}, model saved to {best_mae_model_path}")
                      if ssim_val is not None and ssim_val > self.best_ssim:
                          self.best_ssim = ssim_val
                          self.best_ssim_epoch = epoch + 1
                          self.best_ssim_model_path = os.path.join(self.checkpoint_dir, self.model_dir, 
                                                                   'model_epoch%06d_best_ssim.h5' % self.best_ssim_epoch)
                          # Remove old best_ssim models
                          for f in glob.glob(os.path.join(self.checkpoint_dir, self.model_dir, '*best_ssim.h5')):
                              if f != self.best_ssim_model_path and os.path.exists(f):
                                  os.remove(f)
                          self.unet.save(self.best_ssim_model_path)
                          print(f"Best SSIM improved to {ssim_val:.6f} at epoch {epoch+1}, model saved to {self.best_ssim_model_path}")
                  except Exception as e:
                      print(f"Validation failed: {str(e)}")

        def on_train_end(self, logs=None):
            print("Training completed, closing TensorBoard writers...")
            if hasattr(self, 'train_writer') and self.train_writer is not None:
                self.train_writer.close()
            if hasattr(self, 'test_writer') and self.test_writer is not None:
                self.test_writer.close()
            print("TensorBoard writers closed.")
    

    ''' >>>>>>>>>>>>>>>>>>> Inference <<<<<<<<<<<<<<<<<<<<<<<<<'''
    def diffusion_run_test(self, input_img, sampler='ddim', squeue=None, save_path=None):
      all_images, info = self.read_testing_inputs(input_img)   # [d, h, w, channel],[d, h, w]
      print('pad img:',all_images.shape)
      num_slices = all_images.shape[0] # d
      generated_list = []
      for sli in range(num_slices):
        if (sli+1) % 20 == 0:
            print(f'Processing slice {sli + 1}/{num_slices}...')
        context_sli = all_images[sli]  # [h, w, channel]
        context_sli = np.expand_dims(context_sli, axis=0)  # batch=1，[1, h, w, channel]
        gen = self._diffusion_sample_step(context=context_sli, sampler=sampler, squeue=squeue, save_path=save_path)  # [1, h, w, save_steps]
        # restore
        full_size = info['full_size']
        pad_size = info['pad_size']
        pads = [pad_size[ax] - full_size[ax] for ax in range(len(full_size))]
        gen = gen[:, pads[0] // 2 : pads[0] // 2 + full_size[0],
              pads[1] // 2 : pads[1] // 2 + full_size[1], ...]  # [1, h, w]
        generated_list.append(gen)
      generated = np.concatenate(generated_list, axis=0)  # [num_slices(d), h, w]

      return generated   # [d, h, w]

    def _diffusion_sample_step(self, context, sampler='ddim', squeue=None, save_path=None):
        x_T = tf.random.normal(tf.shape(context))   # [batch_size, H, W, channels]
        if sampler == 'ddim':
            generated = self.diffusion_sampler.ddim_reverse(x_T=x_T, context=context)    
        elif sampler == 'ddpm':
            generated = self.diffusion_sampler.ddpm_reverse(x_T=x_T, context=context)
        # print(f"Generated shape: {generated.shape}")
        if save_path is not None:
          if squeue is not None:
                # Save intermediate results
              for step in range(generated.shape[-1]):   # [batch_size, W, D, save_steps]
                plt.figure(figsize=(6, 6))
                plt.imshow(generated[generated.shape[0]//2, ..., step], cmap='gray')
                plt.axis('off')
                step_num = self.max_timesteps-(step+1)*squeue
                title = f'x_{step_num}'
                filename = f'step_{step_num:04d}.png'
                plt.title(title)
                plt.axis('off')
                plt.savefig(os.path.join(save_path, filename))
                plt.close()
          else:
              # save last result
              plt.figure(figsize=(6, 6))
              plt.imshow(generated[generated.shape[0]//2, ..., -1], cmap='gray')
              plt.title('x_0')
              plt.axis('off')
              plt.savefig(os.path.join(save_path, f'final_x0.png'))
              plt.close()
        
        return generated

    def diffusion_infer_global(self, testing_paths, output_path=None, sampler='ddim', mode='test', **kwargs):
        # np.random.seed(42)
        # tf.random.set_seed(42)
        print("Diffusion global inference start...")
        if mode == 'test':  # Only load model if mode is 'test'
            _loaded, self.counter = self.load()
            if _loaded:
                self.diffusion_sampler = GaussianDiffusionSampler(
                    model=self.unet,
                    beta_1=self.beta_1,
                    beta_T=self.beta_T,
                    T=self.max_timesteps,
                    infer_T=self.infer_T,
                    squeue=self.squeue
                )
            else:
                raise ValueError("model load failed.")
        
        if output_path is not None:
            save_dir = os.path.join(output_path, f"result_2d_epoch_{self.counter}")
            os.makedirs(save_dir, exist_ok=True)

        ## read data
        folder_path = testing_paths
        all_files = []
        # list or tuple
        if isinstance(folder_path, (list, tuple)):
            for single_folder in folder_path:
                if isinstance(single_folder, (list, tuple)):
                    for sub_folder in single_folder:
                        all_files.extend(sorted(glob.glob(os.path.join(str(sub_folder)))))
                else:
                    all_files.extend(sorted(glob.glob(os.path.join(str(single_folder)))))
        else:
            # file folder
            if os.path.isdir(folder_path):
                all_files.extend(sorted(glob.glob(os.path.join(folder_path, "*.hdf5"))))
            # hdf5 file
            elif os.path.isfile(folder_path) and folder_path.endswith('.hdf5'):
                all_files.append(folder_path)
      

        print(f">>>>>>>>>>> Start infer: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} <<<<<<<<<<<<")
        for i, file_path in enumerate(sorted(all_files)):
            with h5py.File(file_path, 'r') as f_h5:
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                if output_path is not None:
                  save_h5_path = os.path.join(save_dir, f"{file_name}.hdf5")
                  if os.path.exists(save_h5_path):
                      print(f'Case {i+1}/{len(all_files)}: {file_name} already exists, skipping...')
                      continue
                print(f'Case {i+1}/{len(all_files)}: {file_name} is preding...')
                input_img = np.asarray(f_h5['input_images'], dtype=np.float32)
                input_images_mask = np.asarray(f_h5['input_images_mask'], dtype=np.uint8)
                output_img = np.asarray(f_h5['output_images'], dtype=np.float32)
                output_images_mask = np.asarray(f_h5['output_images_mask'], dtype=np.uint8)
                print('input_img:',input_img.shape)
                # mask-based crop
                bbox,full_size = self.get_bbox_from_mask(input_images_mask)
                input_img_crop = self.crop_with_bbox(input_img, bbox)
                print('input_img_crop:',input_img_crop.shape)
                # per slice
                t1 = time.time()
                pred_crop = self.diffusion_run_test(input_img=input_img_crop, sampler=sampler)[...,-1]
                t2 = time.time()
                pred_crop = np.array(pred_crop)
                print(f"pred_crop shape: {pred_crop.shape}")
                # restore to original shape
                pred = self.restore_with_bbox(pred_crop,bbox,full_size)
                print(f"pred shape: {pred.shape}")

                # save results
                if output_path is not None:
                    with h5py.File(save_h5_path, 'w') as f:
                        f.create_dataset('output', data=pred, dtype='float32')
                        f.create_dataset('input', data=input_img, dtype='float32')
                        f.create_dataset('target', data=output_img, dtype='float32')
                        f.create_dataset('input_mask', data=input_images_mask, dtype='float32')
                        f.create_dataset('output_mask', data=output_images_mask, dtype='float32')
                    print(f"Saved results to {save_h5_path}")
                # metrics
                mae = np.mean(np.abs(output_img - pred))
                try:
                    ssim_val = ssim(pred, output_img, data_range=1.0, channel_axis=None)
                except Exception:
                    ssim_val = np.nan
                print(f"MAE: {mae:.4f}, SSIM: {ssim_val:.4f}, sampling time: {t2 - t1:.4f} s")
        
        print(f">>>>>>>>>>> Finish infer: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} <<<<<<<<<<<<")

