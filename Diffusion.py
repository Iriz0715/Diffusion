
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from functools import partial

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    v=v.to(device)
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))


    def sample(self,x_0,t,in_ch=1):
        t = t*torch.ones(size=(x_0.shape[0], ))
        device=x_0.device
        t= torch.tensor(t).type(torch.int64).to(device)
        # print(t)
        # print(x_0.shape)
        noise = torch.randn_like(x_0)
        # print('noise',noise.shape)

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        return x_t


    def forward(self, x_0,context=None):##把低dose图像作为噪声
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        # print('noise',noise.shape)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        if context !=None:
            loss = F.mse_loss(self.model(x_t, t,context), noise, reduction='none')
        else:
            loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,infer_T=None,squeue=None):
        '''
        推理时直接使用改T有问题,infer_T用于在某些截断
        squeue:用于保存过程中的结果,None只保留最终的结果,为数字时保留完整的,squeue=10
        '''
        super().__init__()

        self.model = model
        self.T = T
        if infer_T ==None:
            self.infer_T =T
        else:
            self.infer_T = infer_T
        self.squeue = squeue

        # self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
        # self.register_buffer('betas', torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, T, dtype=torch.float64) ** 2)

        linear_start = beta_1
        linear_end = beta_T

        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, T, dtype=torch.float64) ** 2
        )
        betas = betas.numpy()

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        self.v_posterior = 0
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))

        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))



    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, x_t, t,context=None):
        # below: only log_variance is used in the KL computations
        # var = torch.cat([self.posterior_var[1:2], self.betas[1:]])



        if context !=None:
            eps = self.model(x_t, t,context)
        else:
            eps = self.model(x_t, t)
        # xt_prev_mean = self.predict_start_from_noise(x_t, t, eps=eps)
        x_recon = self.predict_start_from_noise(x_t, t=t, noise=eps)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)


    

        return model_mean, posterior_log_variance

    def forward(self, x_T,context=None):
        """
        Algorithm 2.
        """
        x_t = x_T
        infer_num =0


        x_squeue = torch.zeros(list(x_t.shape)).cuda()
        print('T',self.T,'infer_T',self.infer_T)
        for time_step in reversed(range(self.infer_T)):
            # print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step

            # x_t = x_t +x_T ###加入隐变量x_T，正常情况记得删除
            model_mean, model_log_variance= self.p_mean_variance(x_t=x_t, t=t, context= context)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            # x_t = mean + torch.sqrt(var) * noise

            nonzero_mask = (1 - (t == 0).float()).reshape(x_T.shape[0], *((1,) * (len(x_T.shape) - 1)))

            x_t =  model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    



            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            infer_num +=1
            if self.squeue != None:
                if infer_num % int(self.squeue)==0:
                    x_squeue = torch.cat([x_squeue,torch.clip(x_t, -1, 1)],dim=1)
                    # x_squeue.append(torch.clip(x_t, -1, 1))

            
            # if infer_num==self.infer_T:
            #     break
        x_0 = x_t
        x0 = torch.clip(x_0, -1, 1)
        if self.squeue != None:
            # val= torch.tensor([item.cpu().detach().numpy() for item in val]).cuda()

            # x_squeue = torch.Tensor(x_squeue.cpu().detach().numpy()).cuda()
            x0 = x_squeue[:,1:,:,:,:]
    
        return x0












class Guide_DiffusionSampler(nn.Module): ###
    def __init__(self, model, beta_1, beta_T, T,infer_T=None,squeue=None):
        '''
        推理时直接使用改T有问题,infer_T用于在某些截断
        squeue:用于保存过程中的结果,None只保留最终的结果,为数字时保留完整的,squeue=10
        '''
        super().__init__()

        self.model = model
        self.T = T
        if infer_T ==None:
            self.infer_T =T
        else:
            self.infer_T = infer_T
        self.squeue = squeue

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
    


    def guide_sample(self,x_0,t):
        t = t*torch.ones(size=(x_0.shape[0], ))
        device=x_0.device
        t= torch.tensor(t).type(torch.int64).to(device)
        # print(t)
        noise = torch.randn_like(x_0)
        # print('noise',noise.shape)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        return x_t

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T[:,0:1,...]
        x_guide = x_T[:,1:,...]
        for time_step in reversed(range(self.infer_T)):
            # print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise

            ##使用guide:
            # x_t = x_t#0.9*x_t + 0.1*self.guide_sample(x_guide,time_step)

            x_t = x_t # ((self.infer_T-time_step)/self.infer_T)*x_t  +(time_step/self.infer_T)*self.guide_sample(x_guide,time_step)
            

            # print('var',var)
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   


# from model import UNet
# from torchsummary import summary
# if __name__ == "__main__":
#     net_model=UNet(
#         T=1000, ch=32, ch_mult=[1, 2, 2, 2], attn=[1],
#         num_res_blocks=2, dropout=0.1).cuda()


#     trainer = GaussianDiffusionTrainer(
#     net_model, 1e-4, 0.02,1000).cuda()


#     x = torch.randn(2, 1, 96, 96,96).cuda()
#     loss=trainer(x)
#     print(x.shape)
#     summary(trainer,[1,96,96,96])
#     print(loss.shape)


        # sample_eval = GaussianDiffusionSampler(net_model, 1e-4, 0.02,100)

        # y_s=sample_eval(x)
        # print(y_s.shape)

