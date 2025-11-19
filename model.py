import torch.nn as nn
import torch
import math
from unet import Unet
from tqdm import tqdm


class MNISTFlowModel(nn.Module):
    def __init__(self,image_size,in_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8],P_mean=-0.8,P_std=0.8,num_sampling_steps=100,device=None):
        super().__init__()
        self.device = device
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_size=image_size

        self.P_mean = P_mean
        self.P_std = P_std

        self.net=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)

        self.steps = num_sampling_steps
        self.t_eps = 5e-2
        # self.method =  'euler'  # sampling_method
        self.method =  'heun'  # sampling_method

    def sample_t(self, n):
        z = torch.randn(n, device=self.device) * self.P_std + self.P_mean
        return torch.sigmoid(z)
    
    def forward(self,x):
        t = self.sample_t(x.size(0)).to(self.device)

        e = torch.randn_like(x).to(self.device)

        z = t.view(-1,1,1,1) * x + (1 - t).view(-1,1,1,1) * e
        v = (x-z)/(1 - t).view(-1,1,1,1).clamp_min(self.t_eps)

        x_pred = self.net(z,t)
        v_pred = (x_pred - z) / (1 - t).view(-1,1,1,1).clamp_min(self.t_eps)

        loss = (v_pred - v).pow(2).mean(dim=(1, 2, 3)).mean()
        return loss

    @torch.no_grad()
    def generate(self, bsz):
        z = torch.randn(bsz, self.in_channels, self.image_size, self.image_size, device=self.device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=self.device)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[[i]]
            t_next = timesteps[[i + 1]]
            z = stepper(z, t, t_next)
        # last step euler
        z = self._euler_step(z, timesteps[[-2]], timesteps[[-1]])
        return z

    @torch.no_grad()
    def _euler_step(self, z, t, t_next):
        x_pred = self.net(z, t)
        v_pred =  (x_pred - z) / (1.0 - t).view(-1,1,1,1).clamp_min(self.t_eps)
        z_next = z + (t_next - t).view(-1, 1,1,1) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next):
        x_pred = self.net(z, t)
        v_pred_t  = (x_pred - z) / (1.0 - t).view(-1, 1, 1, 1).clamp_min(self.t_eps)

        z_next_euler = z + (t_next - t).view(-1, 1,1,1) * v_pred_t
        x_pred = self.net(z_next_euler, t_next)
        v_pred_t_next = (x_pred - z) / (1.0 - t_next).view(-1, 1, 1, 1).clamp_min(self.t_eps)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t).view(-1, 1,1,1) * v_pred
        return z_next

class MNISTDiffusion(nn.Module):
    def __init__(self,image_size,in_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8],device=None):
        super().__init__()
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_size=image_size

        self.betas=self._cosine_variance_schedule(timesteps).to(device)
        self.alphas=1.0-self.betas
        self.alphas_cumprod=torch.cumprod(self.alphas,axis=0)
        self.sqrt_alphas_cumprod=torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0-self.alphas_cumprod)

        self.net=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)

    def forward(self,x,noise):
        # x:NCHW
        t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)
        x_t=self._forward_diffusion(x,t,noise)
        pred_noise=self.net(x_t,t)
        return pred_noise

    @torch.no_grad()
    def sampling(self,n_samples,clipped_reverse_diffusion=True,device="cuda"):
        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t,t,noise)
            else:
                x_t=self._reverse_diffusion(x_t,t,noise)

        x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        return x_t
    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        return betas

    def _forward_diffusion(self,x_0,t,noise):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise


    @torch.no_grad()
    def _reverse_diffusion(self,x_t,t,noise):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        pred=self.net(x_t,t)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise 


    @torch.no_grad()
    def _reverse_diffusion_with_clip(self,x_t,t,noise): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.net(x_t,t)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise 
    