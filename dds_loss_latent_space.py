import math
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
import numpy as np
from PIL import Image
import random
import argparse
import matplotlib.pyplot as plt
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import ToTensor, ToPILImage

from dataset_load import DatasetClass

# suppress partial model loading warning
logging.set_verbosity_error()

import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd 

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad) 
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype) 

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_prompt', type=str, default=None)
    parser.add_argument('--tar_prompt', type=str, default=None)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--save_step', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default="")

    opt = parser.parse_args()

    device = torch.device('cuda')
    
    guidance = StableDiffusion(device, opt.sd_version, opt.hf_key)
    # guidance.vae.encoder = None

    src = Image.open(os.path.join(opt.save_dir, "flamingo_rollerskating.png")).convert('RGB')
    src = ToTensor()(src).unsqueeze(0).to(device)
    src_512 = F.interpolate(src, (512, 512), mode='bilinear', align_corners=False)

    tar = Image.open(os.path.join(opt.save_dir, "flamingo_rollerskating.png")).convert('RGB')
    tar = ToTensor()(tar).unsqueeze(0).to(device)
    tar_512 = F.interpolate(tar, (512, 512), mode='bilinear', align_corners=False)

    with torch.no_grad():
        src_latent = guidance.encode_imgs(src_512)
        tar_latent = guidance.encode_imgs(tar_512)

    src_latent.requires_grad_(True)
    tar_latent.requires_grad_(True)

    assert src_latent.shape[0] == tar_latent.shape[0]
    assert src_latent.shape[1] == tar_latent.shape[1]
    assert src_latent.shape[2] == tar_latent.shape[2]
    assert src_latent.shape[3] == tar_latent.shape[3]

    src_text_embeddings = guidance.get_text_embeds([opt.src_prompt], [opt.negative])
    tar_text_embeddings = guidance.get_text_embeds([opt.tar_prompt], [opt.negative])

    guidance.text_encoder.to('cpu')
    torch.cuda.empty_cache()

    seed_everything(42)

    optimizer = torch.optim.AdamW([tar_latent], lr=1e-1, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(opt.steps*1.5))

    for step in tqdm(range(opt.steps)):
        optimizer.zero_grad()

        t = torch.randint(guidance.min_step, guidance.max_step + 1, [1], dtype=torch.long, device=guidance.device)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(src_latent)

            src_zt = guidance.scheduler.add_noise(src_latent, noise, t)
            tar_zt = guidance.scheduler.add_noise(tar_latent, noise, t)

            # pred noise
            src_in = torch.cat([src_zt] * 2)
            src_noise_pred = guidance.unet(src_in, t, encoder_hidden_states=src_text_embeddings).sample

            tar_in = torch.cat([tar_zt] * 2)
            tar_noise_pred = guidance.unet(tar_in, t, encoder_hidden_states=tar_text_embeddings).sample

        w = (1 - guidance.alphas[t])

        # perform guidance (high scale from paper!)
        src_noise_pred_uncond, src_noise_pred_text = src_noise_pred.chunk(2)
        src_noise_pred = src_noise_pred_uncond + opt.guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
        src_grad = src_noise_pred - noise

        tar_noise_pred_uncond, tar_noise_pred_text = tar_noise_pred.chunk(2)
        tar_noise_pred = tar_noise_pred_uncond + opt.guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)
        tar_grad = tar_noise_pred - noise
        
        grad = w * (tar_grad - src_grad)

        tar_latent.backward(gradient=grad, retain_graph=True)

        print(tar_latent[0][0])

        optimizer.step()
        scheduler.step()

        if not step or step % opt.save_step == 0:
            rgb = guidance.decode_latents(tar_latent)
            img = rgb.detach().squeeze(0).permute(1,2,0).cpu().numpy()
            print('[INFO] save image', img.shape, img.min(), img.max())
            plt.imsave(f'./{opt.save_dir}/{opt.tar_prompt}_{opt.guidance_scale}_{step}.png', img)