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

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd 

HOMEDIR = "/home/jh27kim"

def show_image(dataloader):
    batch_imgs = next(iter(dataloader))
    batch_imgs = batch_imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    batch_imgs = (batch_imgs * 255).round().astype('uint8')

    img = Image.fromarray(batch_imgs[0])
    img.save(f'./result2/my.png')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_img(opt, imgs, inout, rand_idx):
    print("Saving Image shape of", imgs.shape)

    imgs = torch.clip(imgs, 0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')

    img = Image.fromarray(imgs[0])
    img.save(f'./{opt.save_dir}/{inout}_{opt.prompt}_{opt.guidance_scale}_{rand_idx}.png')


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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

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


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=0):
        
        # interp to 512x512 to be fed into vae.

        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad) 

        return loss 

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

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

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


def train(opt, device):
    # data_dir = os.path.join(HOMEDIR, "train")
    # dataset = DatasetClass(data_dir, opt.H)
    # dataloader = DataLoader(dataset, batch_size = opt.batch_size, shuffle = True, pin_memory = True)

    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)
    text_embeddings= sd.get_text_embeds([opt.prompt], [opt.negative])

    steps = opt.steps

    # rand_idx = random.randint(0, len(dataset))
    rand_idx = 39185

    # x0 = dataset[rand_idx].clone().detach().unsqueeze(0).to(device)
    # x0.requires_grad_(True)

    x0 = Image.open(os.path.join(opt.save_dir, "panda.png")).convert('RGB')
    print("image mode", x0.mode)
    x0 = ToTensor()(x0).unsqueeze(0).to(device)
    x0.requires_grad_(True)
    print(x0.requires_grad, torch.min(x0), torch.max(x0), x0.shape, x0.device)

    # img_shape = (opt.batch_size, 3, opt.H, opt.W)
    # x0 = torch.randn(img_shape, device=device, dtype=torch.float, requires_grad=True)

    save_img(opt, x0, "in", rand_idx)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer = torch.optim.Adam([x0], lr=0.005,betas=(0.9,0.999))

    for t in tqdm(range(steps)):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            loss = sd.train_step(text_embeddings, x0, guidance_scale=opt.guidance_scale)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (t+1) % opt.save_step == 0:
            save_img(opt, x0, f"step_{t+1}", rand_idx)

    save_img(opt, x0, "out", rand_idx)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="a cat")
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--guidance_scale', type=float, default=100)
    parser.add_argument('--save_step', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default="")

    opt = parser.parse_args()

    # Show image
    # show_image(dataloder)
    seed_everything(opt.seed)

    device = torch.device('cuda')
    
    train(opt, device)