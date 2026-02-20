import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
from omegaconf import OmegaConf

# VQGAN-CLIP core imports
import sys
# Make sure the local folders for taming-transformers and CLIP are in the path
# Since we are in the root of the custom node, it's just sys.path.append('.')
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'taming-transformers'))
sys.path.append(os.path.join(current_dir, 'CLIP'))

from taming.models import cond_transformer, vqgan
from CLIP import clip
import kornia.augmentation as K
from torch_optimizer import AdamP

# Import constants and helper functions from generate.py logic
# (Copied and adapted for a class-based structure)

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply

def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)

class Prompt(torch.nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()

def split_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., augments=None):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        
        if augments is None:
            augments = ['Af', 'Pe', 'Ji', 'Er']
            
        augment_list = []
        for item in augments:
            if item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True))
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
                
        self.augs = torch.nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        self.av_pool = torch.nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = torch.nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []
        for _ in range(self.cutn):            
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch

# Model Caching
_cached_models = {}

def load_vqgan_model(config_path, checkpoint_path):
    cache_key = (config_path, checkpoint_path)
    if cache_key in _cached_models:
        return _cached_models[cache_key]
    
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    
    _cached_models[cache_key] = model
    return model

def load_clip_model(clip_model_name, device):
    cache_key = (clip_model_name, str(device))
    if cache_key in _cached_models:
        return _cached_models[cache_key]
    
    jit = True if "1.7.1" in torch.__version__ else False
    perceptor = clip.load(clip_model_name, jit=jit)[0].eval().requires_grad_(False).to(device)
    
    _cached_models[cache_key] = perceptor
    return perceptor

class VQGANCLIP_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_prompt": ("STRING", {"multiline": True, "default": "A painting of an apple in a fruit bowl"}),
                "iterations": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
                "cutn": ("INT", {"default": 32, "min": 1, "max": 512}),
                "vqgan_model": (["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "vqgan_gumbel_f8_8192", "faceshq", "wikiart_16384", "sflckr"],),
                "clip_model": (["ViT-B/32", "ViT-B/16", "ViT-L/14"],),
            },
            "optional": {
                "init_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "VQGAN-CLIP"

    def generate(self, text_prompt, iterations, seed, width, height, learning_rate, cutn, vqgan_model, clip_model, init_image=None):
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths for configs and checkpoints
        base_path = os.path.dirname(os.path.abspath(__file__))
        checkpoints_path = os.path.join(base_path, "checkpoints")
        os.makedirs(checkpoints_path, exist_ok=True)
        
        config_path = os.path.join(checkpoints_path, f"{vqgan_model}.yaml")
        checkpoint_path = os.path.join(checkpoints_path, f"{vqgan_model}.ckpt")
        
        if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"VQGAN config or checkpoint not found at {config_path} / {checkpoint_path}. \nPlease run the provided download_models.sh script in the custom node directory: \ncd {base_path} && bash download_models.sh")

        model = load_vqgan_model(config_path, checkpoint_path).to(device)
        perceptor = load_clip_model(clip_model, device)
        
        cut_size = perceptor.visual.input_resolution
        f = 2**(model.decoder.num_resolutions - 1)
        make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=1.0).to(device)
        
        toksX, toksY = width // f, height // f
        sideX, sideY = toksX * f, toksY * f
        
        # Initialize z
        gumbel = isinstance(model, vqgan.GumbelVQ)
        if gumbel:
            e_dim = 256
            n_toks = model.quantize.n_embed
            z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            e_dim = model.quantize.e_dim
            n_toks = model.quantize.n_e
            z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if init_image is not None:
            # init_image is [B, H, W, C] in ComfyUI
            # We need [1, C, H, W] and scaled to 0-1
            img_tensor = init_image[0].permute(2, 0, 1).unsqueeze(0).to(device)
            # Resize to VQGAN compatible size
            img_tensor = F.interpolate(img_tensor, (sideY, sideX), mode='bicubic', align_corners=True)
            z, *_ = model.encode(img_tensor * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            if gumbel:
                z = one_hot @ model.quantize.embed.weight
            else:
                z = one_hot @ model.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

        z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=learning_rate)
        
        # Prompts
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                          std=[0.26862954, 0.26130258, 0.27577711])
        
        pMs = []
        # Multi-prompt support via |
        prompts = [phrase.strip() for phrase in text_prompt.split("|")]
        for prompt in prompts:
            txt, weight, stop = split_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))
            
        def synth(z):
            if gumbel:
                z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
            else:
                z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
            return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

        torch.manual_seed(seed)
        
        # Training loop
        for i in tqdm(range(iterations)):
            opt.zero_grad(set_to_none=True)
            
            out = synth(z)
            iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
            
            losses = []
            for prompt in pMs:
                losses.append(prompt(iii))
            
            loss = sum(losses)
            loss.backward()
            opt.step()
            
            with torch.no_grad():
                z.copy_(z.maximum(z_min).minimum(z_max))
        
        # Final output
        with torch.no_grad():
            final_out = synth(z)
            # final_out is [1, 3, H, W], scaled 0-1
            # ComfyUI expects [B, H, W, C]
            res = final_out[0].permute(1, 2, 0).unsqueeze(0).cpu()
            return (res,)

NODE_CLASS_MAPPINGS = {
    "VQGANCLIP_Node": VQGANCLIP_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VQGANCLIP_Node": "VQGAN-CLIP Generator"
}
