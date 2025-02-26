# %%

from base64 import b64encode

import numpy
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login

# For video display:
from IPython.display import HTML
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import os

torch.manual_seed(1)
# if not (Path.home()/'.cache/huggingface'/'token').exists(): notebook_login()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if "mps" == torch_device: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"



# %%

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device);

#%% 
# Some settings
prompt = ["An oil painting of many cats"]
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 100            # Number of denoising steps
guidance_scale = 7.5            # Scale for classifier-free guidance
generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
batch_size = 1

# Prep text
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# Prep Scheduler
def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32) # minor fix to ensure MPS compatibility, fixed in diffusers PR 3925

set_timesteps(scheduler,num_inference_steps)

# Prep latents with optional symmetry
batch_size = 1
symmetry_type = "repeat"  # Options: "none", "vertical", "horizontal", "repeat"
repeat_factor = 2  # Number of repetitions in each dimension (2 means 2x2 grid = 4 squares)


if symmetry_type == "none":
    # Standard asymmetric latents
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
else:
    if symmetry_type == "vertical":
        # Generate only half of the latent noise (left side)
        half_width = (width // 8) // 2  # Half width in latent space
        half_latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, half_width),
            generator=generator,
        )
        # Create the mirrored version (right side)
        mirrored_half = torch.flip(half_latents, [3])  # Flip along width dimension
        # Concatenate left and right
        latents = torch.cat([half_latents, mirrored_half], dim=3)
    
    elif symmetry_type == "horizontal":
        # Generate only half of the latent noise (top side)
        half_height = (height // 8) // 2  # Half height in latent space
        half_latents = torch.randn(
            (batch_size, unet.in_channels, half_height, width // 8),
            generator=generator,
        )
        # Create the mirrored version (bottom side)
        mirrored_half = torch.flip(half_latents, [2])  # Flip along height dimension
        # Concatenate top and bottom
        latents = torch.cat([half_latents, mirrored_half], dim=2)
    elif symmetry_type == "repeat":
        # First create full-size latents
        full_latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        
        # Calculate the size of each repeating section
        # Use integer division to ensure exact splits
        section_height = (height // 8) // repeat_factor
        section_width = (width // 8) // repeat_factor + 5 
        
        # Extract the base pattern from the top-left section
        # Be very precise with the slicing to avoid any overlap
        base_pattern = full_latents[:, :, 0:section_height, 5:section_width].clone()
        
        # Create full pattern by stacking copies
        rows = []
        for i in range(repeat_factor):
            # Create a row by concatenating horizontally
            row = torch.cat([base_pattern.clone() for _ in range(repeat_factor)], dim=3)
            rows.append(row)
        # Stack all rows vertically
        latents = torch.cat(rows, dim=2)
        
        # Verify the dimensions are exactly correct
        expected_shape = (batch_size, unet.in_channels, height // 8, width // 8)
        assert latents.shape == expected_shape, f"Shape mismatch: {latents.shape} vs {expected_shape}"

# Move to device and scale
latents = latents.to(torch_device)
latents = latents * scheduler.init_noise_sigma

# Loop
with autocast("cuda"):  # will fallback to CPU if no CUDA; no autocast for MPS
    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        # Scale the latents (preconditioning):
        # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5) # Diffusers 0.3 and below
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        # latents = scheduler.step(noise_pred, i, latents)["prev_sample"] # Diffusers 0.3 and below
        latents = scheduler.step(noise_pred, t, latents).prev_sample

# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

# Display
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0]
# %%
