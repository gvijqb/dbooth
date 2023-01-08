import os
import json
import subprocess
import string
import random
import urllib.request

current_dir = os.getcwd()
model_name = os.getenv("MODEL_NAME", default=None)
output_dir = os.getenv("OUTPUT_DIR", default=None)
final_images_dir = os.getenv("FINAL_IMAGES", default=None)
image_urls = os.getenv("IMAGE_URLS", default=None)
prompt_value = os.getenv("OBJECT", default=None)


prompt_list = [
                f"photo of zwx {prompt_value} as santa claus, artstation, hd, dramatic lighting, detailed",
                f"photo of zwx {prompt_value} as superman, artstation, hd, dramatic lighting, detailed",
                f"photo of zwx {prompt_value} as batman, artstation, hd, dramatic lighting, detailed",
                f"photo of zwx {prompt_value} as ironman, artstation, hd, dramatic lighting, detailed",
                f"photo of zwx {prompt_value} as ghostrider, artstation, hd, dramatic lighting, detailed",
                f"photo of zwx {prompt_value} as astronaut, artstation, hd, dramatic lighting, detailed",
                f"photo of zwx {prompt_value} as astronaut in space, artstation, hd, dramatic lighting, detailed",
                f"photo of zwx {prompt_value} in a painting, by van gogh, artstation, hd, dramatic lighting, detailed",
                f"photo of zwx {prompt_value} in front of Dubai skyline, bokeh effect, artstation, hd, dramatic lighting, detailed"
              ]

# Tasks performed:
# 1. Download and save all the user provided images
# 2. Create or refer to a class-image directory for the provided prompt
# 3. Create a concepts_list.json file
# 4. Run fine-tune command
# 5. Get generated weights and load them to GPU
# 6. Perform Image Generation and Save them

# 1. Download and save all the user provided images
image_urls = image_urls.split()
for image_url in image_urls:
    if image_url.endswith(('.png', '.jpg', '.jpeg')):
        # Download the image
        image_data = urllib.request.urlopen(image_url).read()

        # Get the file name from the URL
        image_name = image_url.split("/")[-1]

        # Save the image to the "input" folder
        if not os.path.exists("input"):
            os.makedirs("input")
        with open(f"input/{image_name}", "wb") as f:
            f.write(image_data)
    else:
        print(f"Error: {image_url} does not point to an image")
# ----------------- end -------------- 


# 2. Create or refer to a class-image directory for the provided prompt
if prompt_value.lower() in ("man", "woman","person"): 
    class_data_dir = f"{current_dir}/sd1.5/{prompt_value}"
else:
    classname = prompt_value.lower().split(None, 1)[0]
    class_data_dir = f"{current_dir}/sd1.5/{classname}"
    os.makedirs(class_data_dir, exist_ok=True)
# ----------------- end --------------

# 3. Create a concepts_list.json file
concepts_list = [
    {
        "instance_prompt":      f"photo of zwx {prompt_value}",
        "class_prompt":         f"photo of a {prompt_value}",
        "instance_data_dir":    f"{current_dir}/input",
        "class_data_dir":       f"{class_data_dir}"
    }
]

for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)
# ----------------- end --------------

# 4. Run fine-tune command
print("Starting Training process")
command = ["accelerate", "launch", "train_dreambooth.py",
           "--pretrained_model_name_or_path=" + model_name,
           "--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse",
           "--output_dir=" + output_dir,
           "--with_prior_preservation", "--prior_loss_weight=1.0",
           "--seed=1337",
           "--resolution=512",
           "--train_batch_size=1",
           "--train_text_encoder",
           "--mixed_precision=fp16",
           "--use_8bit_adam",
           "--gradient_accumulation_steps=1",
           "--learning_rate=1e-6",
           "--lr_scheduler=constant",
           "--lr_warmup_steps=0",
           "--num_class_images=200",
           "--sample_batch_size=4",
           "--max_train_steps=1900",
           "--save_interval=10000",
           "--save_sample_prompt=photo of zwx "+prompt_value,
           "--concepts_list=concepts_list.json"]

subprocess.run(command)
print("Training completed")
# ----------------- end --------------

# INFERENCE

# 5. Get generated weights and load them to GPU
from natsort import natsorted
from glob import glob
WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]

print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display

model_path = WEIGHTS_DIR
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")
g_cuda = None
g_cuda = torch.Generator(device='cuda')
seed = 52362 #@param {type:"number"}
g_cuda.manual_seed(seed)
print("Weights loaded on GPU")

# 6. Perform Image Generation and Save them
for i in range(len(prompt_list)):
    prompt = prompt_list[i]
    negative_prompt = "lowres, signs, memes, labels, text, food, text, error, mutant, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), facebook, youtube, body horror, out of frame, mutilated, tiled, frame, border, porcelain skin, doll like, doll, bad quality, cartoon, lowres, meme, low quality, worst quality, ugly, disfigured"
    num_samples = 4
    guidance_scale = 12
    num_inference_steps = 100
    height = 512
    width = 512

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    for img in images:
        final_name = f"{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}.png"
        img.save(final_images_dir+"/"+final_name)
    
    print(f"Saved images for prompt: {prompt}")
