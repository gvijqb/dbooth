import os
import json
import subprocess

prompt_value = os.getenv("OBJECT", default=None)
model_name = os.getenv("MODEL_NAME", default=None)
output_dir = os.getenv("OUTPUT_DIR", default=None)

# Create Concepts list json file
concepts_list = [
    {
        "instance_prompt":      f"photo of zwx {prompt_value}",
        "class_prompt":         f"photo of a {prompt_value}",
        "instance_data_dir":    "/work/input",
        "class_data_dir":       "/work/class-images"
    }
]

for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)

# Run finetune command
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


