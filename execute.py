import os
import json

key = "OBJECT"

value = os.getenv(key, default=None)
concepts_list = [
    {
        "instance_prompt":      f"photo of zwx {value}",
        "class_prompt":         f"photo of a {value}",
        "instance_data_dir":    "/work/input",
        "class_data_dir":       "/work/class-images"
    }
]

# `class_data_dir` contains regularization images

for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)
