import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import json
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, dispatch_model
from tqdm import tqdm
import copy
import math


def parse_arguments():
    parser = argparse.ArgumentParser(description='video-image Inference')
    parser.add_argument('--model_path', type = str, help="model_path")
    parser.add_argument('--video_root', type = str, help="video_root")
    parser.add_argument('--image_root', type = str, help="image_root")
    parser.add_argument('--data_path', type = str, help="data_path")
    parser.add_argument('--output_path', type = str, help="output_path")
    parser.add_argument('--model_id', type = str, help="model_id")

    args = parser.parse_args()
    return args


def video_infer(video_path, question, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 336 * 336,
                    "nframes": 64
                },
                {
                    "type": "text", 
                    "text": question
                },
            ],
        }
    ]
    template = {"type": "image",  "image": ""}

    if isinstance(image_path, str):
        template["image"] = image_path
        messages[0]["content"].append(template)
    elif isinstance(image_path, list):
        for img in image_path:
            template["image"] = img
            messages[0]["content"].append(copy.deepcopy(template))

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs,
                                       max_new_tokens=64,  
                                      # temperature=0.0,
                                    do_sample = False
                                       )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

if __name__ == "__main__":
    args = parse_arguments()


    model_path = args.model_path

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True).eval()

    # default processer
    min_pixels = 256 * 28 * 28
    max_pixels = 8192 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels,
                                            max_pixels=max_pixels)
    
    video_root = args.video_root
    image_root = args.image_root
    data_path = args.data_path
    output_path = args.output_path

    data_info = [json.loads(line) for line in open(data_path, "r", encoding="utf-8")]
    for idx, line in enumerate(tqdm(data_info)):

        line_copy = copy.deepcopy(line)
        video_path = os.path.join(video_root, line["video_path"]).replace("\\", "/")


        if isinstance(line["frame_path"], str):
            image_path = os.path.join(image_root, line["frame_path"]).replace("\\", "/")
        elif isinstance(line["frame_path"], list):
            image_path = [os.path.join(image_root, img).replace("\\", "/") for img in line["frame_path"]]

        question = line["question"]

        # strict infer
        response = video_infer(video_path, question, image_path)
        line_copy["response"] = response
        line_copy["model"] = args.model_id
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line_copy, ensure_ascii=False) + '\n')
