import ast
import json
import os
import pdb
from collections.abc import Mapping
import pandas as pd
import numpy as np

import torch
from vace_reward.vision_process import process_vision_info

from vace_reward.data import DataConfig
from vace_reward.utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from vace_reward.utils import load_model_from_checkpoint
from vace_reward.train_reward import create_model_and_processor
from vace_reward.prompt_template import build_prompt


def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return config_dict["data_config"], None, config_dict["model_config"], config_dict["peft_lora_config"], \
           config_dict["inference_config"] if "inference_config" in config_dict else None

class VideoVLMRewardInference():
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        # training_args = TrainingConfig(
        #     load_from_pretrained=load_from_pretrained,
        #     load_from_pretrained_step=load_from_pretrained_step,
        #     gradient_checkpointing=False,
        #     disable_flash_attn2=False,
        #     bf16=True if dtype == torch.bfloat16 else False,
        #     fp16=True if dtype == torch.float16 else False,
        #     output_dir="",
        # )
        
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            # training_args=training_args,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config

    def _norm(self, reward):
        return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, batch_ref_paths, batch_video_paths, batch_prompts, fps=None, num_frames=None, max_pixels=None,):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels
        
        def get_ref_imgs_convs(ref_paths):  #
            reference_images_convs = [
                        {
                        "type": "image",
                        "image": ref_img,
                        "max_pixels": max_pixels, 
                        } for ref_img in ref_paths
                        ]
            return reference_images_convs

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": get_ref_imgs_convs(ref_paths=ref_paths) + [  
                            {
                                "type": "video", 
                                "video": f"{video_path}",
                                "max_pixels": max_pixels, 
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for ref_paths, video_path, prompt in zip(batch_ref_paths, batch_video_paths, batch_prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": get_ref_imgs_convs(ref_paths) + [   
                            {
                                "type": "video",
                                "video": f"{video_path}",
                                "max_pixels": max_pixels, 
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for ref_paths, video_path, prompt in zip(batch_ref_paths, batch_video_paths, batch_prompts)
            ]
        image_inputs, video_inputs, videos_kwargs = process_vision_info(chat_data, return_video_kwargs=True)
        # print(video_inputs[0].shape, videos_kwargs)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True, add_vision_id=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True, "fps": videos_kwargs["fps"]},
        )
        # # print(batch)
        batch = self._prepare_inputs(batch)
        return batch

    def __call__(self, ref_paths, video_paths, prompts="", fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """
        Inputs:
            video_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
        
        batch = self.prepare_batch(ref_paths, video_paths, prompts, fps, num_frames, max_pixels)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]
        # print(rewards)
        rewards = [reward[0].item() for reward in rewards]

        return rewards


if __name__ == "__main__":
    load_from_pretrained = "outputs/identity_reward"
    load_from_pretrained_step = -1
    device = "cuda:0"
    dtype = torch.bfloat16

    inferencer = VideoVLMRewardInference(load_from_pretrained, load_from_pretrained_step, device=device, dtype=dtype)

    video_paths = [
        "./数据/a4156e42-MAGREF-VACE14B/A.mp4",
        "./数据/a4156e42-MAGREF-VACE14B/B.mp4",
        "./数据/a4134234-PHANTOM1_3B-VACE1_3B/A.mp4",
        "./数据/a4134234-PHANTOM1_3B-VACE1_3B/B.mp4",
    ]
    prompts = [
        "The video shows two people, a woman in a red sweater and a man in a red and gray checkered sweater, seated at a dining table, using chopsticks to eat from bowls in a relaxed, homey setting with a microwave, framed photo, and figurines visible in the background. The soft, natural lighting and casual atmosphere suggest a daytime meal in a kitchen or dining area.",
        "The video shows two people, a woman in a red sweater and a man in a red and gray checkered sweater, seated at a dining table, using chopsticks to eat from bowls in a relaxed, homey setting with a microwave, framed photo, and figurines visible in the background. The soft, natural lighting and casual atmosphere suggest a daytime meal in a kitchen or dining area.",
        "A young couple, likely in their twenties, strolls side by side on a sidewalk in a residential neighborhood, surrounded by a multi-story building with balconies and palm trees, as they enjoy a casual, daytime outing. The man, wearing a striped sweater and maroon pants, and the woman, dressed in a light pink cardigan and skirt, exude a relaxed atmosphere as they walk together.",
        "A young couple, likely in their twenties, strolls side by side on a sidewalk in a residential neighborhood, surrounded by a multi-story building with balconies and palm trees, as they enjoy a casual, daytime outing. The man, wearing a striped sweater and maroon pants, and the woman, dressed in a light pink cardigan and skirt, exude a relaxed atmosphere as they walk together.",
    ]
    reference_images_paths = [
        ["./数据/a4156e42-MAGREF-VACE14B/reference_img001.jpg", "./数据/a4156e42-MAGREF-VACE14B/reference_img002.jpg"],
        ["./数据/a4156e42-MAGREF-VACE14B/reference_img001.jpg", "./数据/a4156e42-MAGREF-VACE14B/reference_img002.jpg"],
        ["./数据/a4134234-PHANTOM1_3B-VACE1_3B/reference_img001.jpg", "./数据/a4134234-PHANTOM1_3B-VACE1_3B/reference_img002.jpg"],
        ["./数据/a4134234-PHANTOM1_3B-VACE1_3B/reference_img001.jpg", "./数据/a4134234-PHANTOM1_3B-VACE1_3B/reference_img002.jpg"],
    ]

    with torch.no_grad():
        rewards = inferencer(reference_images_paths, video_paths, prompts, use_norm=False)
        print(rewards)