import pdb
import os
from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class DataConfig:
    meta_data: str = "/path/to/dataset/meta_data.csv"
    data_dir: str = "/path/to/dataset"
    meta_data_test: str = None
    max_frame_pixels: int = 480 * 832
    num_frames: float = None
    fps: float = None   # 2
    p_shuffle_frames: float = 0.0
    p_color_jitter: float = 0.0
    eval_dim: Union[str, List[str]] = "SA"
    prompt_template_type: str = "none"
    add_noise: bool = False
    sample_type: str = "uniform"
    use_tied_data: bool = True

def find_images(root_dir):
    exts = ('.jpg', '.jpeg', '.png', '.webp')
    results = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                results.append(os.path.join(root, f))
    return results