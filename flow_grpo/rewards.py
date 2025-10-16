from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

# from einops import rearrange
# def tensor2video(self, frames):
#     frames = rearrange(frames, "C T H W -> T H W C")
#     frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
#     frames = [Image.fromarray(frame) for frame in frames]
#     return frames

# def qwenvl_score(scorer):
#     def _fn(videos, prompts, metadata):
#         if isinstance(videos, torch.Tensor):
#             videos = ((videos+1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
#             videos = videos.transpose(0,2,3,4,1) # NCTHW -> NTHWC
#             videos = [video for video in videos]
#         prompts = [prompt for prompt in prompts]
#         scores = scorer(prompts, videos)
#         return scores, {}

#     return _fn

def qwenvl_score(scorer):
    def _fn(video_paths, prompts, test=False):
        prompts = [prompt for prompt in prompts]
        scores = scorer(video_paths, prompts, test=test)
        return scores, {}

    return _fn


def vace_score(scorer):
    def _fn(ref_paths, video_paths, prompts, test=False):
        prompts = [prompt for prompt in prompts]
        scores = scorer(ref_paths, video_paths, prompts)
        return scores, {}

    return _fn

# def nexus_score(scorer):
#     def _fn(video_paths, prompt_images, image_labels, test=False):
#         prompt_images = [prompt_image for prompt_image in prompt_images]
#         scores = scorer(video_paths, prompt_images, image_labels, test=test)
#         return scores, {}
    
#     return _fn

def facesim_score(scorer):
    def _fn(video_paths, prompt_images, test=False):
        prompt_images = [prompt_image for prompt_image in prompt_images]
        scores = scorer(video_paths, prompt_images, test=test)
        return scores, {}
    
    return _fn

def multi_score(scorer, score_dict, test=False):
    score_functions = {
        "qwenvl": qwenvl_score,
        "facesim": facesim_score,
        "vace": vace_score,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](scorer) # if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name](pretrain_path)
    def _fn(ref_paths, video_paths, prompts, test=test):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name=='vace':
                scores, rewards = score_fns[score_name](ref_paths, video_paths, prompts, test=test)
            else:
                scores, rewards = score_fns[score_name](video_paths, prompts, test=test)
            score_details[score_name] = scores
            if test:
                weighted_scores = scores
            else:
                weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()