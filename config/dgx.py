import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def vace():
    config = base.get_config()

    config.dataset = os.path.join(os.getcwd(), "dataset/generated_img")     # 数据集必须包含 train.csv 和 test.csv
    
    config.pretrained.reward_model = "outputs/identity_reward"
    config.pretrained.model = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    # config.pretrained.model = "/data1/public_models/Wan-AI/Wan2.1-VACE-1.3B-diffusers/"

    config.sample.num_steps = 25 # 采样步数
    config.sample.eval_num_steps = 50 # 测试步数
    config.sample.guidance_scale=4.5 # cfg
    config.run_name = "vace_identity_grpo"
    
    config.height = 240 # 采样视频的高
    config.width = 416 # 采样视频的宽
    config.frames = 33 # 采样视频的帧数
    config.eval_height = 480 # 测试视频高
    config.eval_width = 832 # 测试视频宽
    config.eval_frames = 81 # 测试视频帧数
    config.sample.train_batch_size = 4 # 每张卡上采样视频数
    config.sample.num_image_per_prompt = 8 # group_size
    config.sample.num_batches_per_epoch = 8 # 每次采样的轮数
    config.sample.sample_time_per_prompt = 1
    config.sample.test_batch_size = 2

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt // 2 if (config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt) > 1 else 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.004
    config.train.learning_rate = 1e-4
    config.train.clip_range=1e-3
    # kl reward
    # KL reward and KL loss are two ways to incorporate KL divergence. KL reward adds KL to the reward, while KL loss, introduced by GRPO, directly adds KL loss to the policy loss. We support both methods, but KL loss is recommended as the preferred option.
    config.sample.kl_reward = 0
    # We also support using SFT data in RL training for supervised learning to prevent quality drop, but this option was unused
    config.train.sft=0.0
    config.train.sft_batch_size=3
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std=False
    config.train.ema=True
    config.mixed_precision = "bf16"
    config.diffusion_loss = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.num_epochs = 100000
    config.save_freq = 5 # epoch
    config.eval_freq = 5
    config.resume_from = None
    config.reward_fn = {
        "vace": 1.0,
    }
    # 保存路径
    config.save_dir = f'outputs/identity_grpo/logs/vace/wan-1.3b'           # log dir
    config.ckpt_dir = 'outputs/identity_grpo/ckpt/vace'                     # save ckpt dir
    config.train_save_dir = "outputs/identity_grpo/save_video_train/vace"   # save video dir when training
    config.test_save_dir = "outputs/identity_grpo/save_video_test/vace"     # save video dir when testing
    config.prompt_fn = "i2v"    # vace

    config.eval_baseline = False

    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()
