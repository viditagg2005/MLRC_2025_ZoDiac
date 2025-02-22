import argparse
import yaml
import os
import logging
import shutil
import numpy as np
from PIL import Image 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from diffusers import AutoencoderKL, DDIMScheduler
from datasets import load_dataset
from diffusers.utils.torch_utils import randn_tensor
from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.wmpatch import GTWatermark, GTWatermarkMulti
from main.utils import *
from loss.loss import LossProvider
from loss.pytorch_ssim import ssim
import pandas as pd

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# Adding argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./example/config/config.yaml', help='Path to the config file')
parser.add_argument('--input_folder', type=str, default='./example/input', help='Path to the input folder')
parser.add_argument('--attack', type=bool, default=True, help='Whether to attack the image or not')
parser.add_argument('--adv_only', type=bool, default=False, help='Whether only adversarial attack needs to happen')
parser.add_argument('--budget', type=float, default=0.05, help='The budget for the attack')
parser.add_argument('--iters', type=int, default=50, help='The number of iterations for the attack')
parser.add_argument('--alpha', type=float, default=0.015, help='The alpha value for the attack')  
args = parser.parse_args()

logging.info(f'===== Load Config =====')
device = torch.device('cuda')
with open(args.config, 'r') as file:
    cfgs = yaml.safe_load(file)
logging.info(cfgs)
scheduler = DDIMScheduler.from_pretrained(cfgs['model_id'], subfolder="scheduler")
pipe = WMDetectStableDiffusionPipeline.from_pretrained(cfgs['model_id'], scheduler=scheduler).to(device)
pipe.set_progress_bar_config(disable=True)
input_folder_path = args.input_folder 
wm_path = cfgs['save_img']

# List of SSIM thresholds
ssim_thresholds = [0.8, 0.85, 0.90, 0.95, 0.99]

# Initialize metrics dictionary for each threshold
metrics = {threshold: {
    "ssim": [], "psnr": [], "normal": [],
    'diff_attacker_60':[], 'cheng2020-anchor_3':[], 'bmshj2018-factorized_3':[], 
    'black_white':[], 'AnisotropicDiffusion_blur':[], 'DirectionalGaussian_blur':[], 
    'hue_change':[], 'sepia':[], 'bm3d':[], 'all':[], 'all_norot':[]
} for threshold in ssim_thresholds}

if args.adv_only:
    for threshold in ssim_thresholds:
        metrics[threshold]["adv"] = []

tatta = 0
for imagename in os.listdir(input_folder_path):
    if cfgs['w_type'] == 'single':
        wm_pipe = GTWatermark(device, w_channel=cfgs['w_channel'], w_radius=cfgs['w_radius'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))
    elif cfgs['w_type'] == 'multi':
        wm_pipe = GTWatermarkMulti(device, w_settings=cfgs['w_settings'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))

    gt_img_tensor = get_img_tensor(os.path.join(input_folder_path, imagename), device)

    def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
        img_latents = pipe.get_image_latents(img_tensor, sample=False)
        reversed_latents = pipe.forward_diffusion(
            latents=img_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
        )
        return reversed_latents

    empty_text_embeddings = pipe.get_text_embedding('')
    init_latents_approx = get_init_latent(gt_img_tensor, pipe, empty_text_embeddings)
    init_latents = init_latents_approx.detach().clone()
    init_latents.requires_grad = True
    optimizer = optim.Adam([init_latents], lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.3) 

    totalLoss = LossProvider(cfgs['loss_weights'], device)
    loss_lst = [] 

    for i in range(cfgs['iters']):
        init_latents_wm = wm_pipe.inject_watermark(init_latents)
        if cfgs['empty_prompt']:
            pred_img_tensor = pipe('', guidance_scale=1.0, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm).images
        else:
            pred_img_tensor = pipe(prompt, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm).images
        loss = totalLoss(pred_img_tensor, gt_img_tensor, init_latents_wm, wm_pipe)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_lst.append(loss.item())
        if (i+1) in cfgs['save_iters']:
            path = os.path.join(wm_path, f"{imagename.split('.')[0]}_{i+1}.png")
            save_img(path, pred_img_tensor, pipe)
    torch.cuda.empty_cache()

    for ssim_threshold in ssim_thresholds:
        wm_img_path = os.path.join(wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}.png")
        wm_img_tensor = get_img_tensor(wm_img_path, device)
        ssim_value = ssim(wm_img_tensor, gt_img_tensor).item()

        def binary_search_theta(threshold, lower=0., upper=1., precision=1e-6, max_iter=1000):
            for i in range(max_iter):
                mid_theta = (lower + upper) / 2
                img_tensor = (gt_img_tensor-wm_img_tensor)*mid_theta+wm_img_tensor
                ssim_value = ssim(img_tensor, gt_img_tensor).item()

                if ssim_value <= threshold:
                    lower = mid_theta
                else:
                    upper = mid_theta
                if upper - lower < precision:
                    break
            return lower

        optimal_theta = binary_search_theta(ssim_threshold, precision=0.01)
        img_tensor = (gt_img_tensor-wm_img_tensor)*optimal_theta+wm_img_tensor

        ssim_value = ssim(img_tensor, gt_img_tensor).item()
        psnr_value = compute_psnr(img_tensor, gt_img_tensor)
        metrics[ssim_threshold]["ssim"].append(ssim_value)
        metrics[ssim_threshold]["psnr"].append(psnr_value)
        tester_prompt = '' 
        text_embeddings = pipe.get_text_embedding(tester_prompt)
        det_prob = 1 - watermark_prob(img_tensor, pipe, wm_pipe, text_embeddings)

        path = os.path.join(wm_path, f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold:.2f}.png")
        save_img(path, img_tensor, pipe)

        from main.wmattacker import *
        from main.attdiffusion import ReSDPipeline

        att_pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
        att_pipe.set_progress_bar_config(disable=True)
        att_pipe.to(device)

        if not args.adv_only:
            attackers = {
                'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
                'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
                'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
                'black_white': BlackAndWhiteAttacker(),
                'AnisotropicDiffusion_blur': AnisotropicDiffusionAttacker(num_iter=15, delta_t=0.14, kappa=50),
                'DirectionalGaussian_blur': DirectionalGaussianBlurAttacker(kernel_size=15, sigma=5, angle=45),
                'hue_change': HueChangeAttacker(factor=0.3),
                'sepia': SepiaAttacker(),
                'bm3d': BM3DAttacker(),
            }
        else:
            attackers = {
                'Motion_blur': MotionBlurAttacker(kernel_size=15, angle=45),
                'sepia': SepiaAttacker(),
                'hue_change': HueChangeAttacker(factor=0.3),
            }

        post_img = os.path.join(wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold:.2f}.png")
        for attacker_name, attacker in attackers.items():
            os.makedirs(os.path.join(wm_path, attacker_name), exist_ok=True)
            att_img_path = os.path.join(wm_path, attacker_name, os.path.basename(post_img))
            attackers[attacker_name].attack([post_img], [att_img_path])

        case_list = ['w/ rot', 'w/o rot']
        for case in case_list:
            if case == 'w/ rot':
                multi_name = 'all'
            elif case == 'w/o rot':
                multi_name = 'all_norot'
            
            os.makedirs(os.path.join(wm_path, multi_name), exist_ok=True)
            att_img_path = os.path.join(wm_path, multi_name, os.path.basename(post_img))
            for i, (attacker_name, attacker) in enumerate(attackers.items()):
                print(f'Attacking with {attacker_name}')
                if i == 0:
                    attackers[attacker_name].attack([post_img], [att_img_path], multi=True)
                else:
                    attackers[attacker_name].attack([att_img_path], [att_img_path], multi=True)

        tester_prompt = ''
        text_embeddings = pipe.get_text_embedding(tester_prompt)

        det_prob = 1 - watermark_prob(post_img, pipe, wm_pipe, text_embeddings)
        metrics[ssim_threshold]["normal"].append(det_prob)
        
        for attacker_name in attackers:
            if not os.path.exists(os.path.join(wm_path, attacker_name)):
                logging.info(f'Attacked images under {attacker_name} not exist.')
                continue
                
            det_prob = 1 - watermark_prob(os.path.join(wm_path, attacker_name, os.path.basename(post_img)), pipe, wm_pipe, text_embeddings)
            metrics[ssim_threshold][attacker_name].append(det_prob)

        # if args.attack:
        #     # Adversarial attack code here (omitted for brevity)
        #     # ...
        #     det_prob = 1 - watermark_prob(os.path.join(wm_path, 'perturbed_images', f"{imagename}_perturbed.png"), pipe, wm_pipe, text_embeddings)
        #     metrics[ssim_threshold]["adv"].append(det_prob)

    tatta += 1  
    for threshold in ssim_thresholds:
        df = pd.DataFrame(metrics[threshold])
        df.to_csv(os.path.join(wm_path, f'metrics_SSIM{threshold:.2f}.csv'))
    logging.info(f'Image {imagename} done')
    logging.info(f'Image number {tatta}')

logging.info(f'===== Done =====')
