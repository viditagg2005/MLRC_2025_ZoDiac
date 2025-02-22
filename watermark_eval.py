import argparse
import yaml
import os
import logging
import shutil
import numpy as np
from PIL import Image 
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import torch.optim as optim
from diffusers import DDIMScheduler
from datasets import load_dataset
from diffusers.utils.torch_utils import randn_tensor
from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.wmpatch import GTWatermark, GTWatermarkMulti
from main.utils import *
from loss.loss import LossProvider
from loss.pytorch_ssim import ssim
import pandas as pd

# Adding argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./example/config/config.yaml', help='Path to the config file')
parser.add_argument('--input_folder', type=str, default='./example/input', help='Path to the input folder')
parser.add_argument('--attack', type=bool, default=True, help='Whether to attack the image or not')
parser.add_argument('--adv_only', type=bool, default=False, help='Whether only adversarial attack needs to happen')
parser.add_argument('--budget', type=int, default=0.05, help='The budget for the attack')
parser.add_argument('--iters', type=int, default=50, help='The number of iterations for the attack')
parser.add_argument('--alpha', type=int, default=0.015, help='The alpha value for the attack')  
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
# A dictionary to maintain running average of all the metrics

if not args.adv_only:
    metrics = {"ssim": [], "psnr": [], "normal": [], "adv": [] ,'diff_attacker_60':[],'cheng2020-anchor_3':[], 'bmshj2018-factorized_3':[], 'jpeg_attacker_50':[],'jpeg_attacker_90':[],'jpeg_attacker_99':[], 
                'brightness_0.2':[],'brightness_0.4':[],'brightness_0.5':[],'brightness_0.6':[],'brightness_0.8':[],'brightness_1.25':[],'brightness_1.5':[] ,
                'contrast_0.2':[],'contrast_0.4':[],'contrast_0.6':[],'contrast_0.5':[],'contrast_0.8':[],'contrast_1.25':[],'contrast_1.5':[],'vibrancy_1.25':[],'black_white':[],'lateral_inversion':[], 'Gaussian_noise':[], 'Gaussian_blur':[],'AnisotropicDiffusion_blur':[],
                'DirectionalGaussian_blur':[],'sharpening':[],'salt_pepper_noise':[],'hue_change_0.1':[],'hue_change_0.3':[],
                'hue_change_0.5':[],'hue_change_0.7':[],'elastic_deformation':[],'RGBtoHSV':[],'color_balance':[],'gamma':[],'HistogramEqualization':[],
                'log_transform':[],'color_jitter':[],'color_quantization':[],'sepia':[],'posterization':[],'rotate_30':[],'rotate_60':[],'rotate_90':[],'rotate_120':[],'rotate_150':[],'rotate_180':[],'rotate_210':[],'rotate_240':[],
                'rotate_270':[],'lateral_rotate':[], 'bm3d':[],'all':[], 'all_norot':[]} #CHANGE
else:
    metrics = {"ssim": [], "psnr": [], "normal": [], "adv": []}


tatta = 0
############################################################################################################
for imagename in os.listdir(input_folder_path):
    if cfgs['w_type'] == 'single':
        wm_pipe = GTWatermark(device, w_channel=cfgs['w_channel'], w_radius=cfgs['w_radius'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))
    elif cfgs['w_type'] == 'multi':
        wm_pipe = GTWatermarkMulti(device, w_settings=cfgs['w_settings'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))

    #imagename = 'pepper.tiff'
    gt_img_tensor = get_img_tensor(os.path.join(input_folder_path,imagename), device)
    # Step 1: Get init noise
    def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
        # DDIM inversion from the given image
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
    # Step 2: prepare training
    init_latents = init_latents_approx.detach().clone()
    init_latents.requires_grad = True
    optimizer = optim.Adam([init_latents], lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.3) 

    totalLoss = LossProvider(cfgs['loss_weights'], device)
    loss_lst = [] 
    # Step 3: train the init latents
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
        # save watermarked image
        if (i+1) in cfgs['save_iters']:
            path = os.path.join(wm_path, f"{imagename.split('.')[0]}_{i+1}.png")
            save_img(path, pred_img_tensor, pipe)
    torch.cuda.empty_cache()

    ssim_threshold = cfgs['ssim_threshold']

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
    metrics["ssim"].append(ssim_value)
    metrics["psnr"].append(psnr_value)
    tester_prompt = '' 
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    det_prob = 1 - watermark_prob(img_tensor, pipe, wm_pipe, text_embeddings)

    path = os.path.join(wm_path, f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold}.png")
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
        'jpeg_attacker_50': JPEGAttacker(quality=50),
        'jpeg_attacker_90': JPEGAttacker(quality=10),
        'jpeg_attacker_99': JPEGAttacker(quality=1),
        'rotate_30': RotateAttacker(degree = 30),
        'rotate_60': RotateAttacker(degree = 60),
        'rotate_90': RotateAttacker(degree=90),
        'rotate_120': RotateAttacker(degree = 120),
        'rotate_150': RotateAttacker(degree = 150),
        'rotate_180': RotateAttacker(degree = 180),
        'rotate_210': RotateAttacker(degree = 210),
        'rotate_240': RotateAttacker(degree = 240),
        'rotate_270': RotateAttacker(degree = 270),
        'lateral_rotate': LateralRotateAttacker(),
        'brightness_0.2': BrightnessAttacker(brightness=0.2),
        'brightness_0.4': BrightnessAttacker(brightness=0.4),
        'brightness_0.5': BrightnessAttacker(brightness=0.5),
        'brightness_0.6': BrightnessAttacker(brightness=0.6),
        'brightness_0.8': BrightnessAttacker(brightness=0.8),
        'brightness_1.25': BrightnessAttacker(brightness=1.25),
        'brightness_1.5': BrightnessAttacker(brightness=1.5),
        'contrast_0.2': ContrastAttacker(contrast=0.2),
        'contrast_0.4': ContrastAttacker(contrast=0.4),
        'contrast_0.5': ContrastAttacker(contrast=0.5),
        'contrast_0.6': ContrastAttacker(contrast=0.6),
        'contrast_0.8': ContrastAttacker(contrast=0.8),
        'contrast_1.25': ContrastAttacker(contrast=1.25),
        'contrast_1.5': ContrastAttacker(contrast=1.5),
        'vibrancy_1.25': VibrancyAttacker(vibrancy = 1.25),
        'black_white': BlackAndWhiteAttacker(),
        'lateral_inversion':LateralInversionAttacker(),
        'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
        'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
        'Motion_blur': MotionBlurAttacker(kernel_size=15, angle=45),
        'OutofFocus_blur':OutOfFocusBlurAttacker(kernel_size = 15),
        'Radial_blur': RadialBlurAttacker(strength = 10),
        'Zoom_blur':ZoomBlurAttacker(strength=20),
        'Atmospheric_blur': AtmosphericBlurAttacker(strength=0.5),
        'PSF_blur': PSFBlurAttacker(kernel_size=15, angle=45),
        'Bilateral_blur': BilateralFilterAttacker(d=9, sigmaColor=75, sigmaSpace=75),
        'Iterative_blur': IterativeBlurAttacker(iterations =3),
        'AnisotropicDiffusion_blur':AnisotropicDiffusionAttacker(num_iter=15, delta_t=0.14, kappa=50),
        'DirectionalGaussian_blur': DirectionalGaussianBlurAttacker(kernel_size=15, sigma=5, angle=45),
        'sharpening': SharpeningAttacker(factor = 2.0),
        'salt_pepper_noise':SaltAndPepperNoiseAttacker(amount = 0.1),
        'hue_change_0.1': HueChangeAttacker(factor = 0.1),
        'hue_change_0.3': HueChangeAttacker(factor = 0.3),
        'hue_change_0.5': HueChangeAttacker(factor = 0.5),
        'hue_change_0.7': HueChangeAttacker(factor = 0.7),
        'elastic_deformation': ElasticDeformationAttacker(alpha=1000, sigma=50),
        'RGBtoHSV':RGBtoHSVAttacker(h_shift=0.1, s_scale=1.2, v_scale=1.1),
        'color_balance': ColorBalanceAttacker( r_scale=1.2, g_scale=1.0, b_scale=0.8),
        'gamma':GammaAttacker(gamma = 1.5),
        'HistogramEqualization':HistogramEqualizationAttacker(),
        'log_transform': LogTransformAttacker(c=1),
        'color_jitter':ColorJitterAttacker(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        'color_quantization':ColorQuantizationAttacker(n_colors = 32),
        'sepia':SepiaAttacker(),
        'posterization': PosterizationAttacker(levels =4),
        'bm3d': BM3DAttacker(),
        }#CHANGE
    else:
        attackers = {
            'Motion_blur': MotionBlurAttacker(kernel_size=15, angle=45),
            'sepia':SepiaAttacker(),
            'hue_change': HueChangeAttacker(factor = 0.3),
        }

    post_img = os.path.join(wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}.png")
    for attacker_name, attacker in attackers.items():
        os.makedirs(os.path.join(wm_path, attacker_name), exist_ok=True)
        att_img_path = os.path.join(wm_path, attacker_name, os.path.basename(post_img))
        attackers[attacker_name].attack([post_img], [att_img_path])

    from main.wmattacker import *
    from main.attdiffusion import ReSDPipeline

    case_list = ['w/ rot', 'w/o rot']

    att_pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
    att_pipe.set_progress_bar_config(disable=True)
    att_pipe.to(device)
    for case in case_list:
        if case == 'w/ rot':
            if not args.adv_only:
                attackers = {
                    'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
                    'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
                    'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
                    'jpeg_attacker_50': JPEGAttacker(quality=50),
                    'jpeg_attacker_90': JPEGAttacker(quality=10),
                    'jpeg_attacker_99': JPEGAttacker(quality=1),
                    'rotate_30': RotateAttacker(degree = 30),
                    'rotate_60': RotateAttacker(degree = 60),
                    'rotate_90': RotateAttacker(degree=90),
                    'rotate_120': RotateAttacker(degree = 120),
                    'rotate_150': RotateAttacker(degree = 150),
                    'rotate_180': RotateAttacker(degree = 180),
                    'rotate_210': RotateAttacker(degree = 210),
                    'rotate_240': RotateAttacker(degree = 240),
                    'rotate_270': RotateAttacker(degree = 270),
                    'lateral_rotate': LateralRotateAttacker(),
                    'brightness_0.2': BrightnessAttacker(brightness=0.2),
                    'brightness_0.4': BrightnessAttacker(brightness=0.4),
                    'brightness_0.5': BrightnessAttacker(brightness=0.5),
                    'brightness_0.6': BrightnessAttacker(brightness=0.6),
                    'brightness_0.8': BrightnessAttacker(brightness=0.8),
                    'brightness_1.25': BrightnessAttacker(brightness=1.25),
                    'brightness_1.5': BrightnessAttacker(brightness=1.5),
                    'contrast_0.2': ContrastAttacker(contrast=0.2),
                    'contrast_0.4': ContrastAttacker(contrast=0.4),
                    'contrast_0.5': ContrastAttacker(contrast=0.5),
                    'contrast_0.6': ContrastAttacker(contrast=0.6),
                    'contrast_0.8': ContrastAttacker(contrast=0.8),
                    'contrast_1.25': ContrastAttacker(contrast=1.25),
                    'contrast_1.5': ContrastAttacker(contrast=1.5),
                    'vibrancy_1.25': VibrancyAttacker(vibrancy = 1.25),
                    'black_white': BlackAndWhiteAttacker(),
                    'lateral_inversion':LateralInversionAttacker(),
                    'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
                    'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
                    'Motion_blur': MotionBlurAttacker(kernel_size=15, angle=45),
                    'OutofFocus_blur':OutOfFocusBlurAttacker(kernel_size = 15),
                    'Radial_blur': RadialBlurAttacker(strength = 10),
                    'Zoom_blur':ZoomBlurAttacker(strength=20),
                    'Atmospheric_blur': AtmosphericBlurAttacker(strength=0.5),
                    'PSF_blur': PSFBlurAttacker(kernel_size=15, angle=45),
                    'Bilateral_blur': BilateralFilterAttacker(d=9, sigmaColor=75, sigmaSpace=75),
                    'Iterative_blur': IterativeBlurAttacker(iterations =3),
                    'AnisotropicDiffusion_blur':AnisotropicDiffusionAttacker(num_iter=15, delta_t=0.14, kappa=50),
                    'DirectionalGaussian_blur': DirectionalGaussianBlurAttacker(kernel_size=15, sigma=5, angle=45),
                    'sharpening': SharpeningAttacker(factor = 2.0),
                    'salt_pepper_noise':SaltAndPepperNoiseAttacker(amount = 0.1),
                    'hue_change_0.1': HueChangeAttacker(factor = 0.1),
                    'hue_change_0.3': HueChangeAttacker(factor = 0.3),
                    'hue_change_0.5': HueChangeAttacker(factor = 0.5),
                    'hue_change_0.7': HueChangeAttacker(factor = 0.7),
                    'elastic_deformation': ElasticDeformationAttacker(alpha=1000, sigma=50),
                    'RGBtoHSV':RGBtoHSVAttacker(h_shift=0.1, s_scale=1.2, v_scale=1.1),
                    'color_balance': ColorBalanceAttacker( r_scale=1.2, g_scale=1.0, b_scale=0.8),
                    'gamma':GammaAttacker(gamma = 1.5),
                    'HistogramEqualization':HistogramEqualizationAttacker(),
                    'log_transform': LogTransformAttacker(c=1),
                    'color_jitter':ColorJitterAttacker(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    'color_quantization':ColorQuantizationAttacker(n_colors = 32),
                    'sepia':SepiaAttacker(),
                    'posterization': PosterizationAttacker(levels =4),
                    'bm3d': BM3DAttacker(),
                    }
                multi_name = 'all'
            else:
                attackers = {
                    'Motion_blur': MotionBlurAttacker(kernel_size=15, angle=45),
                    'sepia':SepiaAttacker(),
                    'hue_change': HueChangeAttacker(factor = 0.3),
                }
                multi_name = 'all'
        elif case == 'w/o rot':
            if not args.adv_only:
                attackers = {
                    'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
                    'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
                    'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
                    'jpeg_attacker_50': JPEGAttacker(quality=50),
                    'jpeg_attacker_90': JPEGAttacker(quality=10),
                    'jpeg_attacker_99': JPEGAttacker(quality=1),
                    # 'rotate_30': RotateAttacker(degree = 30),
                    # 'rotate_60': RotateAttacker(degree = 60),
                    # 'rotate_90': RotateAttacker(degree=90),
                    # 'rotate_120': RotateAttacker(degree = 120),
                    # 'rotate_150': RotateAttacker(degree = 150),
                    # 'rotate_180': RotateAttacker(degree = 180),
                    # 'rotate_210': RotateAttacker(degree = 210),
                    # 'rotate_240': RotateAttacker(degree = 240),
                    # 'rotate_270': RotateAttacker(degree = 270),
                    'brightness_0.2': BrightnessAttacker(brightness=0.2),
                    'brightness_0.4': BrightnessAttacker(brightness=0.4),
                    'brightness_0.5': BrightnessAttacker(brightness=0.5),
                    'brightness_0.6': BrightnessAttacker(brightness=0.6),
                    'brightness_0.8': BrightnessAttacker(brightness=0.8),
                    'brightness_1.25': BrightnessAttacker(brightness=1.25),
                    'brightness_1.5': BrightnessAttacker(brightness=1.5),
                    'contrast_0.2': ContrastAttacker(contrast=0.2),
                    'contrast_0.4': ContrastAttacker(contrast=0.4),
                    'contrast_0.5': ContrastAttacker(contrast=0.5),
                    'contrast_0.6': ContrastAttacker(contrast=0.6),
                    'contrast_0.8': ContrastAttacker(contrast=0.8),
                    'contrast_1.25': ContrastAttacker(contrast=1.25),
                    'contrast_1.5': ContrastAttacker(contrast=1.5),
                    'vibrancy_1.25': VibrancyAttacker(vibrancy = 1.25),
                    'black_white': BlackAndWhiteAttacker(),
                    'lateral_inversion':LateralInversionAttacker(),
                    'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
                    'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
                    'Motion_blur': MotionBlurAttacker(kernel_size=15, angle=45),
                    'OutofFocus_blur':OutOfFocusBlurAttacker(kernel_size = 15),
                    'Radial_blur': RadialBlurAttacker(strength = 10),
                    'Zoom_blur':ZoomBlurAttacker(strength=20),
                    'Atmospheric_blur': AtmosphericBlurAttacker(strength=0.5),
                    'PSF_blur': PSFBlurAttacker(kernel_size=15, angle=45),
                    'Bilateral_blur': BilateralFilterAttacker(d=9, sigmaColor=75, sigmaSpace=75),
                    'Iterative_blur': IterativeBlurAttacker(iterations =3),
                    'AnisotropicDiffusion_blur':AnisotropicDiffusionAttacker(num_iter=15, delta_t=0.14, kappa=50),
                    'DirectionalGaussian_blur': DirectionalGaussianBlurAttacker(kernel_size=15, sigma=5, angle=45),
                    'sharpening': SharpeningAttacker(factor = 2.0),
                    'salt_pepper_noise':SaltAndPepperNoiseAttacker(amount = 0.1),
                    'hue_change_0.1': HueChangeAttacker(factor = 0.1),
                    'hue_change_0.3': HueChangeAttacker(factor = 0.3),
                    'hue_change_0.5': HueChangeAttacker(factor = 0.5),
                    'hue_change_0.7': HueChangeAttacker(factor = 0.7),
                    'elastic_deformation': ElasticDeformationAttacker(alpha=1000, sigma=50),
                    'RGBtoHSV':RGBtoHSVAttacker(h_shift=0.1, s_scale=1.2, v_scale=1.1),
                    'color_balance': ColorBalanceAttacker( r_scale=1.2, g_scale=1.0, b_scale=0.8),
                    'gamma':GammaAttacker(gamma = 1.5),
                    'HistogramEqualization':HistogramEqualizationAttacker(),
                    'log_transform': LogTransformAttacker(c=1),
                    'color_jitter':ColorJitterAttacker(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    'color_quantization':ColorQuantizationAttacker(n_colors = 32),
                    'sepia':SepiaAttacker(),
                    'posterization': PosterizationAttacker(levels =4),
                    'bm3d': BM3DAttacker(),
                    }
                multi_name = 'all_norot'
            else:
                attackers = {
                    'Motion_blur': MotionBlurAttacker(kernel_size=15, angle=45),
                    'sepia':SepiaAttacker(),
                    'hue_change': HueChangeAttacker(factor = 0.3),
                }
                multi_name = 'all_norot'   
        
        os.makedirs(os.path.join(wm_path, multi_name), exist_ok=True)
        att_img_path = os.path.join(wm_path, multi_name, os.path.basename(post_img))
        for i, (attacker_name, attacker) in enumerate(attackers.items()):
            print(f'Attacking with {attacker_name}')
            if i == 0:
                attackers[attacker_name].attack([post_img], [att_img_path], multi=True)
            else:
                attackers[attacker_name].attack([att_img_path], [att_img_path], multi=True)
    post_img = os.path.join(wm_path,f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold}.png")

    if not args.adv_only: 
        attackers = ['diff_attacker_60', 'cheng2020-anchor_3', 'bmshj2018-factorized_3', 'jpeg_attacker_50','jpeg_attacker_90','jpeg_attacker_99',
                     'brightness_0.2','brightness_0.4','brightness_0.5','brightness_0.6','brightness_0.8', 
                    'brightness_1.25','brightness_1.5' ,'contrast_0.2','contrast_0.4','contrast_0.5','contrast_0.6','contrast_0.8','contrast_1.25','contrast_1.5',
                    'vibrancy_1.25','black_white','lateral_inversion', 'Gaussian_noise', 
                    'Gaussian_blur','AnisotropicDiffusion_blur','DirectionalGaussian_blur','sharpening','salt_pepper_noise','hue_change_0.1','hue_change_0.3',
                    'hue_change_0.5','hue_change_0.7','elastic_deformation',
                    'RGBtoHSV','color_balance','gamma','HistogramEqualization','log_transform','color_jitter','color_quantization','sepia','posterization',
                    'rotate_30','rotate_60','rotate_90','rotate_120','rotate_150','rotate_180','rotate_210','rotate_240','rotate_270','lateral_rotate', 'bm3d','all', 'all_norot']
    else:
        attackers = [
            'Motion_blur',
            'sepia',
            'hue_change',
        ]
#CHANGE

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    det_prob = 1 - watermark_prob(post_img, pipe, wm_pipe, text_embeddings)
    metrics["normal"].append(det_prob)
    
    for attacker_name in attackers:
        if not os.path.exists(os.path.join(wm_path, attacker_name)):
            logging.info(f'Attacked images under {attacker_name} not exist.')
            continue
            
        det_prob = 1 - watermark_prob(os.path.join(wm_path, attacker_name, os.path.basename(post_img)), pipe, wm_pipe, text_embeddings)
        #CHANGE , APPEND IN METRICS HERE
        metrics[attacker_name].append(det_prob)
    device = torch.device("cuda")
    # if args.attack == True:
    #     def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    #         # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    #             x0 = y0 = size // 2
    #             x0 += x_offset
    #             y0 += y_offset
    #             y, x = np.ogrid[:size, :size]
    #             y = y[::-1]
    #             return ((x - x0)**2 + (y-y0)**2)<= r**2
    #     watermarking_mask = torch.zeros(torch.Size([1, 4, 64, 64]), dtype=torch.bool).to(device)
    #     watermarking_mask[:,3] = torch.tensor(circle_mask(torch.Size([1, 4, 64, 64])[-1])).to(device)
    #     watermarking_mask.shape
    #     watermarking_mask = watermarking_mask.to(device="cpu")

    #     # Load Stable Diffusion 2.1 VAE encoder
    #     vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    #     vae.eval()

    #     # Define a simple model that computes the Fourier Transform of an image
    #     class FourierModel(nn.Module):
    #         def __init__(self, watermarking_mask):
    #             super(FourierModel, self).__init__()
    #             self.vae = vae
    #             self.watermarking_mask = watermarking_mask
    #         def forward(self, x):
    #             latents = self.vae.encode(x).latent_dist.sample()
    #             fft = latents
    #             fft = torch.fft.fft2(latents, dim=(-2, -1))
    #             reversed_latents_w_fft = torch.fft.fftshift(fft)[self.watermarking_mask].flatten()
    #             reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    #             return reversed_latents_w_fft
            
    #     def load_image(image_path):
    #         transform = transforms.Compose([
    #             transforms.Resize((512, 512)),  # Resize to match VAE input size
    #             transforms.ToTensor()
    #         ])
    #         image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    #         return transform(image).unsqueeze(0).requires_grad_(True)  # Add batch dimension
        
    #     image = load_image(os.path.join(wm_path,f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold}.png"))
            
    #         # Initialize model
    #     model = FourierModel(watermarking_mask=watermarking_mask)
    #     target = model(image)

    #     # PGD attack targeting Fourier space disruptions
    #     def pgd_attack(model, image, epsilon=0.015, alpha=0.015, iterations=50):
    #         perturbed_image = image.clone().detach().requires_grad_(True)
            
    #         for _ in range(iterations):
    #             output = model(perturbed_image)
    #             loss = torch.norm(target - output , p=2)  # Maximizing Fourier disruption
    #             print(loss)
    #             loss.backward(retain_graph=True)
                
    #             with torch.no_grad():
    #                 perturbed_image.data += alpha * perturbed_image.grad.sign()
    #                 perturbed_image.data = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
    #                 perturbed_image.data = torch.clamp(perturbed_image, 0, 1)  
            
    #         return perturbed_image.detach()

    #     # Load an example image
    #     image = load_image(os.path.join(wm_path,f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold}.png"))

    #     # Initialize model
    #     model = FourierModel(watermarking_mask=watermarking_mask)

    #     # Perform adversarial attack
    #     perturbed_image = pgd_attack(model, image, epsilon=args.budget , alpha=args.alpha , iterations=args.iters)

    #     save_transform = transforms.ToPILImage()
    #     perturbed_image_pil = save_transform(perturbed_image.squeeze(0))
    #     os.makedirs(os.path.join(wm_path, 'perturbed_images'), exist_ok=True)
    #     perturbed_image_pil.save(os.path.join(wm_path, 'perturbed_images', f"{imagename}_perturbed.png"))
    #     det_prob = 1 - watermark_prob(os.path.join(wm_path, 'perturbed_images', f"{imagename}_perturbed.png"), pipe, wm_pipe, text_embeddings)
    #     metrics["adv"].append(det_prob)
    tatta += 1 
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(wm_path, 'metrics.csv'))
    logging.info(f'Image {imagename} done')
    logging.info(f'Image number {tatta+1}')

logging.info(f'===== Done =====')
