from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2
import torch
import os
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy import signal
from skimage.util import random_noise
from skimage import exposure
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm.auto import tqdm
from bm3d import bm3d_rgb
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor


class WMAttacker:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError


class VAEWMAttacker(WMAttacker):
    def __init__(self, model_name, quality=1, metric='mse', device='cpu'):
        if model_name == 'bmshj2018-factorized':
            self.model = bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'bmshj2018-hyperprior':
            self.model = bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018-mean':
            self.model = mbt2018_mean(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018':
            self.model = mbt2018(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'cheng2020-anchor':
            self.model = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)
        else:
            raise ValueError('model name not supported')
        self.device = device

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize((512, 512))
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            out = self.model(img)
            out['x_hat'].clamp_(0, 1)
            rec = transforms.ToPILImage()(out['x_hat'].squeeze().cpu())
            rec.save(out_path)


class GaussianBlurAttacker(WMAttacker):
    def __init__(self, kernel_size=5, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
            cv2.imwrite(out_path, img)

class MotionBlurAttacker(WMAttacker):
    def __init__(self, kernel_size=15, angle=45):
        self.kernel_size = kernel_size
        self.angle = angle

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            
            # Create the motion blur kernel
            kernel = np.zeros((self.kernel_size, self.kernel_size))
            center = (self.kernel_size - 1) // 2
            angle_rad = np.deg2rad(self.angle)
            x = int(round(center + np.cos(angle_rad) * center))
            y = int(round(center + np.sin(angle_rad) * center))
            kernel[center, center] = 1
            kernel[y, x] = 1
            kernel = cv2.normalize(kernel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            # Apply the motion blur
            blurred = cv2.filter2D(img, -1, kernel)
            
            cv2.imwrite(out_path, blurred)

class OutOfFocusBlurAttacker(WMAttacker):
    def __init__(self, kernel_size=15):
        self.kernel_size = kernel_size

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            
            # Create the out-of-focus blur kernel
            kernel = np.zeros((self.kernel_size, self.kernel_size))
            center = (self.kernel_size - 1) // 2
            y, x = np.ogrid[-center:center+1, -center:center+1]
            mask = x*x + y*y <= center*center
            kernel[mask] = 1
            kernel = kernel / np.sum(kernel)
            
            # Apply the out-of-focus blur
            blurred = cv2.filter2D(img, -1, kernel)
            
            cv2.imwrite(out_path, blurred)

class RadialBlurAttacker(WMAttacker):
    def __init__(self, strength=10):
        self.strength = strength

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            
            # Create the radial blur effect
            for i in range(self.strength):
                weight = 1.0 - (i / self.strength)
                size = i * 2 + 1
                blurred = cv2.GaussianBlur(img, (size, size), 0)
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.circle(mask, center, i, (weight), -1)
                img = img * (1 - mask[:,:,np.newaxis]) + blurred * mask[:,:,np.newaxis]
            
            cv2.imwrite(out_path, img.astype(np.uint8))

class ZoomBlurAttacker(WMAttacker):
    def __init__(self, strength=20):
        self.strength = strength

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            
            # Create the zoom blur effect
            result = np.zeros_like(img, dtype=np.float32)
            for i in range(self.strength):
                scale = 1 + (i / self.strength) * 0.2
                scaled = cv2.resize(img, None, fx=scale, fy=scale)
                sh, sw = scaled.shape[:2]
                y_start, x_start = max(0, (sh - h) // 2), max(0, (sw - w) // 2)
                y_end, x_end = y_start + h, x_start + w
                crop = scaled[y_start:y_end, x_start:x_end]
                result += crop
            
            result /= self.strength
            cv2.imwrite(out_path, result.astype(np.uint8))

class AtmosphericBlurAttacker(WMAttacker):
    def __init__(self, strength=0.5):
        self.strength = strength

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            
            # Create atmospheric blur effect
            kernel_size = int(min(img.shape[:2]) * self.strength)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            result = cv2.addWeighted(img, 1 - self.strength, blurred, self.strength, 0)
            
            cv2.imwrite(out_path, result)

class PSFBlurAttacker(WMAttacker):
    def __init__(self, kernel_size=15, angle=45):
        self.kernel_size = kernel_size
        self.angle = angle

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            
            # Create PSF kernel
            kernel = np.zeros((self.kernel_size, self.kernel_size))
            center = self.kernel_size // 2
            angle_rad = np.deg2rad(self.angle)
            dx = int(center * np.cos(angle_rad))
            dy = int(center * np.sin(angle_rad))
            kernel[center, center] = 1
            kernel[center + dy, center + dx] = 1
            kernel = kernel / np.sum(kernel)
            
            # Apply PSF blur
            blurred = np.zeros_like(img, dtype=np.float32)
            for i in range(3):  # Apply to each color channel
                blurred[:,:,i] = signal.convolve2d(img[:,:,i], kernel, mode='same', boundary='wrap')
            
            cv2.imwrite(out_path, blurred.astype(np.uint8))

class BilateralFilterAttacker(WMAttacker):
    def __init__(self, d=9, sigmaColor=75, sigmaSpace=75):
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            
            # Apply bilateral filter
            filtered = cv2.bilateralFilter(img, self.d, self.sigmaColor, self.sigmaSpace)
            
            cv2.imwrite(out_path, filtered)

class IterativeBlurAttacker(WMAttacker):
    def __init__(self, iterations=3):
        self.iterations = iterations

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            
            # Apply iterative blurring
            for _ in range(self.iterations):
                img = cv2.GaussianBlur(img, (5, 5), 0)
                img = cv2.medianBlur(img, 5)
            
            cv2.imwrite(out_path, img)

class AnisotropicDiffusionAttacker(WMAttacker):
    def __init__(self, num_iter=15, delta_t=0.14, kappa=50):
        self.num_iter = num_iter
        self.delta_t = delta_t
        self.kappa = kappa

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path, 0).astype(np.float32)
            
            for _ in range(self.num_iter):
                gradient_N = np.roll(img, -1, axis=0) - img
                gradient_S = np.roll(img, 1, axis=0) - img
                gradient_E = np.roll(img, -1, axis=1) - img
                gradient_W = np.roll(img, 1, axis=1) - img
                
                cN = np.exp(-(gradient_N/self.kappa)**2)
                cS = np.exp(-(gradient_S/self.kappa)**2)
                cE = np.exp(-(gradient_E/self.kappa)**2)
                cW = np.exp(-(gradient_W/self.kappa)**2)
                
                img += self.delta_t * (cN*gradient_N + cS*gradient_S + cE*gradient_E + cW*gradient_W)
            
            cv2.imwrite(out_path, img.astype(np.uint8))

class DirectionalGaussianBlurAttacker(WMAttacker):
    def __init__(self, kernel_size=15, sigma=5, angle=45):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.angle = angle

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            
            kernel = np.zeros((self.kernel_size, self.kernel_size))
            center = self.kernel_size // 2
            angle_rad = np.deg2rad(self.angle)
            
            for x in range(self.kernel_size):
                for y in range(self.kernel_size):
                    rotated_x = (x - center) * np.cos(angle_rad) - (y - center) * np.sin(angle_rad)
                    kernel[y, x] = np.exp(-rotated_x**2 / (2 * self.sigma**2))
            
            kernel /= np.sum(kernel)
            blurred = cv2.filter2D(img, -1, kernel)
            
            cv2.imwrite(out_path, blurred)

class GaussianNoiseAttacker(WMAttacker):
    def __init__(self, std=0.05):
        self.std = std

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            image = cv2.imread(img_path)
            image = image / 255.0
            # Add Gaussian noise to the image
            noise_sigma = self.std  # Vary this to change the amount of noise
            noisy_image = random_noise(image, mode='gaussian', var=noise_sigma ** 2)
            # Clip the values to [0, 1] range after adding the noise
            noisy_image = np.clip(noisy_image, 0, 1)
            noisy_image = np.array(255 * noisy_image, dtype='uint8')
            cv2.imwrite(out_path, noisy_image)


class BM3DAttacker(WMAttacker):
    def __init__(self):
        pass

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path).convert('RGB')
            y_est = bm3d_rgb(np.array(img) / 255, 0.1)  # use standard deviation as 0.1, 0.05 also works
            plt.imsave(out_path, np.clip(y_est, 0, 1), cmap='gray', vmin=0, vmax=1)


class JPEGAttacker(WMAttacker):
    def __init__(self, quality=80):
        self.quality = quality

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            img.save(out_path, "JPEG", quality=self.quality)


class BrightnessAttacker(WMAttacker):
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.brightness)
            img.save(out_path)

class BlackAndWhiteAttacker(WMAttacker):
    def __init__(self):
        pass

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            img = img.convert('L')
            img.save(out_path)


class ContrastAttacker(WMAttacker):
    def __init__(self, contrast=0.2):
        self.contrast = contrast

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast)
            img.save(out_path)

class VibrancyAttacker(WMAttacker):
    def __init__(self, vibrancy=0.2):
        self.vibrancy = vibrancy

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(self.vibrancy)
            img.save(out_path)


class RotateAttacker(WMAttacker):
    def __init__(self, degree=30, expand=1):
        self.degree = degree
        self.expand = expand

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            img = img.rotate(self.degree, expand=self.expand)
            img = img.resize((512,512))
            img.save(out_path)


class LateralInversionAttacker(WMAttacker):
    def __init__(self):
        pass

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(out_path)

class LateralRotateAttacker(WMAttacker):
    def __init__(self):
        pass

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.rotate(180)
            img.save(out_path)


class ScaleAttacker(WMAttacker):
    def __init__(self, scale=0.5):
        self.scale = scale

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            w, h = img.size
            img = img.resize((int(w * self.scale), int(h * self.scale)))
            img.save(out_path)


class CropAttacker(WMAttacker):
    def __init__(self, crop_size=0.5):
        self.crop_size = crop_size

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            w, h = img.size
            img = img.crop((int(w * self.crop_size), int(h * self.crop_size), w, h))
            img.save(out_path)


class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, batch_size=20, noise_step=60, captions={}):
        self.pipe = pipe
        self.BATCH_SIZE = batch_size
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

    def attack(self, image_paths, out_paths, return_latents=False, return_dist=False, multi=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(prompts_buf,
                                   head_start_latents=latents,
                                   head_start_step=50 - max(self.noise_step // 20, 1),
                                   guidance_scale=7.5,
                                   generator=generator, )
                images = images[0]
                for img, out in zip(images, outs_buf):
                    img.save(out)

            if len(self.captions) != 0:
                prompts = []
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    if img_name[:-4] in self.captions:
                        prompts.append(self.captions[img_name[:-4]])
                    else:
                        prompts.append("")
            else:
                prompts = [""] * len(image_paths)

            for (img_path, out_path), prompt in tqdm(zip(zip(image_paths, out_paths), prompts)):
                if os.path.exists(out_path) and not multi:
                    continue
                
                img = Image.open(img_path)
                img = np.asarray(img) / 255
                img = (img - 0.5) * 2
                img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
                latents = self.pipe.vae.encode(img).latent_dist
                latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
                noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
                if return_dist:
                    return self.pipe.scheduler.add_noise(latents, noise, timestep, return_dist=True)
                latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)
                latents_buf.append(latents)
                outs_buf.append(out_path)
                prompts_buf.append(prompt)
                if len(latents_buf) == self.BATCH_SIZE:
                    batched_attack(latents_buf, prompts_buf, outs_buf)
                    latents_buf = []
                    prompts_buf = []
                    outs_buf = []
                if return_latents:
                    ret_latents.append(latents.cpu())

            if len(latents_buf) != 0:
                batched_attack(latents_buf, prompts_buf, outs_buf)
            if return_latents:
                return ret_latents

class SharpeningAttacker(WMAttacker):
    def __init__(self, factor=2.0):
        self.factor = factor

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.factor)
            img.save(out_path)

class SaltAndPepperNoiseAttacker(WMAttacker):
    def __init__(self, amount=0.05):
        self.amount = amount

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Salt noise
            num_salt = np.ceil(self.amount * img_array.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
            img_array[coords[0], coords[1], :] = 255

            # Pepper noise
            num_pepper = np.ceil(self.amount * img_array.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
            img_array[coords[0], coords[1], :] = 0
            
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(out_path)

class HueChangeAttacker(WMAttacker):
    def __init__(self, factor=0.5):
        self.factor = factor

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path).convert('HSV')
            h, s, v = img.split()
            h = h.point(lambda i: (i + int(self.factor * 255)) % 255)
            img = Image.merge('HSV', (h, s, v)).convert('RGB')
            img.save(out_path)

class ElasticDeformationAttacker(WMAttacker):
    def __init__(self, alpha=1000, sigma=50):
        self.alpha = alpha
        self.sigma = sigma

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            img_array = np.array(img)
            
            shape = img_array.shape
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            
            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
            
            distorted_image = map_coordinates(img_array, indices, order=1, mode='reflect')
            distorted_image = distorted_image.reshape(img_array.shape)
            
            img = Image.fromarray(distorted_image.astype(np.uint8))
            img.save(out_path)

class RGBtoHSVAttacker(WMAttacker):
    def __init__(self, h_shift=0.1, s_scale=1.2, v_scale=1.1):
        self.h_shift = h_shift
        self.s_scale = s_scale
        self.v_scale = v_scale

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,0] = (hsv[:,:,0] + self.h_shift * 180) % 180
            hsv[:,:,1] = np.clip(hsv[:,:,1] * self.s_scale, 0, 255)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * self.v_scale, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            cv2.imwrite(out_path, img)

class ColorBalanceAttacker(WMAttacker):
    def __init__(self, r_scale=1.2, g_scale=1.0, b_scale=0.8):
        self.r_scale = r_scale
        self.g_scale = g_scale
        self.b_scale = b_scale

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            img = img.astype(np.float32)
            img[:,:,0] = np.clip(img[:,:,0] * self.b_scale, 0, 255)
            img[:,:,1] = np.clip(img[:,:,1] * self.g_scale, 0, 255)
            img[:,:,2] = np.clip(img[:,:,2] * self.r_scale, 0, 255)
            cv2.imwrite(out_path, img.astype(np.uint8))

class GammaAttacker(WMAttacker):
    def __init__(self, gamma=1.5):
        self.gamma = gamma

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            img = np.power(img / 255.0, self.gamma)
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(out_path, img)

class HistogramEqualizationAttacker(WMAttacker):
    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            cv2.imwrite(out_path, img)

class LogTransformAttacker(WMAttacker):
    def __init__(self, c=1):
        self.c = c

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            img = np.float32(img)
            img = self.c * np.log1p(img)
            img = np.uint8(255 * img / np.max(img))
            cv2.imwrite(out_path, img)

class ColorJitterAttacker(WMAttacker):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            img = ImageEnhance.Brightness(img).enhance(1 + np.random.uniform(-self.brightness, self.brightness))
            img = ImageEnhance.Contrast(img).enhance(1 + np.random.uniform(-self.contrast, self.contrast))
            img = ImageEnhance.Color(img).enhance(1 + np.random.uniform(-self.saturation, self.saturation))
            img = img.convert('HSV')
            h, s, v = img.split()
            h = h.point(lambda i: (i + int(np.random.uniform(-self.hue, self.hue) * 255)) % 255)
            img = Image.merge('HSV', (h, s, v)).convert('RGB')
            img.save(out_path)

class ColorQuantizationAttacker(WMAttacker):
    def __init__(self, n_colors=32):
        self.n_colors = n_colors

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            Z = img.reshape((-1,3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, label, center = cv2.kmeans(Z, self.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((img.shape))
            cv2.imwrite(out_path, res2)

class SepiaAttacker(WMAttacker):
    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
            img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                            [0.349, 0.686, 0.168],
                                                            [0.393, 0.769, 0.189]]))
            img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
            img_sepia = np.array(img_sepia, dtype=np.uint8)
            cv2.imwrite(out_path, img_sepia)

class PosterizationAttacker(WMAttacker):
    def __init__(self, levels=4):
        self.levels = levels

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path)
            img = ImageOps.posterize(img, self.levels)
            img.save(out_path)
