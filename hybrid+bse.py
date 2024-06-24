from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as Image
import PIL.ImageEnhance as ImageEnhance
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import rawpy
from skimage import exposure
from skimage.restoration import denoise_tv_chambolle

import deps.monodepth2.networks as networks
from deps.monodepth2.utils import download_model_if_doesnt_exist

from seathru import *

# Set environment variable to handle OpenMP issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def correct_color(image):
    """
    Perform color correction to reduce blue tint.
    """
    avgR = np.mean(image[:, :, 0])
    avgG = np.mean(image[:, :, 1])
    avgB = np.mean(image[:, :, 2])

    avg = (avgR + avgG + avgB) / 3

    image[:, :, 0] = np.clip(image[:, :, 0] * (avg / avgR), 0, 1)
    image[:, :, 1] = np.clip(image[:, :, 1] * (avg / avgG), 0, 1)
    image[:, :, 2] = np.clip(image[:, :, 2] * (avg / avgB), 0, 1)

    return image

def enhance_brightness(image, factor=1.5):
    """
    Enhance the brightness of the image.
    :param image: Input image in numpy array format.
    :param factor: Factor to control brightness. >1 increases brightness, <1 decreases.
    :return: Brightness enhanced image.
    """
    pil_img = Image.fromarray((image * 255).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(pil_img)
    enhanced_img = enhancer.enhance(factor)
    return np.asarray(enhanced_img) / 255.0

def read_image(image_path):
    try:
        image_file_raw = rawpy.imread(image_path).postprocess()
        image_file = Image.fromarray(image_file_raw)
    except rawpy.LibRawFileUnsupportedError:
        image_file = Image.open(image_path)
    return np.float64(image_file) / 255.0

# Add Backscatter Estimation Functions

def predict_backscatter(z, veil, backscatter, recover, attenuation):
    return (veil * (1 - np.exp(-backscatter * z)) + recover * np.exp(-attenuation * z))

def find_reference_points(image, depths, frac=0.02):
    z_max = np.max(depths)
    z_min = np.min(depths)
    z_bins = np.linspace(z_min, z_max, 10 + 1)
    rgb_norm = np.linalg.norm(image, axis=2)
    ret = []
    for i in range(10):
        lo, hi = z_bins[i], z_bins[i + 1]
        indices = np.where(np.logical_and(depths >= lo, depths < hi))
        if indices[0].size == 0:
            continue
        bin_rgb_norm, bin_z, bin_color = rgb_norm[indices], depths[indices], image[indices]
        points_sorted = sorted(zip(bin_rgb_norm, bin_z, bin_color[:,0], bin_color[:,1], bin_color[:,2]), key=lambda p: p[0])
        for j in range(math.ceil(len(points_sorted) * frac)):
            ret.append(points_sorted[j])
    return np.asarray(ret)

def estimate_channel_backscatter(points, depths, channel, attempts=50):
    lo = np.array([0, 0, 0, 0])
    hi = np.array([1, 5, 1, 5])
    best_loss = np.inf
    best_coeffs = []
    for _ in range(attempts):
        try:
            popt, pcov = scipy.optimize.curve_fit(predict_backscatter, points[:, 1], points[:, channel + 2], 
                                                np.random.random(4) * (hi - lo) + lo, bounds=(lo, hi))
            cur_loss = np.mean(np.square(predict_backscatter(points[:, 1], *popt) - points[:, channel + 2]))
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_coeffs = popt
        except RuntimeError as re:
            print(re, file=sys.stderr)
    Bc_channel = predict_backscatter(depths, *best_coeffs)
    return Bc_channel, best_coeffs

def estimate_backscatter(image, depths):
    points = find_reference_points(image, depths)
    backscatter_channels = []
    backscatter_coeffs = []
    for channel in range(3):
        Bc, coeffs = estimate_channel_backscatter(points, depths, channel)
        backscatter_channels.append(Bc)
        backscatter_coeffs.append(coeffs)
    Ba = np.stack(backscatter_channels, axis=2)
    return Ba, backscatter_coeffs

def run(args):
    """Function to process image with provided depth map and merge with Monodepth2 prediction"""
    assert args.depth_map is not None, "You must specify the --depth_map parameter"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # Load the image
    img = Image.fromarray(rawpy.imread(args.image).postprocess()) if args.raw else Image.open(args.image).convert('RGB')
    original_width, original_height = img.size
    input_image = img.resize((feed_width, feed_height), Image.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    print('Preprocessed image', flush=True)

    # PREDICTION using Monodepth2
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Save the depth map from Monodepth2
    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    plt.imsave('monodepth2_depth_map.png', disp_resized_np, cmap='plasma')

    # Load provided depth map
    provided_depth_map = Image.open(args.depth_map)
    provided_depth_map = provided_depth_map.resize((original_width, original_height), Image.LANCZOS)
    provided_depth_map_np = np.array(provided_depth_map).astype(np.float32)
    provided_depth_map_np = (provided_depth_map_np - np.min(provided_depth_map_np)) / (np.max(provided_depth_map_np) - np.min(provided_depth_map_np))

    # Merge the depth maps
    combined_depth_map = (provided_depth_map_np + disp_resized_np) / 2.0
    plt.imsave('combined_depth_map.png', combined_depth_map, cmap='plasma')

    # Adjust depth map preprocessing
    mapped_im_depths = ((combined_depth_map - np.min(combined_depth_map)) / (np.max(combined_depth_map) - np.min(combined_depth_map))).astype(np.float32)
    print("Processed image", flush=True)
    print('Loading image...', flush=True)
    depths = preprocess_monodepth_depth_map(mapped_im_depths, args.monodepth_add_depth, args.monodepth_multiply_depth)
    original = np.array(img) / 255.0

    # Estimating backscatter
    print("Estimating backscatter...")
    Ba, coeffs = estimate_backscatter(original, depths)

    Da = original - Ba
    Da = np.clip(Da, 0, 1)

    # Further processing to enhance image quality
    sigma_est = estimate_sigma(Da, channel_axis=-1, average_sigmas=True) / 10.0
    Da = denoise_tv_chambolle(Da, weight=sigma_est, channel_axis=-1)

    # Optional additional enhancements
    Da = exposure.equalize_adapthist(Da)

    # Apply color correction to reduce blue tint
    Da = correct_color(Da)

    # Enhance the brightness of the image
    Da = enhance_brightness(Da, factor=args.brightness_factor)

    im = Image.fromarray((np.round(Da * 255.0)).astype(np.uint8))
    im.save(args.output, format='png')
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input image')
    parser.add_argument('--depth_map', required=True, help='Input depth map')
    parser.add_argument('--output', default='output.png', help='Output filename')
    parser.add_argument('--f', type=float, default=1.3, help='f value (controls brightness)')
    parser.add_argument('--l', type=float, default=0.5, help='l value (controls balance of attenuation constants)')
    parser.add_argument('--p', type=float, default=0.01, help='p value (controls locality of illuminant map)')
    parser.add_argument('--min-depth', type=float, default=0.0, help='Minimum depth value to use in estimations (range 0-1)')
    parser.add_argument('--max-depth', type=float, default=1.0, help='Replacement depth percentile value for invalid depths (range 0-1)')
    parser.add_argument('--spread-data-fraction', type=float, default=0.05, help='Require data to be this fraction of depth range away from each other in attenuation estimations')
    parser.add_argument('--size', type=int, default=320, help='Size to output')
    parser.add_argument('--monodepth-add-depth', type=float, default=2.0, help='Additive value for monodepth map')
    parser.add_argument('--monodepth-multiply-depth', type=float, default=10.0, help='Multiplicative value for monodepth map')
    parser.add_argument('--model-name', type=str, default="mono_1024x320", help='monodepth model name')
    parser.add_argument('--output-graphs', action='store_true', help='Output graphs')
    parser.add_argument('--raw', action='store_true', help='RAW image')
    parser.add_argument('--no_cuda', action='store_true', help='If set, disables CUDA')
    parser.add_argument('--brightness_factor', type=float, default=1.5, help='Brightness enhancement factor')
    args = parser.parse_args()
    run(args)
