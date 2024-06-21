from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import rawpy
from skimage import exposure
from skimage.restoration import denoise_tv_chambolle

import deps.monodepth2.networks as networks
from deps.monodepth2.utils import download_model_if_doesnt_exist

from seathru import *

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

def read_image(image_path):
    try:
        image_file_raw = rawpy.imread(image_path).postprocess()
        image_file = Image.fromarray(image_file_raw)
    except rawpy.LibRawFileUnsupportedError:
        image_file = Image.open(image_path)
    return np.float64(image_file) / 255.0

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
    recovered = run_pipeline(np.array(img) / 255.0, depths, args)

    # Further processing to enhance image quality
    sigma_est = estimate_sigma(recovered, channel_axis=-1, average_sigmas=True) / 10.0
    recovered = denoise_tv_chambolle(recovered, weight=sigma_est, channel_axis=-1)

    # Optional additional enhancements
    recovered = exposure.equalize_adapthist(recovered)

    # Apply color correction to reduce blue tint
    recovered = correct_color(recovered)

    im = Image.fromarray((np.round(recovered * 255.0)).astype(np.uint8))
    im.save(args.output, format='png')
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input image')
    parser.add_argument('--depth_map', required=True, help='Input depth map')
    parser.add_argument('--output', default='output.png', help='Output filename')
    parser.add_argument('--f', type=float, default=1.0, help='f value (controls brightness)')
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
    args = parser.parse_args()
    run(args)
