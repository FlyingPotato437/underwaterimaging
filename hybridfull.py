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
from skimage.restoration import denoise_tv_chambolle, estimate_sigma
from skimage.morphology import closing, disk

import deps.monodepth2.networks as networks
from deps.monodepth2.utils import download_model_if_doesnt_exist

import collections
import scipy as sp
import scipy.optimize
import scipy.stats
import math
from skimage.restoration import denoise_bilateral

from seathru import *

def correct_color(image):

    avgR = np.mean(image[:, :, 0])
    avgG = np.mean(image[:, :, 1])
    avgB = np.mean(image[:, :, 2])

    avg = (avgR + avgG + avgB) / 3

    image[:, :, 0] = np.clip(image[:, :, 0] * (avg / avgR) * 1.2, 0, 1)
    image[:, :, 1] = np.clip(image[:, :, 1] * (avg / avgG) * 1.1, 0, 1)
    image[:, :, 2] = np.clip(image[:, :, 2] * (avg / avgB), 0, 1)

    return image

def read_image(image_path):
    try:
        image_file_raw = rawpy.imread(image_path).postprocess()
        image_file = Image.fromarray(image_file_raw)
    except rawpy.LibRawFileUnsupportedError:
        image_file = Image.open(image_path)
    return np.float64(image_file) / 255.0

def find_backscatter_estimation_points(img, depths, num_bins=10, fraction=0.01, max_vals=20, min_depth_percent=0.0):
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_percent * (z_max - z_min))
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.mean(img, axis=2)
    points_r, points_g, points_b = [], [], []
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i + 1]
        locs = np.where(np.logical_and(depths > min_depth, np.logical_and(depths >= a, depths <= b)))
        norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
        arr = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
        points = arr[:min(math.ceil(fraction * len(arr)), max_vals)]
        points_r.extend([(z, p[0]) for n, p, z in points])
        points_g.extend([(z, p[1]) for n, p, z in points])
        points_b.extend([(z, p[2]) for n, p, z in points])
    return np.array(points_r), np.array(points_g), np.array(points_b)

def find_backscatter_values(B_pts, depths, restarts=10, max_mean_loss_fraction=0.1):
    B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
    z_max, z_min = np.max(depths), np.min(depths)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf
    def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime):
        return (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
    def loss(B_inf, beta_B, J_prime, beta_D_prime):
        return np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))
    bounds_lower = [0, 0, 0, 0]
    bounds_upper = [1, 5, 1, 5]
    for _ in range(restarts):
        try:
            optp, pcov = sp.optimize.curve_fit(
                f=estimate,
                xdata=B_depths,
                ydata=B_vals,
                p0=np.random.random(4) * bounds_upper,
                bounds=(bounds_lower, bounds_upper),
            )
            l = loss(*optp)
            if l < best_loss:
                best_loss = l
                coefs = optp
        except RuntimeError as re:
            print(re, file=sys.stderr)
    if best_loss > max_mean_loss:
        print('could not find accurate reconstruction, switching to linear model.', flush=True)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(B_depths, B_vals)
        BD = (slope * depths) + intercept
        return BD, np.array([slope, intercept])
    return estimate(depths, *coefs), coefs

def estimate_illumination(img, B, neighborhood_map, num_neighborhoods, p=0.5, f=2.0, max_iters=100, tol=1E-5):
    D = img - B
    avg_cs = np.zeros_like(img)
    avg_cs_prime = np.copy(avg_cs)
    sizes = np.zeros(num_neighborhoods)
    locs_list = [None] * num_neighborhoods
    for label in range(1, num_neighborhoods + 1):
        locs_list[label - 1] = np.where(neighborhood_map == label)
        sizes[label - 1] = np.size(locs_list[label - 1][0])
    for _ in range(max_iters):
        for label in range(1, num_neighborhoods + 1):
            locs = locs_list[label - 1]
            size = sizes[label - 1] - 1
            avg_cs_prime[locs] = (1 / size) * (np.sum(avg_cs[locs]) - avg_cs[locs])
        new_avg_cs = (D * p) + (avg_cs_prime * (1 - p))
        if(np.max(np.abs(avg_cs - new_avg_cs)) < tol):
            break
        avg_cs = new_avg_cs
    return f * denoise_bilateral(np.maximum(0, avg_cs))

def estimate_wideband_attentuation(depths, illum, radius = 6, max_val = 10.0):
    eps = 1E-8
    BD = np.minimum(max_val, -np.log(illum + eps) / (np.maximum(0, depths) + eps))
    mask = np.where(np.logical_and(depths > eps, illum > eps), 1, 0)
    refined_attenuations = denoise_bilateral(closing(np.maximum(0, BD * mask), disk(radius)))
    return refined_attenuations, []

def calculate_beta_D(depths, a, b, c, d):
    return (a * np.exp(b * depths)) + (c * np.exp(d * depths))

def filter_data(X, Y, radius_fraction=0.01):
    idxs = np.argsort(X)
    X_s = X[idxs]
    Y_s = Y[idxs]
    x_max, x_min = np.max(X), np.min(X)
    radius = (radius_fraction * (x_max - x_min))
    ds = np.cumsum(X_s - np.roll(X_s, (1,)))
    dX = [X_s[0]]
    dY = [Y_s[0]]
    tempX = []
    tempY = []
    pos = 0
    for i in range(1, ds.shape[0]):
        if ds[i] - ds[pos] >= radius:
            tempX.append(X_s[i])
            tempY.append(Y_s[i])
            idxs = np.argsort(tempY)
            med_idx = len(idxs) // 2
            dX.append(tempX[med_idx])
            dY.append(tempY[med_idx])
            pos = i
        else:
            tempX.append(X_s[i])
            tempY.append(Y_s[i])
    return np.array(dX), np.array(dY)

def refine_wideband_attentuation(depths, illum, estimation, restarts=10, min_depth_fraction = 0.1, max_mean_loss_fraction=np.inf, l=1.0, radius_fraction=0.01):
    eps = 1E-8
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_fraction * (z_max - z_min))
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf
    locs = np.where(np.logical_and(illum > 0, np.logical_and(depths > min_depth, estimation > eps)))
    def calculate_reconstructed_depths(depths, illum, a, b, c, d):
        eps = 1E-5
        res = -np.log(illum + eps) / (calculate_beta_D(depths, a, b, c, d) + eps)
        return res
    def loss(a, b, c, d):
        return np.mean(np.abs(depths[locs] - calculate_reconstructed_depths(depths[locs], illum[locs], a, b, c, d)))
    dX, dY = filter_data(depths[locs], estimation[locs], radius_fraction)
    for _ in range(restarts):
        try:
            optp, pcov = sp.optimize.curve_fit(
                f=calculate_beta_D,
                xdata=dX,
                ydata=dY,
                p0=np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.]),
                bounds=([0, -100, 0, -100], [100, 0, 100, 0]))
            L = loss(*optp)
            if L < best_loss:
                best_loss = L
                coefs = optp
        except RuntimeError as re:
            print(re, file=sys.stderr)
    if best_loss > max_mean_loss:
        print('could not find accurate reconstruction, switching to linear model.', flush=True)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(depths[locs], estimation[locs])
        BD = (slope * depths + intercept)
        return l * BD, np.array([slope, intercept])
    print(f'best loss {best_loss}', flush=True)
    BD = l * calculate_beta_D(depths, *coefs)
    return BD, coefs

def recover_image(img, depths, B, beta_D, nmap):
    res = (img - B) * np.exp(beta_D * np.expand_dims(depths, axis=2))
    res = np.maximum(0.0, np.minimum(1.0, res))
    res[nmap == 0] = 0
    res = scale(wbalance_no_red_10p(res))
    res[nmap == 0] = img[nmap == 0]
    return res

def construct_neighborhood_map(depths, epsilon=0.05):
    eps = (np.max(depths) - np.min(depths)) * epsilon
    nmap = np.zeros_like(depths).astype(np.int32)
    n_neighborhoods = 1
    while np.any(nmap == 0):
        locs_x, locs_y = np.where(nmap == 0)
        start_index = np.random.randint(0, len(locs_x))
        start_x, start_y = locs_x[start_index], locs_y[start_index]
        q = collections.deque()
        q.append((start_x, start_y))
        while not len(q) == 0:
            x, y = q.pop()
            if np.abs(depths[x, y] - depths[start_x, start_y]) <= eps:
                nmap[x, y] = n_neighborhoods
                if 0 <= x < depths.shape[0] - 1:
                    x2, y2 = x + 1, y
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 1 <= x < depths.shape[0]:
                    x2, y2 = x - 1, y
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 0 <= y < depths.shape[1] - 1:
                    x2, y2 = x, y + 1
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 1 <= y < depths.shape[1]:
                    x2, y2 = x, y - 1
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
        n_neighborhoods += 1
    zeros_size_arr = sorted(zip(*np.unique(nmap[depths == 0], return_counts=True)), key=lambda x: x[1], reverse=True)
    if len(zeros_size_arr) > 0:
        nmap[nmap == zeros_size_arr[0][0]] = 0 
    return nmap, n_neighborhoods - 1

def refine_neighborhood_map(nmap, min_size = 10, radius = 3):
    refined_nmap = np.zeros_like(nmap)
    vals, counts = np.unique(nmap, return_counts=True)
    neighborhood_sizes = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)
    num_labels = 1
    for label, size in neighborhood_sizes:
        if size >= min_size and label != 0:
            refined_nmap[nmap == label] = num_labels
            num_labels += 1
    for label, size in neighborhood_sizes:
        if size < min_size and label != 0:
            for x, y in zip(*np.where(nmap == label)):
                refined_nmap[x, y] = find_closest_label(refined_nmap, x, y)
    refined_nmap = closing(refined_nmap, disk(radius))
    return refined_nmap, num_labels - 1

def find_closest_label(nmap, start_x, start_y):
    mask = np.zeros_like(nmap).astype(np.bool_)
    q = collections.deque()
    q.append((start_x, start_y))
    while not len(q) == 0:
        x, y = q.pop()
        if 0 <= x < nmap.shape[0] and 0 <= y < nmap.shape[1]:
            if nmap[x, y] != 0:
                return nmap[x, y]
            mask[x, y] = True
            if 0 <= x < nmap.shape[0] - 1:
                x2, y2 = x + 1, y
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 1 <= x < nmap.shape[0]:
                x2, y2 = x - 1, y
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 0 <= y < nmap.shape[1] - 1:
                x2, y2 = x, y + 1
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 1 <= y < nmap.shape[1]:
                x2, y2 = x, y - 1
                if not mask[x2, y2]:
                    q.append((x2, y2))

def run(args):
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

   
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

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

    img = Image.fromarray(rawpy.imread(args.image).postprocess()) if args.raw else Image.open(args.image).convert('RGB')
    original_width, original_height = img.size
    input_image = img.resize((feed_width, feed_height), Image.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    print('Preprocessed image', flush=True)

    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    plt.imsave('monodepth2_depth_map.png', disp_resized_np, cmap='plasma')

    provided_depth_map = Image.open(args.depth_map)
    provided_depth_map = provided_depth_map.resize((original_width, original_height), Image.LANCZOS)
    provided_depth_map_np = np.array(provided_depth_map).astype(np.float32)
    provided_depth_map_np = (provided_depth_map_np - np.min(provided_depth_map_np)) / (np.max(provided_depth_map_np) - np.min(provided_depth_map_np))

    combined_depth_map = (provided_depth_map_np + disp_resized_np) / 2.0
    plt.imsave('combined_depth_map.png', combined_depth_map, cmap='plasma')

    mapped_im_depths = ((combined_depth_map - np.min(combined_depth_map)) / (np.max(combined_depth_map) - np.min(combined_depth_map))).astype(np.float32)
    print("Processed image", flush=True)
    print('Loading image...', flush=True)
    depths = preprocess_monodepth_depth_map(mapped_im_depths, args.monodepth_add_depth, args.monodepth_multiply_depth)
    recovered = run_pipeline(np.array(img) / 255.0, depths, args)

    sigma_est = estimate_sigma(recovered, channel_axis=-1, average_sigmas=True) / 10.0
    recovered = denoise_tv_chambolle(recovered, weight=sigma_est, channel_axis=-1)

    recovered = exposure.equalize_adapthist(recovered)

    recovered = correct_color(recovered)

    im = Image.fromarray((np.round(recovered * 255.0)).astype(np.uint8))
    im.save(args.output, format='png')
    print('Done.')

def run_pipeline(img, depths, args):
    print('Estimating backscatter...', flush=True)
    ptsR, ptsG, ptsB = find_backscatter_estimation_points(img, depths, fraction=0.01, min_depth_percent=args.min_depth)

    print('Finding backscatter coefficients...', flush=True)
    Br, coefsR = find_backscatter_values(ptsR, depths, restarts=25)
    Bg, coefsG = find_backscatter_values(ptsG, depths, restarts=25)
    Bb, coefsB = find_backscatter_values(ptsB, depths, restarts=25)

    print('Constructing neighborhood map...', flush=True)
    nmap, _ = construct_neighborhood_map(depths, 0.1)

    print('Refining neighborhood map...', flush=True)
    nmap, n = refine_neighborhood_map(nmap, 50)

    print('Estimating illumination...', flush=True)
    illR = estimate_illumination(img[:, :, 0], Br, nmap, n, p=args.p, max_iters=100, tol=1E-5, f=args.f)
    illG = estimate_illumination(img[:, :, 1], Bg, nmap, n, p=args.p, max_iters=100, tol=1E-5, f=args.f)
    illB = estimate_illumination(img[:, :, 2], Bb, nmap, n, p=args.p, max_iters=100, tol=1E-5, f=args.f)
    ill = np.stack([illR, illG, illB], axis=2)

    print('Estimating wideband attenuation...', flush=True)
    beta_D_r, _ = estimate_wideband_attentuation(depths, illR)
    refined_beta_D_r, coefsR = refine_wideband_attentuation(depths, illR, beta_D_r, radius_fraction=args.spread_data_fraction, l=args.l)
    beta_D_g, _ = estimate_wideband_attentuation(depths, illG)
    refined_beta_D_g, coefsG = refine_wideband_attentuation(depths, illG, beta_D_g, radius_fraction=args.spread_data_fraction, l=args.l)
    beta_D_b, _ = estimate_wideband_attentuation(depths, illB)
    refined_beta_D_b, coefsB = refine_wideband_attentuation(depths, illB, beta_D_b, radius_fraction=args.spread_data_fraction, l=args.l)

    print('Reconstructing image...', flush=True)
    B = np.stack([Br, Bg, Bb], axis=2)
    beta_D = np.stack([refined_beta_D_r, refined_beta_D_g, refined_beta_D_b], axis=2)
    recovered = recover_image(img, depths, B, beta_D, nmap)

    return recovered

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
    args = parser.parse_args()
    run(args)

