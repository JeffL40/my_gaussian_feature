#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, feat_decoder
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import torch.nn.functional as F
import numpy as np
import pdb
from scene.cameras import Camera, MiniCam
from train import pad_signal_coefficients, reshape_features, ifft_features
import imageio

global global_args


# coefficients of shape ... x 4k(real(x | y) | im(x | y)), return of shape ... x signal_length x 2(x,y)
def reconstruct_signal(coefficients, signal_length):
    len_xy = coefficients.shape[-1] // 2
    len_x = len_xy // 2
    real_xy = coefficients[..., :len_xy]
    imag_xy = coefficients[..., len_xy:]
    real_x = real_xy[..., :len_x]
    real_y = real_xy[..., len_x:]
    imag_x = imag_xy[..., :len_x]
    imag_y = imag_xy[..., len_x:]
    complex_x = real_x + imag_x
    complex_y = real_y + imag_y
    x_signal = torch.fft.ifft(
        pad_signal_coefficients(complex_x, signal_length)
    )  # ... x signal_length
    y_signal = torch.fft.ifft(pad_signal_coefficients(complex_y, signal_length))
    return torch.stack([x_signal, y_signal], dim=-1)


def declare_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def torch2numpy(a):
    return a.detach().cpu().numpy()


def make_in_between_views(views):
    ret = []
    for i in range(len(views) - 1):
        view_1 = views[i]
        view_2 = views[i + 1]
        camera = MiniCam(
            width=view_1.image_width,
            height=view_1.image_height,
            fovy=(view_1.FoVy + view_2.FoVy) / 2,
            fovx=(view_1.FoVx + view_2.FoVx) / 2,
            znear=view_1.znear,
            zfar=view_1.zfar,
            world_view_transform=(
                view_1.world_view_transform + view_2.world_view_transform
            )
            / 2,
            full_proj_transform=(
                view_1.full_proj_transform + view_2.full_proj_transform
            )
            / 2,
        )
        ret.append(camera)
    return ret


# gaussians have field ._distill_features of shape n_gaussians x 4k
# returns n_gaussians x 3(xyz) x signal_length
def get_gaussian_trajectory(gaussians, signal_length):
    compressed_trajectory = gaussians._distill_features
    reshaped_features = reshape_features(compressed_trajectory)
    padded_trajectory = pad_signal_coefficients(reshaped_features, signal_length)
    return ifft_features(padded_trajectory)


def set_gaussian_shift(gaussians, xyz):
    gaussians._xyz = xyz


def render_set(
    model_path, name, iteration, views, gaussians, pipeline, background, my_feat_decoder
):
    global global_args
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    in_between_views = make_in_between_views(views)
    new_views = []

    gaussian_trajectory = get_gaussian_trajectory(
        gaussians, args.n_frames
    )  # n_gaussians x 3 x n_frames
    original_gaussians_xyz = gaussians._xyz

    for i in range(len(views)):
        new_views.append(views[i])
        if i < len(in_between_views):
            new_views.append(in_between_views[i])
    for idx, view in enumerate(tqdm(new_views, desc="Rendering progress")):
        output_folder = declare_dir(
            os.path.join(global_args.special_output_folder, f"{idx}")
        )
        use_gt = not isinstance(view, MiniCam)
        if use_gt:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(
                gt, os.path.join(output_folder, "gt_image-{0:05d}".format(idx) + ".png")
            )

        wr = imageio.get_writer(
            os.path.join(output_folder, f"video_{idx}.mp4"), fps=60
        )  # TODO hard coded fps to 60 for now
        for t in range(args.n_frames):
            set_gaussian_shift(
                gaussians, original_gaussians_xyz + gaussian_trajectory[:, :, t]
            )
            render_pkg = render(view, gaussians, pipeline, background)
            rendering = render_pkg[
                "render"
            ]  # c x h x w , values from 0 to 1;; writer requires h x w x c
            rendering = rendering.permute((1, 2, 0))
            save_im = (rendering * 255).detach().cpu().numpy().astype(np.uint8)
            wr.append_data(save_im)
        wr.close()


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    global global_args
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, 64)  # hard coded 64 for now
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        ###
        (model_params, first_iter) = torch.load(global_args.gaussian_checkpoint)
        (
            gaussians.active_sh_degree,
            gaussians._xyz,
            gaussians._features_dc,
            gaussians._features_rest,
            gaussians._scaling,
            gaussians._rotation,
            gaussians._opacity,
            gaussians._distill_features,
            gaussians.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            gaussians.spatial_lr_scale,
        ) = model_params
        # gaussians.training_setup(training_args)
        gaussians.xyz_gradient_accum = xyz_gradient_accum
        gaussians.denom = denom
        # gaussians.optimizer.load_state_dict(opt_dict)
        assert global_args.decoder_checkpoint != ""
        my_feat_decoder = feat_decoder([64, 256, 512, global_args.feature_size]).cuda()
        my_feat_decoder.load_state_dict(torch.load(global_args.decoder_checkpoint))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                my_feat_decoder,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                my_feat_decoder,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gaussian_checkpoint", type=str)
    parser.add_argument("--decoder_checkpoint", type=str, required=False, default="")
    parser.add_argument("--feature_size", type=int)
    parser.add_argument("--n_frames", type=int)
    parser.add_argument("--special_output_folder", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    global_args = args
    makedirs(args.special_output_folder, exist_ok=True)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
    )
