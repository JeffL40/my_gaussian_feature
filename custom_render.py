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

global global_args


def declare_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def torch2numpy(a):
    return a.detach().cpu().numpy()


def make_in_between_views(views):
    ret = []
    pdb.set_trace()
    for i in range(len(views) - 1):
        view_1 = views[i]
        view_2 = views[i + 1]
        camera = MiniCam(
            width=view_1.image_width,
            height=view_1.image_height,
            fovy=view_1.fovy,
            fovx=view_1.fovx,
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
    for i in range(len(views)):
        new_views.append(views[i])
        if i < len(in_between_views):
            new_views.append(in_between_views[i])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        gt_feat = view.feat_chw.cuda()

        rendered_feat = render_pkg["render_feat"]
        rendered_feat_bhwc = F.interpolate(
            rendered_feat.unsqueeze(0),
            size=gt_feat.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        resized_feat = my_feat_decoder(rendered_feat_bhwc)
        resized_feat = resized_feat.squeeze(
            0
        )  # both gt feat and resized feat now 96 x h x w

        # save the render image and features
        output_folder = declare_dir(
            os.path.join(global_args.special_output_folder, f"{idx}")
        )
        # save regular image
        torchvision.utils.save_image(
            rendering,
            os.path.join(output_folder, "rendering-{0:05d}".format(idx) + ".png"),
        )
        torchvision.utils.save_image(
            gt, os.path.join(output_folder, "gt_image-{0:05d}".format(idx) + ".png")
        )
        # save rendered features
        np.save(
            os.path.join(output_folder, "rendered_feat.npy"), torch2numpy(resized_feat)
        )
        # save gt features
        np.save(os.path.join(output_folder, "gt_feat.npy"), torch2numpy(gt_feat))


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
