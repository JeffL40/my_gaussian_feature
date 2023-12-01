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
import pdb

global global_args


def render_set(
    model_path, name, iteration, views, gaussians, pipeline, background, my_feat_decoder
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    pdb.set_trace()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]

        rendered_feat = rendering["render_feat"]
        gt_feat = view.feat_chw.cuda()

        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    global global_args
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, global_args.distill_feature_dim)
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
    parser.add_argument("--distill_feature_dim", type=int, default=64, required=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    global_args = args
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
    )
