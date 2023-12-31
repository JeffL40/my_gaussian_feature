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

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, feat_decoder
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import pdb
from utils.graphics_utils import (
    compute_camera_intrinsics_matrix,
    compute_camera_to_pixel_matrix,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

PROFILING_FLAG = False

global global_args


# for flow, unravels the blended_features of shape ... x raveled_xyz_real_coefficients to ... x 3(xyz) x k complex cefficients
def reshape_features(blended_features):
    len_real = blended_features.shape[-1] // 2
    complex_blended_features = (
        blended_features[..., :len_real] + blended_features[..., len_real:]
    )
    len_x = complex_blended_features.shape[-1] // 3
    x_feats = complex_blended_features[..., :len_x]
    y_feats = complex_blended_features[..., len_x : 2 * len_x]
    z_feats = complex_blended_features[..., 2 * len_x :]
    return torch.stack([x_feats, y_feats, z_feats], dim=-2)


def pad_signal_coefficients(complex_coefficients, signal_length):
    ret = torch.zeros(complex_coefficients.shape[:-1] + (signal_length,))
    k = complex_coefficients.shape[-1]
    ret[..., 1 : k + 1] += complex_coefficients
    ret[..., -k:] += -torch.conj(complex_coefficients)
    return ret


# features of shape ... x k, turn the k into signal_length through decompression
def ifft_features(features, signal_length):
    return torch.real(torch.fft.ifft(pad_signal_coefficients(features, signal_length)))


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    low_dim_feat,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    FAULTY_CARD_FLAG = False  # my old 3090 crashes when utilization is too high...
    if FAULTY_CARD_FLAG:
        print(
            "Faulty card flag set, will sleep for 50ms after each iteration to avoid crashing the GPU"
        )

    global global_args
    if low_dim_feat:
        # TODO(roger): automate this layer based on low_dim_feat flag and GT feat dim
        my_feat_decoder = feat_decoder([64, 256, 512, global_args.feature_size]).cuda()
        decoder_optimizer = torch.optim.Adam(my_feat_decoder.parameters(), lr=0.001)

    for iteration in range(first_iter, opt.iterations + 1):
        if FAULTY_CARD_FLAG:
            time.sleep(0.04)  # force a little delay to avoid crashing the GPU
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        start_cp = time.time()

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        rendered_feat = render_pkg["render_feat"]

        if viewpoint_cam.feat_chw is not None:
            gt_feat = viewpoint_cam.feat_chw.cuda()
        else:
            gt_feat = None

        if PROFILING_FLAG:
            # Synchronize CUDA stream
            torch.cuda.synchronize()
            time_elapsed = time.time() - start_cp
            print("Rasterization took {} seconds".format(time_elapsed))

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        if gt_feat is not None:
            rendered_feat_bhwc = F.interpolate(
                rendered_feat.unsqueeze(0),
                size=gt_feat.shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            # gt_feat = torch.ones((rendered_feat.shape[0], 24, 32), device="cuda")
            if low_dim_feat:
                resized_feat = my_feat_decoder(rendered_feat_bhwc)
                resized_feat = resized_feat.squeeze(0)
            else:
                resized_feat = rendered_feat_bhwc.squeeze(0)
            if global_args.is_flow:
                reshaped_features = reshape_features(resized_feat)  # ... x 3(xyz) x k
                tmp = reshaped_features.transpose(-1, -2)  # ... x k x 3(xyz)
                # project features onto camera plane
                camera_intrinsics_matrix = compute_camera_intrinsics_matrix(
                    image.shape[1],
                    image.shape[0],
                    viewpoint_cam.FoVx,
                    viewpoint_cam.FoVy,
                )  # 3 x 3
                image_flow = torch.einsum(
                    "...c,jc->...j", tmp, camera_intrinsics_matrix
                )  # ... x 3(xyh)
                image_flow = (image_flow / image_flow[..., 2:3])[
                    ..., :2
                ]  # ... x k x 2(xy)
                real_part = torch.real(image_flow)  # ... x k x 2(xy)
                imag_part = torch.imag(image_flow)
                processed_feat = torch.concatenate([real_part, imag_part], dim=-1)
                feat_loss = l2_loss(processed_feat, gt_feat)
            else:
                feat_loss = l2_loss(resized_feat, gt_feat)
        else:
            feat_loss = 0.0

        loss = loss + 0.1 * feat_loss

        if FAULTY_CARD_FLAG:
            time.sleep(0.04)  # force a little delay to avoid crashing the GPU

        start_cp = time.time()

        loss.backward()

        if PROFILING_FLAG:
            torch.cuda.synchronize()
            time_elapsed = time.time() - start_cp
            print("Backward took {} seconds".format(time_elapsed))

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                print(
                    "[ITER {}] Number of points {}".format(
                        iteration, gaussians.get_xyz.shape[0]
                    )
                )
                scene.save(iteration)
                if low_dim_feat:
                    print("[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save(
                        my_feat_decoder.state_dict(),
                        os.path.join(scene.model_path, "feat_decoder.pth"),
                    )

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if low_dim_feat and iteration < 5000:
                    decoder_optimizer.step()
                gaussians.optimizer.step()
                if low_dim_feat and iteration < 5000:
                    decoder_optimizer.zero_grad()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--feature_size", type=int, default=768, required=False)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[1_000, 4_000, 7_000, 1_0000, 15_000, 20_000, 25_000, 30_000],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[500, 1_000, 2_000, 4_000, 7_000, 10_000, 30_000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--low_dim_feat", action="store_true", default=False)
    parser.add_argument("--is_flow", type=int, required=True, help="0 or 1")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    global_args = args
    print(f"Assuming features have size {global_args.feature_size}")

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.low_dim_feat,
    )

    # All done
    print("\nTraining complete.")
