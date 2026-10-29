import sys

import roma
import torch
import wandb
import logging
import argparse
import numpy as np
from roma import unitquat_to_rotmat, rotvec_to_unitquat
from tqdm import tqdm
from time import time
from cryosphere import model
import torch.nn.functional as F
import torch.multiprocessing as mp
from cryosphere.model import renderer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from cryosphere.model.utils import low_pass_images, ddp_setup, compute_rotation_matrix_in_0
from torch.distributed import destroy_process_group
from cryosphere.model.loss import compute_loss, calc_cor_loss

logger = logging.getLogger(__name__)

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True, help="path to the yaml containing all the parameters for the cryoSPHERE run.")

def train(rank, world_size, yaml_setting_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    """
    ddp_setup(rank, world_size)
    (vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, device, scheduler, 
    base_structure, lp_mask2d, mask_images, amortized, path_results, structural_loss_parameters, segmenter) = model.utils.parse_yaml(yaml_setting_path, rank)
    start_training(vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, scheduler, 
    base_structure, lp_mask2d, mask_images, amortized, path_results, structural_loss_parameters, segmenter, rank)
    destroy_process_group()

def start_training(vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, scheduler, 
    base_structure, lp_mask2d, mask_images, amortized, path_results, structural_loss_parameters, segmenter, gpu_id):
    vae = DDP(vae, device_ids=[gpu_id])
    segmenter = DDP(segmenter, device_ids=[gpu_id])
    for epoch in range(N_epochs):
        tracking_metrics = {"wandb":experiment_settings["wandb"], "epoch": epoch, "path_results":path_results ,"correlation_loss":[], "kl_prior_latent":[], 
                            "kl_prior_segmentation_mean":[], "kl_prior_segmentation_std":[], "kl_prior_segmentation_proportions":[], "l2_pen":[], "continuity_loss":[], 
                            "clashing_loss":[], "geodesic_rotation":[]}

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = experiment_settings["num_workers"], drop_last=True, sampler=DistributedSampler(dataset, drop_last=True))
        start_tot = time()
        data_loader.sampler.set_epoch(epoch) 
        data_loader = tqdm(iter(data_loader))
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_quat, batch_init_poses, batch_poses_translation, _) in enumerate(data_loader):
            batch_images = batch_images.to(gpu_id)
            batch_poses = batch_poses.device(gpu_id)
            batch_init_poses = batch_init_poses.device(gpu_id)
            batch_poses_translation = batch_poses_translation.to(gpu_id)
            indexes = indexes.to(gpu_id)
            flattened_batch_images = batch_images.flatten(start_dim=-2)
            batch_translated_images = image_translator.transform(batch_images, batch_poses_translation[:, None, :])
            lp_batch_translated_images = low_pass_images(batch_translated_images, lp_mask2d)
            if amortized:
                latent_variables, latent_mean, latent_std = vae.module.sample_latent(flattened_batch_images)
            else:
                latent_variables, latent_mean, latent_std = vae.module.sample_latent(None, indexes)

            segmentation = segmenter.module.sample_segments(batch_images.shape[0])
            quaternions_per_domain, translations_per_domain = vae.module.decode(latent_variables)
            translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, segmentation, base_structure.coord.shape[0], batch_size, gpu_id)
            predicted_structures = model.utils.deform_structure(gmm_repr.mus, translation_per_residue, quaternions_per_domain, segmentation, gpu_id)
            for _ in range(experiment_settings["structure_iters"]):
                posed_predicted_structures = renderer.rotate_structure(predicted_structures, batch_poses)
                predicted_images  = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
                batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)/dataset.f_std
                loss = compute_loss(batch_predicted_images, lp_batch_translated_images, None, latent_mean, latent_std, vae.module, segmenter.module, experiment_settings, tracking_metrics,
                    structural_loss_parameters= structural_loss_parameters, epoch=epoch, predicted_structures=predicted_structures, device=gpu_id)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            vae.module.requires_grad_(False)
            segmenter.module.requires_grad_(False)
            for _ in range(experiment_settings["pose_iters"]):
                identity_w = torch.zeros((batch_size, 3), device=gpu_id, requires_grad=True)
                optimizer_pose = torch.optim.Adam(identity_w, lr=experiment_settings["learning_rate"]["pose_lr"])
                identity_rotmat = compute_rotation_matrix_in_0(identity_w)
                new_rotmat = torch.einsum("bij, bjk", identity_rotmat, batch_poses)
                posed_predicted_structures = renderer.rotate_structure(predicted_structures,new_rotmat)
                predicted_images  = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
                batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)/dataset.f_std
                loss_poses = calc_cor_loss(batch_predicted_images, lp_batch_translated_images, None)
                loss_poses.backward()
                optimizer_pose.step()
                optimizer_pose.zero_grad()
                new_quat = rotvec_to_unitquat(identity_w)
                batch_poses_quat = roma.quat_normalize(roma.quat_product(new_quat, batch_poses_quat))
                batch_poses = unitquat_to_rotmat(batch_poses_quat)


            geodesic_distance = roma.rotmat_geodesic_distance(batch_poses.detach(), batch_init_poses).mean()
            tracking_metrics["geodesic_rotation"].append(geodesic_distance.detach().cpu().numpy())
            dataset.poses_quat[indexes] = batch_poses_quat.detach()
            vae.module.requires_grad_(True)
            segmenter.module.requires_grad_(True)

        if scheduler:
            scheduler.step()

        model.utils.monitor_training(segmentation, segmenter.module, tracking_metrics, experiment_settings, vae.module, optimizer, predicted_images, batch_images, gpu_id)


def cryosphere_train():
    """
    This function serves as an entry point to be called from the command line 
    """
    args = parser_arg.parse_args()
    path = args.experiment_yaml

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, path), nprocs=world_size)


if __name__ == '__main__':
    cryosphere_train()

