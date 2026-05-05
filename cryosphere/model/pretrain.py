import wandb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler




def monitor_pretraining(tracking_metrics):
    """
    Tracks the loss curve of the pretraining on wandb
    :param tracking_metrics: dictionnary, containing the
    """
    wandb.log({key: val for key, val in tracking_metrics.items()})



def disable_encoder_gradient(vae):
    """
    Cancel the need for gradient on encoder
    """
    for p in vae.module.encoder.parameters():
        p.requires_grad = False

def enable_encoder_gradient(vae):
    """
    Enable the need for gradient on encoder.
    """
    for p in vae.module.encoder.parameters():
        p.requires_grad = True

def pretrain(vae, dataset, experiment_settings, gpu_id):
    """
    Pretraining the decoder so that it output an identity rotation. The encoder training is disabled and enabled at the
    end of the pretraining.
    :param vae: object of type VAE
    :param dataset: object of type dataset
    :param batch_size: integer, size of batch
    :param gpu_id: id of the gpu
    """
    tracking_metrics = {"mse":[]}
    batch_size = experiment_settings["pretraining"]["batch_size"]
    disable_encoder_gradient(vae)
    list_param = [{"params": vae.module.decoder.parameters(), "lr":experiment_settings["pretraining"]["optimizer"]["learning_rate"]}]
    optimizer = torch.optim.Adam(list_param)
    for epoch in range(experiment_settings["pretraining"]["N_epochs"]):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=experiment_settings["num_workers"], drop_last=True,
                                 sampler=DistributedSampler(dataset, drop_last=True))
        data_loader.sampler.set_epoch(epoch)
        data_loader = tqdm(iter(data_loader))
        all_losses = []
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation, _) in enumerate(data_loader):
            batch_images = batch_images.to(gpu_id)
            flattened_batch_images = batch_images.flatten(start_dim=-2)
            latent_variables, latent_mean, latent_std = vae.module.sample_latent(flattened_batch_images)

            r6_per_domain, translations_per_domain = vae.module.decode(latent_variables)
            loss = torch.zeros(1, device=gpu_id)
            for part, predicted_r6 in r6_per_domain.items():
                r6_identity = torch.zeros_like(r6_per_domain[part], device=gpu_id)
                r6_identity[:, :, 0, 0] = 1
                r6_identity[:, :, 1, 1] = 1
                loss += torch.mean(torch.sum((r6_identity - r6_per_domain[part])**2, dim=-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_losses.append(loss.detach().cpu().numpy())


        tracking_metrics["mse"] = np.mean(all_losses)
        monitor_pretraining(tracking_metrics)

    enable_encoder_gradient(vae)