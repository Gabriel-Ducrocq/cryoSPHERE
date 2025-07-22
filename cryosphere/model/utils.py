import os
import sys
import yaml
import wandb
import torch
import shutil
import einops
import random
import ntpath
import logging
import mrcfile
import warnings
import starfile
import numpy as np
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from scipy.spatial import distance
from cryosphere.model.vae import VAE
from cryosphere.model.mlp import MLP
from cryosphere.model.ctf import CTF
from biotite.structure.io.pdb import PDBFile
#from pytorch3d.transforms import Transform3d
from cryosphere.model.polymer import Polymer
from torch.distributed import init_process_group
from cryosphere.model.dataset import ImageDataSet
from cryosphere.model.gmm import Gaussian, EMAN2Grid
from cryosphere.model.segmentation import Segmentation
#from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix, axis_angle_to_quaternion, quaternion_apply
from cryosphere.model.loss import compute_loss, find_range_cutoff_pairs, remove_duplicate_pairs, find_continuous_pairs, calc_dist_by_pair_indices
import roma
from roma import unitquat_to_rotvec, rotvec_to_rotmat, rotvec_to_unitquat, quat_composition



def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


def primal_to_fourier2d(images):
    """
    Computes the fourier transform of the images.
    images: torch.tensor(batch_size, N_pix, N_pix)
    return: torch.tensor(batch_size, N_pix, N_pix) fourier transform of the images
    """
    r = torch.fft.ifftshift(images, dim=(-2, -1))
    fourier_images = torch.fft.fftshift(torch.fft.fft2(r, dim=(-2, -1), s=(r.shape[-2], r.shape[-1])), dim=(-2, -1))
    return fourier_images

def fourier2d_to_primal(fourier_images):
    """
    Computes the inverse fourier transform
    fourier_images: torch.tensor(batch_size, N_pix, N_pix)
    return: torch.tensor(batch_size, N_pix, N_pix) images in real space
    """
    f = torch.fft.ifftshift(fourier_images, dim=(-2, -1))
    r = torch.fft.fftshift(torch.fft.ifft2(f, dim=(-2, -1), s=(f.shape[-2], f.shape[-1])),dim=(-2, -1)).real
    return r

class Mask(torch.nn.Module):

    def __init__(self, im_size, rad, device):
        """
        Mask applied to the image, to exclude parts of the images that are only noise
        im_size: integer, number of pixels a side
        rad: float, radius of the mask
        """
        super(Mask, self).__init__()

        self.device=device
        self.rad = rad
        if rad is not None:
            mask = torch.lt(torch.linspace(-1, 1, im_size)[None]**2 + torch.linspace(-1, 1, im_size)[:, None]**2, rad**2).to(self.device)
            # float for pl ddp broadcast compatible
            self.register_buffer('mask', mask.float())
            self.num_masked = torch.sum(mask).item()

    def forward(self, x):
        """
        Applies the mask to batch of images
        x: torch.tensor(batch_size, im_size, im_size)
        """
        if self.rad is None:
            return x

        return x * self.mask



def low_pass_images(images, lp_mask2d):
    """
    Low pass filtering of the images.
    images: torch.tensor(batch_size, N_pix, N_pix)
    lp_mask2d: torch.tensor(N_pix, N_pix)
    return: torch.tensor(batch_size, N_pix, N_pix) low pass filtered images
    """
    f_images = primal_to_fourier2d(images)
    f_images = f_images * lp_mask2d
    images = fourier2d_to_primal(f_images).real
    return images


def low_pass_mask2d(shape, apix=1., bandwidth=2):
    """
    Defines a mask to apply in Fourier space for low pass filtering the images.
    shape: Number of pixels on one side.
    apix: size of a pixel
    bandwidth: set the radial frequencies greater than 1/bandwidth to 0.
    return: np.array(N_pix, N_pix) with entry one for the radial frequencies lower than 1/bandwidth and 0 otherwise.
    """
    freq = np.fft.fftshift(np.fft.fftfreq(shape, apix))
    freq = freq**2
    freq = np.sqrt(freq[:, None] + freq[None, :])
    if bandwidth is not None:
        mask = np.asarray(freq < 1 / bandwidth, dtype=np.float32)
    else:
        mask = np.ones(freq.shape)

    return mask


def set_wandb(experiment_settings):
    if experiment_settings["wandb"] == True:
        wandb.login()
        if experiment_settings["resume_training"]["model"] != None:
            name = f"experiment_{experiment_settings['name']}_resume"
        else:
            name = f"experiment_{experiment_settings['name']}"

        wandb.init(
            # Set the project where this run will be logged
            project=experiment_settings['wandb_project'],
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=name,


            # Track hyperparameters and run metadata
            config={
                "learning_rate": experiment_settings["optimizer"]["learning_rate"],
                "architecture": "VAE",
                "dataset": experiment_settings["cs_star_file"],
                "epochs": experiment_settings["N_epochs"],
            })


def parse_yaml(path, gpu_id, analyze=False):
    """
    Parse the yaml file to get the setting for the run.
    :param path: str, path to the yaml file
    :param analyze: boolean, set to true if this function is called from the analysis script. Otherwise False.
    :return: settings for the run
    """
    with open(path, "r") as file:
        experiment_settings = yaml.safe_load(file)

    if not analyze and gpu_id ==0:
        set_wandb(experiment_settings)

    if experiment_settings["seed"]:
        seed = experiment_settings["seed"]
        torch.manual_seed(seed)
        # Set seed for CUDA (if using GPUs)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # Set seed for Python's random module
        random.seed(seed)
        # Set seed for NumPy
        np.random.seed(seed)
        
    if experiment_settings["deterministic_cuda"]:
        # Ensure deterministic behavior for PyTorch operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(experiment_settings["seed"])
    folder_path = experiment_settings["folder_path"]
    image_file = os.path.join(folder_path, experiment_settings["image_yaml"])
    path_results = os.path.join(folder_path, "cryoSPHERE")
    if not os.path.exists(path_results):
        os.makedirs(path_results, exist_ok=True)

    #Getting name of the parameters yaml file
    parameter_file = os.path.basename(path)
    image_file = os.path.join(folder_path, experiment_settings["image_yaml"])
    if not analyze:
        #Copying the parameters yaml file to the results folder
        shutil.copyfile(path, os.path.join(path_results, parameter_file))
        #Copying the image yaml file to the results folder
        shutil.copyfile(image_file, os.path.join(path_results, experiment_settings["image_yaml"]))

    particles_path = os.path.join(folder_path, experiment_settings["particles_path"])

    with open(image_file, "r") as file:
        image_settings = yaml.safe_load(file)

    if experiment_settings["device"] == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"


    logging.basicConfig(filename=os.path.join(path_results, "run.log"), encoding='utf-8', level=logging.DEBUG, filemode='w', format='%(asctime)s %(levelname)s : %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S')

    N_images = experiment_settings["N_images"]
    apix = image_settings["apix"]
    Npix = image_settings["Npix"]
    Npix_downsize = image_settings["Npix_downsize"]
    amortized = experiment_settings["amortized"]
    apix_downsize = Npix * apix /Npix_downsize
    image_translator = SpatialGridTranslate(D=Npix_downsize, device=device)

    encoder = MLP(Npix_downsize**2,
                  experiment_settings["latent_dimension"] * 2,
                  experiment_settings["encoder"]["hidden_dimensions"], network_type="encoder", device=device)

    n_total_segments = 0 
    for part, part_config in experiment_settings["segmentation_config"].items():
        n_total_segments += part_config["N_segm"]

    decoder = MLP(experiment_settings["latent_dimension"], n_total_segments*6,
                  experiment_settings["decoder"]["hidden_dimensions"], network_type="decoder", device=device)


    vae = VAE(encoder, decoder, device, experiment_settings["segmentation_config"], latent_dim=experiment_settings["latent_dimension"], N_images = N_images, amortized=amortized)
    vae.to(device)
    if experiment_settings["resume_training"]["model"]:
        vae.load_state_dict(torch.load(experiment_settings["resume_training"]["model"]))
        vae.to(device)


    grid = EMAN2Grid(Npix_downsize, apix_downsize, device=device)
    base_structure_path = os.path.join(folder_path, experiment_settings["base_structure_path"])
    base_structure = Polymer.from_pdb(base_structure_path)
    amplitudes = torch.tensor(base_structure.num_electron, dtype=torch.float32, device=device)[:, None]
    gmm_repr = Gaussian(torch.tensor(base_structure.coord, dtype=torch.float32, device=device), 
                torch.ones((base_structure.coord.shape[0], 1), dtype=torch.float32, device=device)*image_settings["sigma_gmm"], 
                amplitudes)
    residues_chain = base_structure.chain_id
    residues_indexes = np.array([i for i in range(len(residues_chain))])
    N_residues = len(residues_indexes)

    segmenter = Segmentation(experiment_settings["segmentation_config"], residues_indexes, residues_chain, tau_segmentation=experiment_settings["tau_segmentation"], device=device)
    segmenter.to(device)
    experiment_settings["segmentation_prior"] = segmenter.segmentation_prior
    if experiment_settings["resume_training"]["segmentation"]:
        segmenter.load_state_dict(torch.load(experiment_settings["resume_training"]["segmentation"]))
        segmenter.to(device)
  

    if experiment_settings["optimizer"]["name"] == "adam":
        if "learning_rate_segmentation" not in experiment_settings["optimizer"]:
            list_param = [{"params": vae.parameters(), "lr":experiment_settings["optimizer"]["learning_rate"]}]
            list_param.append({"params": segmenter.parameters(), "lr":experiment_settings["optimizer"]["learning_rate"]})
            optimizer = torch.optim.Adam(list_param)
        else:
            list_param = [{"params": param, "lr":experiment_settings["optimizer"]["learning_rate_segmentation"]} for name, param in
                          segmenter.named_parameters() if "segments" in name]
            list_param.append({"params": vae.encoder.parameters(), "lr":experiment_settings["optimizer"]["learning_rate"]})
            list_param.append({"params": vae.decoder.parameters(), "lr":experiment_settings["optimizer"]["learning_rate"]})
            if not amortized:
                list_param.append({"params": vae.latent_variables_mean, "lr":experiment_settings["optimizer"]["learning_rate"]})

            optimizer = torch.optim.Adam(list_param)
    else:
        raise Exception("Optimizer must be Adam")


    cs_star_config = experiment_settings["cs_star_file"]
    ctf_experiment = CTF.create_ctf(cs_star_config, apix = apix_downsize, side_shape=Npix_downsize , device=device)
    dataset = ImageDataSet(apix, Npix, cs_star_config, particles_path, down_side_shape=Npix_downsize, rad_mask=experiment_settings.get("input_mask_radius"))

    scheduler = None
    if "scheduler" in experiment_settings:
        milestones = experiment_settings["scheduler"]["milestones"]
        decay = experiment_settings["scheduler"]["decay"]
        print(f"Using MultiStepLR scheduler with milestones: {milestones} and decay factor {decay}.")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay)

    N_epochs = experiment_settings["N_epochs"]
    batch_size = experiment_settings["batch_size"]

    lp_mask2d = low_pass_mask2d(Npix_downsize, apix_downsize, experiment_settings.get("lp_bandwidth"))
    lp_mask2d = torch.from_numpy(lp_mask2d).to(device).float()

    mask = Mask(Npix_downsize, experiment_settings.get("loss_mask_radius"), device)


    connect_pairs = find_continuous_pairs(base_structure.chain_id, base_structure.res_id, base_structure.atom_name)
    dists = calc_dist_by_pair_indices(base_structure.coord, connect_pairs)
    dists = torch.tensor(dists, device=device, dtype=torch.float32)
    assert "full_clashing_loss" in experiment_settings["loss"]["clashing_loss"], "Please indicate whether you want to use the full clashing loss or its light version."
    if experiment_settings["loss"]["clashing_loss"]["full_clashing_loss"]:
        clash_pairs = None
    else:
        clash_pairs = find_range_cutoff_pairs(base_structure.coord, experiment_settings["loss"]["clashing_loss"]["min_clashing_cutoff_pairs"],experiment_settings["loss"]["clashing_loss"]["max_clashing_cutoff_pairs"])
        clash_pairs = remove_duplicate_pairs(clash_pairs, connect_pairs)
        clash_pairs = torch.tensor(clash_pairs, device=device, dtype=torch.long)

    connect_pairs = torch.tensor(connect_pairs, device=device, dtype=torch.long)
    structural_loss_parameters = {"connect_pairs":connect_pairs, 
                       "clash_pairs":clash_pairs, 
                       "connect_distances":dists}


    logging.info(f"Running cryoSPHERE on folder: {folder_path}")
    if torch.cuda.device_count() == 1:
        logging.info(f"Running cryoSPHERE using one gpu: {device} with device number {torch.cuda.current_device()} and name {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logging.info(f"Running cryoSPHERE with {torch.cuda.device_count()} gpus.")
    logging.info(f"Find checkpoints at {path_results}")
    logging.info(f"Using particles: {particles_path}. Using starfile: {cs_star_config['file']}.")
    logging.info(f"Running the amortized version of cryoSPHERE: {amortized}. Training for {N_epochs} epochs.")
    logging.info(f"Image size: {Npix}. Pixel size: {apix}. Running cryoSPHERE on downsampled images of size: {Npix_downsize} with pixel size {apix_downsize}.")
    logging.info(f"""Low pass filtering bandwidth: {experiment_settings.get("lp_bandwidth")}. Input images mask radius: {experiment_settings.get('input_mask_radius')}. Correlation loss radius: {experiment_settings.get("loss_mask_radius")}.""")
    logging.info(f"Base structure: {experiment_settings['base_structure_path']} with {N_residues} residues.")
    logging.info(f"Latent dimension: {experiment_settings['latent_dimension']}")
    if amortized:
        logging.info(f"Encoder hidden layers: {experiment_settings['encoder']['hidden_dimensions']}")

    logging.info(f"Decoder hidden layers: {experiment_settings['decoder']['hidden_dimensions']}")
    logging.info(f"Batch size: {batch_size}.")
    logging.info(f"Learning rate for the encoder and decoder: {experiment_settings['optimizer']['learning_rate']}.")
    logging.info(f"""Learning rate for the segmentation GMM: {experiment_settings["optimizer"]["learning_rate"] if "learning_rate_segmentation" not in experiment_settings["optimizer"] 
                    else experiment_settings["optimizer"]["learning_rate_segmentation"]}.""")
    logging.info(f"""Using the model from previous run: {experiment_settings["resume_training"]["model"]}""")
    logging.info(f"""Using the segmentation from previous run: {experiment_settings["resume_training"]["segmentation"]}""")




    return vae, image_translator, ctf_experiment, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, device, \
    scheduler, base_structure, lp_mask2d, mask, amortized, path_results, structural_loss_parameters, segmenter


class SpatialGridTranslate(torch.nn.Module):
    """
    Class that defines the way we translate the images in real space.
    """
    def __init__(self, D, device=None) -> None:
        super().__init__()
        self.D = D
        # yapf: disable
        #Coord is of shape (N_coord, 2). The coordinates go from -1 to 1, representing not actual physical coordinates but rather proportion of a half image.
        coords = torch.stack(torch.meshgrid([
            torch.linspace(-1.0, 1.0, self.D, device=device),
            torch.linspace(-1.0, 1.0, self.D, device=device)],
        indexing="ij"), dim=-1).reshape(-1, 2)
        # yapf: enable
        self.register_buffer("coords", coords)

    def transform(self, images: torch.Tensor, trans: torch.Tensor):
        """
            The `images` are stored in `YX` mode, so the `trans` is also `YX`!

            Supposing that D is 96, a point is at 0.0:
                - adding 48 should move it to the right corner which is 1.0
                    1.0 = 0.0 + 48 / (96 / 2)
                - adding 96(>48) should leave it at 0.0
                    0.0 = 0.0 + 96 / (96 / 2) - 2.0
                - adding -96(<48) should leave it at 0.0
                    0.0 = 0.0 - 96 / (96 / 2) + 2.0

            Input:
                images: (B, N_pix, N_pix)
                trans:  (B, T,  2)

            Returns:
                images: (B, N_pix, N_pix)
        """
        B, NY, NX = images.shape
        assert self.D == NY == NX
        assert images.shape[0] == trans.shape[0]
        #We translate the coordinates not in terms of absolute translations but in terms of fractions of a half image, to be consistent with the way coord is defined.
        grid = einops.rearrange(self.coords, "N C2 -> 1 1 N C2") - \
            einops.rearrange(trans, "B T C2 -> B T 1 C2") * 2 / self.D
        grid = grid.flip(-1)  # convert the first axis from slow-axis to fast-axis
        grid[grid >= 1] -= 2
        grid[grid <= -1] += 2
        grid.clamp_(-1.0, 1.0)

        #We sample values at coordinates given by grid, using bilinear interpolation with padding mode zero.
        sampled = F.grid_sample(einops.rearrange(images, "B NY NX -> B 1 NY NX"), grid, align_corners=True)

        sampled = einops.rearrange(sampled, "B 1 T (NY NX) -> B T NY NX", NX=NX, NY=NY)
        return sampled[:, 0, :, :]


def monitor_training(segmentation, segmenter, tracking_metrics, experiment_settings, vae, optimizer, pred_im, true_im, gpu_id):
    """
    Monitors the training process through wandb and saving models. The metrics are logged into a file and optionnally sent to Weight and Biases.
    :param segmentation: torch.tensor(N_batch, N_residues, N_segments) weights of the segmentation
    :param segmenter: object of class Segmentation.
    :param tracking_metrics: dictionnary containing metrics to plot.
    :param experiment_settings: dictionnary containing parameters of the current experiment
    :param vae: object of class VAE.
    :param optimizer: optimizer object used in this run
    :param pred_im: torch.tensor(N_batch, N_pix, N_pix), sample of predicted images, without CTF corruption
    :param true_im: torch.tensor(N_batch, N_pix, N_pix), corresponding sample of true images. 
    """
    if gpu_id == 0:
        if tracking_metrics["wandb"] == True:
            ignore = ["wandb", "epoch", "path_results", "betas"]
            wandb.log({key: np.mean(val) for key, val in tracking_metrics.items() if key not in ignore})
            wandb.log({"epoch": tracking_metrics["epoch"]})
            wandb.log({"lr_segmentation":optimizer.param_groups[0]['lr']})
            wandb.log({"lr":optimizer.param_groups[1]['lr']})
            for part, segm in segmentation.items():
                hard_segments = np.argmax(segm["segmentation"].detach().cpu().numpy(), axis=-1)
                for l in range(segm["segmentation"].shape[-1]):
                    wandb.log({f"segments/{part}/segment_{l}": np.sum(hard_segments[0] == l)})


            pred_im = pred_im[0].detach().cpu().numpy()[:, :, None]
            true_im = true_im[0].detach().cpu().numpy()[:, :, None]
            predicted_image_wandb = wandb.Image(pred_im, caption="Predicted image")
            true_image_wandb = wandb.Image(true_im, caption="True image")
            wandb.log({"Images/true_image": true_image_wandb})
            wandb.log({"Images/predicted_image": predicted_image_wandb})
            for loss_term, beta in tracking_metrics["betas"].items():
                wandb.log({f"betas/{loss_term}": beta})

        model_path = os.path.join(experiment_settings["folder_path"], "cryoSPHERE", "ckpt" + str(tracking_metrics["epoch"]) + ".pt" )
        segmenter_path = os.path.join(experiment_settings["folder_path"], "cryoSPHERE", "seg" + str(tracking_metrics["epoch"]) + ".pt" )
        torch.save(vae.state_dict(), model_path)
        torch.save(segmenter.state_dict(), segmenter_path)
        information_strings = [f"""Epoch: {tracking_metrics["epoch"]} || Correlation loss: {tracking_metrics["correlation_loss"][0]} || KL prior latent: {tracking_metrics["kl_prior_latent"][0]} 
            || KL prior segmentation std: {tracking_metrics["kl_prior_segmentation_std"][0]} || KL prior segmentation proportions: {tracking_metrics["kl_prior_segmentation_proportions"][0]} ||
            l2 penalty: {tracking_metrics["l2_pen"][0]} || Continuity loss: {tracking_metrics["continuity_loss"][0]} || Clashing loss: {tracking_metrics["clashing_loss"][0]}"""]
        information_strings += [f"{loss_term} beta: {beta}" for loss_term, beta in tracking_metrics["betas"].items()]
        information_string = " || ".join(information_strings)
        logging.info(information_string)


def read_pdb(path):
    """
    Reads a pdb file in a structure object of biopdb
    :param path: str, path to the pdb file.
    :return: a biotite AtomArray or AtomArrayStack
    """
    _, extension = os.path.splitext(path)
    assert extension == "pdb", "The code currently supports only pdb files."
    f = PDBFile.read(path)
    atom_array_stack = f.get_structure()
    if len(atom_array_stack) > 1:
        warnings.warn("More than one structure in the initial pdb file. Using the first one")

    return atom_array_stack[0]

def compute_rotations_per_residue_einops(quaternions, segmentation, device):
    """
    Computes the rotation matrix corresponding to each residue, for the part we want to tackle.
    :param quaternions: tensor (N_batch, N_segments, 4) of non normalized quaternions defining rotations
    :param segmentation: tensor (N_batch, N_residues, N_segments)
    :return: tensor (N_batch, N_residues, 3, 3) rotation matrix for each residue
    """

    N_residues = segmentation.shape[1]
    batch_size = quaternions.shape[0]
    N_segments = segmentation.shape[-1]
    # NOTE: no need to normalize the quaternions, as we fix the real part to 1, when taking sin theta/2 / cos theta/2 = norm/real_part = norm and norm can be more than one, as sin/cos can be arbitrarily high.
    rotation_per_domains_axis_angle = unitquat_to_rotvec(quaternions[:, [1, 2, 3, 0]])
    segmentation_rotation_per_domains_axis_angle = segmentation[:, :, :, None] * rotation_per_domains_axis_angle[:, None, :, :]
    segmentation_rotation_matrix_per_domain_per_residue = rotvec_to_rotmat(segmentation_rotation_per_domains_axis_angle)
    ## Flipping to keep in line with the previous implementation
    segmentation_rotation_matrix_per_domain_per_residue = torch.einsum("brdle->dbrle", segmentation_rotation_matrix_per_domain_per_residue).flip(0)
    dimensions = ",".join([f"b r a{i} a{i+1}" for i in range(N_segments)])
    dimensions += f"-> b r a0 a{N_segments}"
    overall_rotation_matrices = einops.einsum(*segmentation_rotation_matrix_per_domain_per_residue, dimensions)
    return overall_rotation_matrices


def rotate_residues_einops(atom_positions, quaternions, segmentation, device):
    """
    Rotates each residues based on the rotation predicted for each domain and the predicted segmentation.
    :param positions: torch.tensor(N_residues, 3)
    :param quaternions: tensor (N_batch, N_segments, 4) of non normalized quaternions defining rotations
    :param segmentation: tensor (N_batch, N_residues, N_segments)
    :return: tensor (N_batch, N_residues, 3, 3) rotation matrix for each residue
    """

    N_residues = segmentation.shape[1]
    batch_size = quaternions.shape[0]
    N_segments = segmentation.shape[-1]
    # NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
    rotation_per_segments_axis_angle = unitquat_to_rotvec(quaternions[:, :, [1, 2, 3, 0]])
    #The below tensor is [N_batch, N_residues, N_segments, 3]
    segmentation_rotation_per_segments_axis_angle = segmentation[:, :, :, None] * rotation_per_segments_axis_angle[:, None, :, :]
    #The below tensor is [N_batch, N_residues, N_segments, 4] with the real part as the last element from now on !!!!!
    segmentation_rotation_per_segments_quaternions = rotvec_to_unitquat(segmentation_rotation_per_segments_axis_angle)
    segmentation_rotation_per_residue = quat_composition(segmentation_rotation_per_segments_quaternions, normalize=True)
    #T = Transform3d(dtype=torch.float32, device = device)
    transform = roma.RotationUnitQuat(segmentation_rotation_per_residue)
    print("Quaternions shape", segmentation_rotation_per_residue.shape)
    print("atom positions", atom_positions.shape)
    atom_positions = transform.apply(atom_positions[None, :, :])
    return atom_positions

def compute_translations_per_residue(translation_vectors, segmentations, N_residues, batch_size, device):
    """
    Computes one translation vector per residue based on the segmentation
    :param translation_vectors: dictionnary, for each part of the protein torch.tensor (Batch_size, N_segments, 3) translations for each domain 
    :param segmentations: dictionnary of torch.tensor(N_batch, N_residues, N_segments) representing the weights of the segmentation
                         and mask to find the relevant residues among the protein.
    :param N_residues: integer, total number of residues in the protein
    :param batch_size: integer, size of the batch.
    :param device: torch device on which we perform the computations.
    :return: translation per residue torch.tensor(batch_size, N_residues, 3)
    """
    translation_per_residue = torch.zeros((batch_size, N_residues, 3), dtype=torch.float32, device=device)
    for part, segm in segmentations.items():
        translation_per_residue[:, segm["mask"] == 1] += torch.einsum("bij, bjk -> bik", segm["segmentation"], translation_vectors[part])

    return translation_per_residue

def deform_structure(atom_positions, translation_per_residue, quaternions, segmentations, device):
    """
    Deform the base structure according to rotations and translation of each segment, together with the segmentation.
    :param atom_positions: torch.tensor(N_residues, 3)
    :param translation_per_residue: tensor (Batch_size, N_residues, 3)
    :param quaternions: tensor (N_batch, N_segments, 4) of quaternions for the rotation of the segments
    :param segmentations: dictionnary of torch.tensor(N_batch, N_residues, N_segments) representing the weights of the segmentation 
                          and mask to find the relevant residues among the protein.
    :param device: torch device on which the computation takes place
    :return: tensor (Batch_size, N_residues, 3) corresponding to translated structure
    """
    batch_size = translation_per_residue.shape[0]
    transformed_atom_positions = atom_positions[None, :, :].repeat((batch_size, 1, 1))
    for part, segm in segmentations.items():
        transformed_atom_positions[:, segm["mask"]==1]  = rotate_residues_einops(atom_positions[segm["mask"]==1] , quaternions[part], segm["segmentation"], device)

    new_atom_positions = transformed_atom_positions + translation_per_residue
    return new_atom_positions



