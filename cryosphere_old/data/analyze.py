import sys
import os
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import torch
from cryosphere.model import utils
import argparse
import starfile
import numpy as np
import seaborn as sns
from time import time
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from cryosphere.model.polymer import Polymer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True, help="path to the yaml defining the experimentation")
parser_arg.add_argument("--model", type=str, required=True, help="path to the model we want to analyze")
parser_arg.add_argument("--segmenter", type=str, required=True, help="path to the segmenter we want to analyze")
parser_arg.add_argument("--output_path", type=str, required=True, help="path of the directory to save the results")
parser_arg.add_argument("--z", type=str, required=False, help="path of the latent variables in npy format, if we already have them")
parser_arg.add_argument("--thinning", type=int, required=False, default= 1,  help="""thinning to apply on the latent variables to perform the PCA analysis: if there are too many images,
                        the PCA may take a long time, hence thinning might be needed. For example, thinning = 10 takes one latent variable out of ten for the PCA analysis.""")
parser_arg.add_argument("--num_points", type=int, required=False, default= 20, help="Number of points to generate for the PC traversals")
parser_arg.add_argument('--dimensions','--list', nargs='+', type=int, default= [0, 1, 2], help='<Required> PC dimensions along which we compute the trajectories. If not set, use pc 1, 2, 3', required=False)
parser_arg.add_argument('--generate_structures', action=argparse.BooleanOptionalAction, default= False, help="""If False: run a PCA analysis with PCA traversal. If True,
                            generates the structures corresponding to the latent variables given in z.""")



class LatentDataSet(Dataset):
    def __init__(self, z):
        """
        Create a dataset of images and poses
        :param z: latent variable to decode into structures
        """

        self.z = z

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        """
        #Return a batch of true images, as 2d array
        # return: the set of indexes queried for the batch, the corresponding images as a torch.tensor((batch_size, side_shape, side_shape)), 
        # the corresponding poses rotation matrices as torch.tensor((batch_size, 3, 3)), the corresponding poses translations as torch.tensor((batch_size, 2))
        # NOTA BENE: the convention for the rotation matrix is left multiplication of the coordinates of the atoms of the protein !!
        """
        return idx, self.z[idx]



def gather(tens, tens_list=None, root=0, group=None):
    """
        Sends tensor to root process, which store it in tens_list.
    """
  
    rank = torch.distributed.get_rank()
    if group is None:
        group = torch.distributed.group.WORLD
    if rank == root:
        assert(tens_list is not None)
        torch.distributed.gather(tens, gather_list=tens_list, group=group)
    else:
        torch.distributed.gather(tens, dst=root, group=group)

def all_gather(tens, tens_list=None, root=0, group=None):
    """
        Sends tensor to root process, which store it in tens_list.
    """
  
    rank = torch.distributed.get_rank()
    if group is None:
        group = torch.distributed.group.WORLD
    if rank == root:
        assert(tens_list is not None)
        torch.distributed.all_gather(tens_list, tens, group=group)
    else:
        torch.distributed.gather(tens, group=group)


def concat_and_save(tens, path):
    """
    Concatenate the lsit of tensor along the dimension 0
    :param tens: list of tensor with batch size as dim 0
    :param path: str, path to save the torch tensor
    :return: tensor of concatenated tensors
    """
    concatenated = torch.concat(tens, dim=0)
    np.save(path, concatenated.detach().numpy())
    return concatenated


def compute_traversals(z, dimensions = [0, 1, 2], num_points=10):
    pca = PCA()
    z_pca = pca.fit_transform(z)
    all_trajectories = []
    all_trajectories_pca = []
    for dim in dimensions:
            traj_pca = graph_traversal(z_pca, dim, num_points)
            ztraj_pca = pca.inverse_transform(traj_pca)
            nearest_points, _ = get_nearest_point(z, ztraj_pca)
            all_trajectories.append(nearest_points)
            all_trajectories_pca.append(traj_pca)
        
    return all_trajectories, all_trajectories_pca, z_pca, pca



def get_nearest_point(data, query):
    """
    Find closest point in @data to @query
    Return datapoint, index
    """
    ind = cdist(query, data).argmin(axis=1)
    return data[ind], ind

def graph_traversal(z_pca, dim, num_points=10):
    z_pca_dim = z_pca[:, int(dim)]
    start = np.percentile(z_pca_dim, 5)
    stop = np.percentile(z_pca_dim, 95)
    traj_pca = np.zeros((num_points, z_pca.shape[1]))
    traj_pca[:, dim] = np.linspace(start, stop, num_points)
    return traj_pca

def start_sample_latent(rank, world_size,  yaml_setting_path, output_path, model_path, segmenter_path, num_workers=4):
    utils.ddp_setup(rank, world_size)
    (vae, image_translator, ctf_experiment, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, device,
    scheduler, base_structure, lp_mask2d, mask, amortized, path_results, structural_loss_parameters, segmenter)  = utils.parse_yaml(yaml_setting_path, rank, analyze=True)
    vae.load_state_dict(torch.load(model_path))
    vae.eval()
    z = sample_latent_variables(rank, world_size, vae, dataset, batch_size, output_path)
    destroy_process_group()

def sample_latent_variables(gpu_id, world_size, vae, dataset, batch_size, output_path, num_workers=4):
    """
    Sample all the latent variables of the dataset and save them in a .npy file
    :param vae: object of class VAE corresponding to the model we want to analyze.
    :param dataset: object of class dataset: data on which to analyze the model
    :param batch_size: integer, batch size
    :param output_path: str, path where we want to register the latent variables
    :param num_workers: integer, number of workers
    return 
    """
    vae.to(gpu_id)
    vae = DDP(vae, device_ids=[gpu_id])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, sampler=DistributedSampler(dataset, shuffle=False, drop_last=False))
    data_loader.sampler.set_epoch(0)
    data_loader = tqdm(iter(data_loader))
    all_latent_variables = []
    all_indexes = []
    for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation, _) in enumerate(data_loader):
        batch_images = batch_images.to(gpu_id)
        batch_poses = batch_poses.to(gpu_id)
        batch_poses_translation = batch_poses_translation.to(gpu_id)
        indexes = indexes.to(gpu_id)

        batch_images = batch_images.flatten(start_dim=-2)
        latent_variables, latent_mean, latent_std = vae.module.sample_latent(batch_images, indexes)
        latent_mean = latent_mean.contiguous()
        indexes = indexes.contiguous()
        if gpu_id == 0:
            batch_latent_mean_list = [torch.zeros_like(latent_mean, device=latent_mean.device).contiguous() for _ in range(world_size)]
            batch_indexes = [torch.zeros_like(indexes, device=batch_images.device).contiguous() for _ in range(world_size)]
            gather(latent_mean, batch_latent_mean_list)
            gather(indexes, batch_indexes)
        else:
            gather(latent_mean)
            gather(indexes)

        if gpu_id == 0:
            all_gpu_indexes = torch.concat(batch_indexes, dim=0)
            all_gpu_latent_mean = torch.concat(batch_latent_mean_list, dim=0)
            sorted_batch_indexes = torch.argsort(all_gpu_indexes, dim=0)
            sorted_batch_latent_mean = all_gpu_latent_mean[sorted_batch_indexes]
            all_latent_variables.append(sorted_batch_latent_mean.detach().cpu().numpy())
            all_indexes.append(all_gpu_indexes[sorted_batch_indexes].detach().cpu().numpy())


    if gpu_id == 0:
        all_latent_variables = np.concatenate(all_latent_variables, axis=0)
        all_indexes = np.concatenate(all_indexes, axis = 0)
        latent_path = os.path.join(output_path, "z.npy")
        np.save(latent_path, all_latent_variables)


def plot_pca(output_path, dim, all_trajectories_pca, z_pca, pca):
    """
    Function in charge of plotting the PCA of the latent space with the PC traversals
    :param output_path: str, path to the output directory
    :param dim: intger, dimension along which we generate a traversal.
    :param all_trajectories_pca: np.array(num_points, N_pc) coordinates of the points sampled during the traversal.
    :param z_pca: np.array(N_latent_variables, PCA_latent_dim) all the latent variables in the PCA coordinate system.
    :param pca: scikit-learn PCA object.
    """
    os.makedirs(os.path.join(output_path, f"pc{dim}/"), exist_ok=True)
    sns.kdeplot(x=z_pca[:, dim], y=z_pca[:, dim+1], fill=True, clip= (-50, 50))
    print("TRJACTORIES", all_trajectories_pca[dim][:, :])
    plt.scatter(x=all_trajectories_pca[dim][:, dim], y=all_trajectories_pca[dim][:, dim+1], c="red")
    plt.title("PCA of the latent space")
    plt.xlabel(f"PC {dim+1}, variance {pca.explained_variance_ratio_[dim]} ")
    plt.ylabel(f"PC {dim+2}, variance variance {pca.explained_variance_ratio_[dim+1]}")
    plt.savefig(os.path.join(output_path, f"pc{dim}/pca.png"))
    plt.close()

def predict_structures(vae, z_dim, gmm_repr, segmenter, device):
    """
    Function predicting the structures for a PC traversal along a specific PC.
    :param vae: object of class VAE.
    :param z_dim: np.array(num_points, latent_dim) coordinates of the sampled structures for the PC traversal
    :param gmm_repr: Gaussian representation. Object of class Gaussian.
    :param predicted_structures: torch.tensor(num_points, N_residues, 3), predicted structutres for each one of the sampled points of the PC traversal.
    :param segmenter: object of class Segmentation
    :param device: torch device
    """
    z_dim = torch.tensor(z_dim, dtype=torch.float32, device=device)
    segmentation = segmenter.sample_segments(z_dim.shape[0])
    quaternions_per_domain, translations_per_domain = vae.decode(z_dim)
    translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, segmentation, gmm_repr.mus.shape[0],z_dim.shape[0], device)
    predicted_structures = utils.deform_structure(gmm_repr.mus, translation_per_residue, quaternions_per_domain, segmentation, device)
    return predicted_structures


def save_structure(base_structure, path):
    """
    Save one structure in a PDB file, saved at path
    :param base_structure: object of class Polymer
    :param path: str, path to which we save the PDB
    """
    base_structure.to_pdb(path)


def save_structures_pca(predicted_structures, dim, output_path, base_structure):
    """
    Save a set of structures given in a torch tensor in different pdb files.
    :param predicted_structures: torch.tensor(N_predicted_structures, N_residues, 3), et of structures
    :param dim: integer, dimension along which we sample
    :param output_path: str, path to the directory in which we save the structures.
    :param base_structrue: object of class Polymer
    """
    for i, pred_struct in enumerate(predicted_structures):
        print("Saving structure", i+1, "from pc", dim)
        base_structure.coord = pred_struct.detach().cpu().numpy()
        save_structure(base_structure, os.path.join(output_path, f"pc{dim}/structure_z_{i}.pdb"))

def save_structures(predicted_structures, base_structure, batch_num, output_path, batch_size, indexes):
    """
    Save structures in batch, with the correct numbering .
    :param predicted_structures: torch.tensor(N_batch, N_residues, 3) of predicted structures
    :param base_structure: object of class Polymer.
    :param batch_num: integer, batch number
    :param output_path: str, path where we want to save the structures
    """
    for i, pred_struct in enumerate(predicted_structures):
        print("Saving structure", batch_num*batch_size + i)
        base_structure.coord = pred_struct.detach().cpu().numpy()
        base_structure.to_pdb(os.path.join(output_path, f"structure_z_{indexes[i]}.pdb"))

def run_pca_analysis(vae, z, dimensions, num_points, output_path, gmm_repr, base_structure, thinning, segmenter, device):
    """
    Runs a PCA analysis of the latent space and return PC traversals and plots of the PCA of the latent space
    :param vae: object of class VAE.
    :param z: torch.tensor(N_latent, latent_dim) containing all the latent variables
    :param dimensions: list of integer, list of PC dimensions we want to traverse
    :param num_points: integer, number of points to sample along a PC for the PC traversals
    :param output_path: str, path to the directory where we want to save the PCA resuls
    :param gmm_repr: object of class Gaussian.
    :param segmenter: object of class segmenter.
    :param device: torch device on which we perform the computations
    """
    if z.shape[-1] > 1:
        all_trajectories, all_trajectories_pca, z_pca, pca = compute_traversals(z[::thinning], dimensions=dimensions, num_points=num_points)
        sns.set_style("white")
        for dim in dimensions:
            plot_pca(output_path, dim, all_trajectories_pca, z_pca, pca)
            predicted_structures = predict_structures(vae, all_trajectories[dim], gmm_repr, segmenter, device)
            save_structures_pca(predicted_structures, dim, output_path, base_structure)

    else:
            os.makedirs(os.path.join(output_path, f"pc0/"), exist_ok=True)
            all_trajectories = graph_traversal(z, 0, num_points=num_points)
            z_dim = torch.tensor(all_trajectories, dtype=torch.float32, device=device)
            predicted_structures = predicted_structures(all_trajectories)
            save_structures_pca(predicted_structures, 0, output_path, base_structure)


def generate_structures_wrapper(rank, world_size, z, base_structure, path_structures, batch_size, gmm_repr, yaml_setting_path, model_path, segmenter_path):
    """
    Wrapper function to decode the latent variable in parallel
    :param rank: integer, rank of the device
    :param world_size: integer, number of devices
    :param z: torch.tensor(N_latent, latent_dim) latent variable from which we want to output images.
    :param vae: vae object.
    :param segmenter: segmenter object.
    """
    utils.ddp_setup(rank, world_size)
    (vae, image_translator, ctf_experiment, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, device,
    scheduler, base_structure, lp_mask2d, mask, amortized, path_results, structural_loss_parameters, segmenter)  = utils.parse_yaml(yaml_setting_path, rank, analyze=True)
    vae.load_state_dict(torch.load(model_path))
    vae.eval()
    segmenter.load_state_dict(torch.load(segmenter_path))
    segmenter.eval()
    latent_variable_dataset = LatentDataSet(z)
    generate_structures(rank, vae, segmenter, base_structure, path_structures, latent_variable_dataset, batch_size, gmm_repr)
    destroy_process_group()

def generate_structures(rank, vae, segmenter, base_structure, path_structures, latent_variable_dataset, batch_size, gmm_repr):
    vae = DDP(vae, device_ids=[rank])
    segmenter = DDP(segmenter, device_ids=[rank])
    latent_variables_loader = iter(DataLoader(latent_variable_dataset, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=False, sampler=DistributedSampler(latent_variable_dataset, shuffle=False)))
    for batch_num, (indexes, z) in enumerate(latent_variables_loader): 
        z = z.to(rank)
        predicted_structures = predict_structures(vae.module, z, gmm_repr, segmenter.module, rank)
        save_structures(predicted_structures, base_structure, batch_num, path_structures, batch_size, indexes)


def analyze(yaml_setting_path, model_path, segmenter_path, output_path, z, thinning=1, dimensions=[0, 1, 2], num_points=10, generate_structures=False):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment.
    :param model_path: str, path to the model we want to analyze.
    :param segmenter_path: str, path to the segmenter used for the analysis.
    :param structures_path: 
    :return:
    """
    (vae, image_translator, ctf_experiment, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, device,
    scheduler, base_structure, lp_mask2d, mask, amortized, path_results, structural_loss_parameters, segmenter)  = utils.parse_yaml(yaml_setting_path, gpu_id = 0, analyze=True)
    vae.load_state_dict(torch.load(model_path))
    vae.eval()
    segmenter.load_state_dict(torch.load(segmenter_path))
    segmenter.eval()
    if not os.path.exists(output_path):
            os.makedirs(output_path)

    world_size = torch.cuda.device_count()
    if z is None:
        mp.spawn(start_sample_latent, args=(world_size, yaml_setting_path, output_path, model_path, segmenter_path), nprocs=world_size)
        latent_path = os.path.join(output_path, "z.npy")
        z = np.load(latent_path)

    if not generate_structures:
        run_pca_analysis(vae, z, dimensions, num_points, output_path, gmm_repr, base_structure, thinning, segmenter, device=device)

    else:
        path_structures = os.path.join(output_path, "predicted_structures")
        if not os.path.exists(path_structures):
            os.makedirs(path_structures)

        z = torch.tensor(z, dtype=torch.float32)
        latent_variable_dataset = LatentDataSet(z)
        mp.spawn(generate_structures_wrapper, args=(world_size, z, base_structure, path_structures, batch_size, gmm_repr, yaml_setting_path, model_path, segmenter_path), nprocs=world_size)


def analyze_run():
    args = parser_arg.parse_args()
    output_path = args.output_path
    thinning = args.thinning
    model_path = args.model
    num_points = args.num_points
    path = args.experiment_yaml
    dimensions = args.dimensions
    segmenter_path = args.segmenter
    z = None
    if args.z is not None:
        z = np.load(args.z)
        
    generate_structures = args.generate_structures
    analyze(path, model_path, segmenter_path, output_path, z, dimensions=dimensions, generate_structures=generate_structures, thinning=thinning, num_points=num_points)


if __name__ == '__main__':
    analyze_run()






