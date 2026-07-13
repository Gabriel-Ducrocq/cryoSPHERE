import torch
import numpy as np
from torch import linalg as LA
import torch.nn.functional as F
from scipy.spatial import distance



AA_ATOMS = ("CA", )
NT_ATOMS = ("C1'", )


def compute_linear_beta_schedule(epoch, M):
    """
    Compute a beta for a KL divergence according to a linear schedule. See paper:
    https://arxiv.org/pdf/1903.10145
    :param epoch: integer, current epoch.
    :param M: epoch at which beta reaches 1.
    return float, beta
    """
    return min(epoch/M, 1.0)


def compute_cyclical_beta_schedule(epoch, M, N_epochs, R=0.5):
    """
    Compute a beta for a KL divergence according to a schedule. See paper:
    https://arxiv.org/pdf/1903.10145
    :param epoch: integer, epoch number
    :param R: float in [0, 1], proportion of the cycle spent at beta != 1 for cyclical annealing. For 
    :param M: integer, number of cycles for the cyclical scheduling.
    :param N_epochs: integer, total number of epochs we are training for
    return float, beta value
    """
    tau = (epoch % np.ceil(N_epochs/M)) / (N_epochs/M)
    if tau < R:
        return tau/R
    else:
        return 1

def compute_beta_schedule(epoch, N_epochs, loss_parameters):
    """
    Computes the beta value for the KL divergence according to the appropriate schedule.
    :param epoch: integer, epoch number
    :param N_epochs: integer, total number of epochs we are training for
    :param loss_parameters: dictionnary containing the loss parameters, such as the beta value if constant schedule, or the parameters for the other schedules.
    return float, beta value
    """
    schedule = loss_parameters["schedule"]
    assert schedule in ["constant", "linear", "cyclical"], f"Schedule must be constant, linear or cyclical. {schedule} value is not handled"
    if schedule == "constant":
        return loss_parameters.get("beta", 1)
    elif schedule == "linear": 
        M = loss_parameters.get("M", N_epochs//2)
        return compute_linear_beta_schedule(epoch, M)
    else:
        M = loss_parameters.get("M", N_epochs//10)
        R = loss_parameters.get("R", 0.5)
        return compute_cyclical_beta_schedule(epoch, M, N_epochs, R)

def compute_all_beta_schedule(epoch, N_epochs, all_losses_parameters):
    """
    Computes the beta for all the loss terms according to their own respective schedules
    :param epoch: integer, epoch number
    :param N_epochs: integer, total number of epochs to perform
    return dictionnary of all beta values.
    """
    all_beta_values = {}
    for loss_name, loss_parameters in all_losses_parameters.items():
        all_beta_values[loss_name] = compute_beta_schedule(epoch, N_epochs, loss_parameters)

    return all_beta_values



def calc_clash_loss(pred_struc, pair_index, clash_cutoff=4.0):
    """
    Computes the clashing loss for the predicted structures. The clashing loss is applied to the pairs for which the distance is inferior to a cutoff.
    :param pred_struc: torch.tensor(N_batch, N_residues, 3)
    :param pair_index: torch.tensor(N_pairs, 2) of pairs of residues for which the clashing loss should be computed. 
    return: torch.tensor(1) of averaged over batch clashing loss.
    """
    pred_dist = pred_struc[:, pair_index]  # bsz, num_pair, 2, 3
    pred_dist = LA.vector_norm(torch.diff(pred_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
    possible_clash_dist = pred_dist[pred_dist < clash_cutoff]
    if possible_clash_dist.numel() == 0:
        avg_loss = torch.tensor(0.0).to(pred_struc)
    else:
        possible_clash_loss = (clash_cutoff - possible_clash_dist)**2
        avg_loss = possible_clash_loss.mean()
    return avg_loss


def calc_pair_dist_loss(pred_struc, pair_index, target_dist):
    """
    Computes the continuity loss, for a set of defined pairs and the corresponding distances in the base structure.
    :param pred_struc: torch.tensor(N_batch, N_residues, 3) predicted structures
    :param pair_index: torch.tensor(N_pairs, 2), pairs of residues for which we compute the continuity loss
    :param target_dist: torch.tensor(N_pairs), distances between the two residues of each corresponding pair.
    """
    bsz = pred_struc.shape[0]
    pred_dist = pred_struc[:, pair_index]  # bsz, num_pair, 2, 3
    pred_dist = LA.vector_norm(torch.diff(pred_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
    return F.mse_loss(pred_dist, target_dist.repeat(bsz, 1))


def calc_dist_by_pair_indices(coord_arr, pair_indices):
    """
    Compute the distances between pairs of residues.
    :param coord_arr: np.array(N_residues, 3)
    """
    coord_pair_arr = coord_arr[pair_indices]  # num_pair, 2, 3
    dist = np.linalg.norm(np.diff(coord_pair_arr, axis=1), ord=2, axis=-1)
    return dist.flatten()




def find_continuous_pairs(chain_id_arr, res_id_arr, atom_name_arr):
    """
    We fnd pairs of subsequent residues (residue_index, residue_index + 1) in the protein. Note that we do not consider two residues subsequent if they do not belong to the
    same chain.
    :param chain_id_arr: np.array(N_residues), chain_id of each residue
    :param res_id_arr: np.array(N_residues), residue index of each residue
    :param atom_name_arr: np.array(N_residues), atom name for each residue (in that case C_alpha )
    return np.array(N_pairs, 2) array of residues indexes paired (the indexes are expressed as part of the whole protein, not within the chain).
    """
    pairs = []

    # res_id in different chains are duplicated, so loop on chains
    u_chain_id = np.unique(chain_id_arr)

    #For each chain
    for c_id in u_chain_id:
        #Identify which residue belong to that chain
        tmp_mask = chain_id_arr == c_id
        tmp_indices_in_pdb = np.nonzero(tmp_mask)[0]

        #Get these residues and their indexes
        tmp_res_id_arr = res_id_arr[tmp_mask]
        tmp_atom_name_arr = atom_name_arr[tmp_mask]

        # check is aa or nt
        tmp_atom_name_set = set(tmp_atom_name_arr)

        if len(tmp_atom_name_set.intersection(AA_ATOMS)) > len(tmp_atom_name_set.intersection(NT_ATOMS)):
            in_res_atom_names = AA_ATOMS
        elif len(tmp_atom_name_set.intersection(AA_ATOMS)) < len(tmp_atom_name_set.intersection(NT_ATOMS)):
            in_res_atom_names = NT_ATOMS
        else:
            raise NotImplemented("Cannot determine chain is amino acid or nucleotide.")

        # find pairs
        if len(in_res_atom_names) == 1:
            #Get the unique residue indices as well as their indexes array of that chain
            u_res_id, indices_in_chain = np.unique(tmp_res_id_arr, return_index=True)
            if len(u_res_id) != np.sum(tmp_mask):
                raise ValueError(f"Found duplicate residue id in single chain {c_id}.")

            #Pair each residue with the following one and stack their indices column wise
            indices_in_chain_pair = np.column_stack((indices_in_chain[:-1], indices_in_chain[1:]))

            # must be adjacent on residue id
            valid_mask = np.abs(np.diff(u_res_id[indices_in_chain_pair], axis=1)) == 1

            #Keep only the pairs that have actually subsequent indices in the chain.
            indices_in_chain_pair = indices_in_chain_pair[valid_mask.flatten()]

            #Get their inndexes in the entire PDB, not only within the chain
            indices_in_pdb_pair = tmp_indices_in_pdb[indices_in_chain_pair]
        elif len(in_res_atom_names) > 1:

            def _cmp(a, b):
                # res_id compare
                if a[0] != b[0]:
                    return a[0] - b[0]
                else:
                    # atom_name in the same order of AA_ATOMS or NT_ATOMS
                    return in_res_atom_names.index(a[1]) - in_res_atom_names.index(b[1])

            cache = list(zip(tmp_res_id_arr, tmp_atom_name_arr, tmp_indices_in_pdb))
            sorted_cache = list(sorted(cache, key=cmp_to_key(_cmp)))

            sorted_indices_in_pdb = [item[2] for item in sorted_cache]
            sorted_res_id = [item[0] for item in sorted_cache]

            indices_in_pdb_pair = np.column_stack((sorted_indices_in_pdb[:-1], sorted_indices_in_pdb[1:]))

            valid_mask = np.abs(np.diff(np.column_stack((sorted_res_id[:-1], sorted_res_id[1:])), axis=1)) <= 1

            indices_in_pdb_pair = indices_in_pdb_pair[valid_mask.flatten()]
        else:
            raise NotImplemented("No enough atoms to construct continuous pairs.")

        pairs.append(indices_in_pdb_pair)

    pairs = np.vstack(pairs)
    return pairs

def find_range_cutoff_pairs(coord_arr, min_cutoff=4., max_cutoff=10.):
    """
    For big protein, compute the clashing loss for every residue is computationnally too expensive. In this case, we only compute it for pairs of residues
    which had a distance in the base structure in a certain interval.
    :param coord_arr: np.array(N_residues, 3) coordinates of each residue.
    :param min_cutoff: minimum cutoff to consider the pair for the clashing loss
    :param max_cutoff: maximum cutoff to consider the pair the clashing loss
    return np.array(N_pairs, 2) pairs of residues considered for the clashing loss
    """
    dist_map = distance.cdist(coord_arr, coord_arr, metric='euclidean')
    sel_mask = (dist_map <= max_cutoff) & (dist_map >= min_cutoff)
    indices_in_pdb = np.nonzero(sel_mask)
    indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
    return indices_in_pdb


def remove_duplicate_pairs(pairs_a, pairs_b, remove_flip=True):
    """
    We do not want to keep the same residues in the clashing loss and continuity loss, since this can introduce contradictory requirements.
    This function remove the pairs in a that are also in b.
    :param pairs_a: np.array(N_pairs_a, 2)
    :param pairs_b: np.array(N_pairs_a, 2)
    return np.array(N_pairs_in_a_but_not_in_b, 2)
    """
    """Remove pair b from a"""
    s = max(pairs_a.max(), pairs_b.max()) + 1
    # trick for fast comparison
    mask = np.zeros((s, s), dtype=bool)

    #np.ravel_multi_index gets the index of the elements in non linear shape as if the array was linear
    #so ravel_multi_index(pairs_a.T, mask.shape) get the indexes of the elements in the pair as if mask was linear
    #So the next line sets all the values of the mask array where the indexes are in pairs_a to True
    np.put(mask, np.ravel_multi_index(pairs_a.T, mask.shape), True)
    #This line set all the values of the mask array where the indexes are in pairs_b to False. This step is needed so that pairs in a that are also
    #in b are set to False
    np.put(mask, np.ravel_multi_index(pairs_b.T, mask.shape), False)
    if remove_flip:
        #This line does the same thing except we first flip the coordinates in pairs_b, so we get both (x, y) and (y, x) to False
        np.put(mask, np.ravel_multi_index(np.flip(pairs_b, 1).T, mask.shape), False)

    #Finally, we return the non False elements, e.g the pairs in a that are not in b.
    return np.column_stack(np.nonzero(mask))


def calc_cor_loss(pred_images, gt_images, mask=None):
    """
    Compute the cross-correlation for each pair (predicted_image, true) image in a batch. And average them
    pred_images: torch.tensor(batch_size, side_shape**2) predicted images
    gt_images: torch.tensor(batch_size, side_shape**2) of true images, translated according to the poses.
    return torch.tensor(1) of average correlation accross the batch.
    """
    if mask is not None:
        pred_images = mask(pred_images)
        gt_images = mask(gt_images)
        pixel_num = mask.num_masked
    else:
        pixel_num = pred_images.shape[-2] * pred_images.shape[-1]

    pred_images = torch.flatten(pred_images, start_dim=-2, end_dim=-1)
    gt_images = torch.flatten(gt_images, start_dim=-2, end_dim=-1)
    # b, h, w -> b, num_pix
    #pred_images = pred_images.flatten(start_dim=2)
    #gt_images = gt_images.flatten(start_dim=2)

    # b 
    dots = (pred_images * gt_images).sum(-1)
    # b -> b 
    err = -dots / (gt_images.std(-1) + 1e-5) / (pred_images.std(-1) + 1e-5)
    # b -> 1 value
    err = err.mean() / pixel_num
    return err

def compute_KL_prior_latent(latent_mean, latent_std, epsilon_loss):
    """
    Computes the KL divergence between the approximate posterior and the prior over the latent variable z,
    where the latent prior is given by a standard Gaussian distribution.
    :param latent_mean: torch.tensor(N_batch, latent_dim), mean of the Gaussian approximate posterior
    :param latent_std: torch.tensor(N_batch, latent_dim), std of the Gaussian approximate posterior
    :param epsilon_loss: float, a constant added in the log to avoid log(0) situation.
    :return: torch.float32, average of the KL losses accross batch samples
    """
    return torch.mean(-0.5 * torch.sum(1 + torch.log(latent_std ** 2 + eval(epsilon_loss)) \
                                           - latent_mean ** 2 \
                                           - latent_std ** 2, dim=1))


def compute_KL_prior_segments(segmenter, segments_prior, variable, epsilon_kl):
    """
    Compute the KL divergence loss between the prior and the approximated posterior distribution for the segmentation.
    :param segmenter: object of class Segmentation.
    :param segments_prior: dictionnary, containing the tensor of segmentation prior
    :return: torch.float32, KL divergence loss
    """
    all_kl_losses = 0
    assert variable in ["means", "stds", "proportions"]
    for part in segmenter.segments_means_stds:
        if variable == "means":
            segments_stds = segmenter.segments_means_stds[part]
            segments_means = segmenter.segments_means_means[part]
        elif variable == "stds":
            segments_stds = segmenter.segments_stds_stds[part]
            segments_means = segmenter.segments_stds_means[part]
        else:
            segments_stds = segmenter.segments_proportions_stds[part]
            segments_means = segmenter.segments_proportions_means[part]           

        all_kl_losses += torch.sum(-1/2 + torch.log(segments_prior[part][variable]["std"]/segments_stds + eval(epsilon_kl)) \
                    + (1/2)*(segments_stds**2 + (segments_prior[part][variable]["mean"] - segments_means)**2)/segments_prior[part][variable]["std"]**2)

    return all_kl_losses



def compute_l2_pen(network):
    """
    Compute the l2 norm of the network's weight
    :param network: torch.nn.Module
    :return: torch.float32, l2 squared norm of the network's weights
    """
    l2_pen = 0
    for name, p in network.named_parameters():
        if "weight" in name and ("encoder" in name or "decoder" in name):
            l2_pen += torch.sum(p ** 2)

    return l2_pen

def compute_clashing_distances(new_structures, device, cutoff=4):
    """
    Computes the clashing distance loss. The cutoff is set to 4Å for non contiguous residues and the distance above this cutoff
    are not penalized
    Computes the distances between all the atoms
    :param new_structures: torch.tensor(N_batch, N_residues, 3), atom positions
    :return: torch.tensor(1, ) of the averaged clashing distance for distance inferior to cutoff,
    reaverage over the batch dimension
    """
    N_residues = new_structures.shape[1]
    distances = torch.cdist(new_structures, new_structures)
    triu_indices = torch.triu_indices(N_residues, N_residues, offset=2, device=device)
    distances = distances[:, triu_indices[0], triu_indices[1]]
    number_clash_per_sample = torch.sum(distances < cutoff, dim=-1)
    distances = torch.minimum((distances - cutoff), torch.zeros_like(distances))**2
    average_clahing = torch.sum(distances, dim=-1)/number_clash_per_sample
    return torch.mean(average_clahing)


def compute_loss(predicted_images, images, segmentation_image, latent_mean, latent_std, vae, segmenter, experiment_settings, tracking_dict, structural_loss_parameters,
                 epoch, predicted_structures = None, device=None):
    """
    Compute the entire loss
    :param predicted_images: torch.tensor(batch_size, N_pix), predicted images
    :param images: torch.tensor(batch_size, N_pix), true images
    :param latent_mean:torch.tensor(batch_size, latent_dim), mean of the approximate latent distribution
    :param latent_std:torch.tensor(batch_size, latent_dim), std of the approximate latent distribution
    :param segmenter: object of the class VAE.
    :param segmenter: object of the class Segmentation.
    :param experiment_settings: dictionnary with the settings of the current experiment
    :param tracking_dict: dictionnary containing the different metrics we want to track
    :param structural_loss_parameters: dictionnary containing all that is required to compute the structural loss, such as the pairs for clashing loss, continuity loss and
                                        the target distances.
    :param predicted_structures: torch.tensor(N_batch, N_residues, 3) of predicted structures to compute the structural losses.
    :param device: torch device on which we perform the computations.
    :return: torch.float32, average loss over the batch dimension
    """
    rmsd = calc_cor_loss(predicted_images, images, segmentation_image)
    KL_prior_latent = compute_KL_prior_latent(latent_mean, latent_std, experiment_settings["epsilon_kl"])
    KL_prior_segmentation_means = compute_KL_prior_segments(
        segmenter, experiment_settings["segmentation_prior"],
        "means", epsilon_kl=experiment_settings["epsilon_kl"])

    continuity_loss = calc_pair_dist_loss(predicted_structures, structural_loss_parameters["connect_pairs"], 
        structural_loss_parameters["connect_distances"])

    if structural_loss_parameters["clash_pairs"] is None:
        clashing_loss = compute_clashing_distances(predicted_structures, device, cutoff=experiment_settings["loss"]["clashing_loss"]["clashing_cutoff"])
    else:
        clashing_loss =  calc_clash_loss(predicted_structures, structural_loss_parameters["clash_pairs"], clash_cutoff=experiment_settings["loss"]["clashing_loss"]["clashing_cutoff"])

    KL_prior_segmentation_stds = compute_KL_prior_segments(segmenter, experiment_settings["segmentation_prior"],
                                               "stds", epsilon_kl=experiment_settings["epsilon_kl"])
    KL_prior_segmentation_proportions = compute_KL_prior_segments(segmenter, experiment_settings["segmentation_prior"],
                                               "proportions", epsilon_kl=experiment_settings["epsilon_kl"])
    l2_pen = compute_l2_pen(vae)


    loss_weights = compute_all_beta_schedule(epoch, experiment_settings["N_epochs"], experiment_settings["loss"])

    pixel_num = predicted_images.shape[-1]*predicted_images.shape[-2]
    tracking_dict["correlation_loss"].append(rmsd.detach().cpu().numpy())
    tracking_dict["kl_prior_latent"].append(KL_prior_latent.detach().cpu().numpy())
    tracking_dict["kl_prior_segmentation_mean"].append(KL_prior_segmentation_means.detach().cpu().numpy())
    tracking_dict["kl_prior_segmentation_std"].append(KL_prior_segmentation_stds.detach().cpu().numpy())
    tracking_dict["kl_prior_segmentation_proportions"].append(KL_prior_segmentation_proportions.detach().cpu().numpy())
    tracking_dict["l2_pen"].append(l2_pen.detach().cpu().numpy())
    tracking_dict["continuity_loss"].append(continuity_loss.detach().cpu().numpy())
    tracking_dict["clashing_loss"].append(clashing_loss.detach().cpu().numpy())
    tracking_dict["clashing_loss"].append(clashing_loss.detach().cpu().numpy())
    tracking_dict["betas"] = loss_weights

    loss = rmsd + loss_weights["KL_prior_latent"]*KL_prior_latent/pixel_num \
           + loss_weights["KL_prior_segmentation_mean"]*KL_prior_segmentation_means/pixel_num \
           + loss_weights["KL_prior_segmentation_std"] * KL_prior_segmentation_stds/pixel_num \
           + loss_weights["KL_prior_segmentation_proportions"] * KL_prior_segmentation_proportions/pixel_num \
           + loss_weights["l2_pen"] * l2_pen \
           + loss_weights["continuity_loss"]*continuity_loss \
           + loss_weights["clashing_loss"]*clashing_loss

    return loss
