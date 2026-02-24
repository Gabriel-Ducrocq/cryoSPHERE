import sys
import os
import cryosphere.data.mrc as mrc
path = os.path.abspath("model")
sys.path.append(path)
import yaml
import torch
import mrcfile
import argparse
import numpy as np
from cryosphere.model import renderer
import time
from cryosphere.model.polymer import Polymer
from cryosphere.model import utils
from cryosphere.model.gmm import Gaussian, EMAN2Grid, BaseGrid




def structure_to_volume(image_yaml, structure_path, output_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(image_yaml, "r") as file:
        image_settings = yaml.safe_load(file)

    apix = image_settings["apix"]
    Npix = image_settings["Npix"]
    Npix_downsize = image_settings["Npix_downsize"]
    apix_downsize = apix*Npix/Npix_downsize

    base_structure = Polymer.from_pdb(structure_path, True)
    amplitudes = torch.tensor(base_structure.num_electron, dtype=torch.float32, device=device)[:, None]
    grid = EMAN2Grid(Npix_downsize, apix_downsize, device=device)
    #grid = BaseGrid(Npix_downsize, apix_downsize, device=device)
    gmm_repr = Gaussian(torch.tensor(base_structure.coord, dtype=torch.float32, device=device), 
            torch.ones((base_structure.coord.shape[0], 1), dtype=torch.float32, device=device)*image_settings["sigma_gmm"], 
            amplitudes)
    start = time.time()
    struct = gmm_repr.mus[None, :, :]
    origin = - Npix_downsize // 2 * apix_downsize
    volume = renderer.structure_to_volume(struct, gmm_repr.sigmas, gmm_repr.amplitudes, grid, device)
    mrc.MRCFile.write(output_path, np.transpose(volume[0].detach().cpu().numpy(), axes=(2, 1, 0)), Apix=apix_downsize, is_vol=True, xorg=origin, yorg=origin, zorg=origin)

def turn_structure_to_volume():
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--image_yaml', type=str, required=True, help="path to the yaml containing the images informations.")
    parser_arg.add_argument("--structure_path", type=str, required=True, help="path to the pdb file we want to turn into a volume.")
    parser_arg.add_argument("--output_path", type=str, required=True, help="path of the output mrc file containing the volume.")
    args = parser_arg.parse_args()
    image_yaml = args.image_yaml
    structure_path = args.structure_path
    output_path = args.output_path
    structure_to_volume(image_yaml, structure_path, output_path)

if __name__ == '__main__':
    turn_structure_to_volume()




