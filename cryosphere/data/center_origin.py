import mrcfile
import argparse
import numpy as np
import os.path as osp
from pathlib import Path
import biotite.structure as bt_struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import get_structure


#This code has been taken from the cryoStar repository:
#https://github.com/bytedance/cryostar/


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--pdb_file_path', type=str, required=True)
parser_arg.add_argument('--mrc_file_path', type=str, required=True)


def _get_file_ext(file_path):
    if isinstance(file_path, Path):
        return file_path.suffix
    else:
        return osp.splitext(file_path)[1]


def _get_file_name(file_path):
    return osp.splitext(osp.basename(file_path))[0]

def save_mrc(vol, path, voxel_size = None, origin = None):
    """
    Save volumetric data to mrc file, set voxel_size, origin.
    See Also: https://mrcfile.readthedocs.io/en/stable/source/mrcfile.html#mrcfile.mrcobject.MrcObject.voxel_size
    Args:
        vol: density volume
        path: save path
        voxel_size: a single number, a 3-tuple (x, y ,z) or a modified version of the voxel_size array, default 1.
        origin: a single number, a 3-tuple (x, y ,z) or a modified version of the origin array, default 0.

    """
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(vol)

        if voxel_size is not None:
            m.voxel_size = voxel_size

        if origin is not None:
            m.header.origin = origin


def bt_read_pdb(file_path):
    """Read pdb file by biotite, return all models as AtomArrayStack

    Parameters
    ----------
    file_path: pdb file path

    Returns
    -------
    atom_arr_stack: biotite AtomArrayStack containing all models

    """
    file_ext = _get_file_ext(file_path)
    if file_ext == ".pdb":
        f = PDBFile.read(file_path)
        atom_arr_stack = f.get_structure()
    elif file_ext == ".cif":
        f = PDBxFile.read(file_path)
        atom_arr_stack = get_structure(f)
    else:
        raise NotImplementedError("Only support .pdb, .cif extension.")
    return atom_arr_stack


def bt_save_pdb(file_path, array, **kwargs):
    """Save biotite AtomArray or AtomArrayStack to pdb file

    Parameters
    ----------
    file_path: save file path
    array: the structure to be saved
    kwargs: additional parameters to be passed, always empty

    """
    bt_struc.io.save_structure(file_path, array, **kwargs)


def center_origin(pdb_file_path, mrc_file_path):
    """
    Centers the origin of PDB and MRC file

    This function moves the origin of coordinates for both PDB and MRC files to the
    center of the MRC three-dimensional data matrix, so that the center of the 3D
    data matrix becomes (0,0,0). It then saves the adjusted files in the current
    directory with a '_centered' suffix.

    Usage:
    center_origin <reference_structure_path.pdb> <consensus_map_path.mrc>

    :param pdb_file_path: str, path to the pdb file
    :param mrc_file_path: str, path to the mrc file
    """
    with mrcfile.open(mrc_file_path) as m:
        if m.voxel_size.x == m.voxel_size.y == m.voxel_size.z and np.all(np.asarray(m.data.shape) == m.data.shape[0]):
            new_origin = (- m.data.shape[0] // 2 * m.voxel_size.x, ) * 3
        else:
            print("The voxel sizes or shapes differ across the three axes in the three-dimensional data.")
            new_origin = (- m.data.shape[2] // 2 * m.voxel_size.x, - m.data.shape[1] // 2 * m.voxel_size.y,
                          - m.data.shape[0] // 2 * m.voxel_size.z)
        save_mrc(m.data.copy(), _get_file_name(mrc_file_path) + "_centered.mrc",
                 m.voxel_size, new_origin)
        print(f"Result centered MRC saved to {_get_file_name(mrc_file_path)}_centered.mrc.")

        atom_arr = bt_read_pdb(pdb_file_path)[0]
        atom_arr.coord += np.asarray(new_origin)
        bt_save_pdb(_get_file_name(pdb_file_path) + "_centered.pdb", atom_arr)
        print(f"Result centered PDB saved to {_get_file_name(pdb_file_path)}_centered.pdb.")

def run_center_origin():
    args = parser_arg.parse_args()
    pdb_file_path = args.pdb_file_path
    mrc_file_path = args.mrc_file_path
    center_origin(pdb_file_path, mrc_file_path)


if __name__ == '__main__':
    run_center_origin()
