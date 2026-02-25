import gemmi
import numpy as np
import dataclasses
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

# careful about the order
AA_ATOMS = ("CA", )
NT_ATOMS = ("C1'", )


def get_num_electrons(atom_arr):
    return np.sum(np.array([gemmi.Element(x).atomic_number for x in atom_arr.element]))


@dataclasses.dataclass
class Polymer:
    chain_id: np.ndarray
    res_id: np.ndarray
    res_name: np.ndarray
    coord: np.ndarray
    atom_name: np.ndarray
    element: np.ndarray
    num_electron: np.ndarray

    def __init__(self, num):
        """
        This class defines the base structure. cryoSPHERE considers only the C_alpha atoms of the protein.
        :param num: number of residues

        """
        self.chain_id = np.empty(num, dtype="U4")
        self.res_id = np.zeros(num, dtype=int)
        self.res_name = np.empty(num, dtype="U3")
        self.coord = np.zeros((num, 3), dtype=np.float32)
        self.atom_name = np.empty(num, dtype="U6")
        self.element = np.empty(num, dtype="U2")
        self.num_electron = np.zeros(num, dtype=int)

    def __setitem__(self, index, kwargs):
        assert set(kwargs.keys()).issubset(f.name for f in dataclasses.fields(self))
        for k, v in kwargs.items():
            getattr(self, k)[index] = v

    def __getitem__(self, index):
        return {f.name: getattr(self, f.name)[index] for f in dataclasses.fields(self)}

    def __len__(self):
        return len(self.chain_id)

    @property
    def num_amino_acids(self):
        """
        Computes the number of C_alpha atoms in the structure.
        """
        return np.sum(np.isin(self.atom_name, AA_ATOMS))

    @property
    def num_nucleotides(self):
        """
        Computes the number of nucleotides in the structure.
        """
        return np.sum(np.isin(self.atom_name, NT_ATOMS))

    @property
    def num_chains(self):
        """
        Computes the number of chains in the structure.
        """
        return len(np.unique(self.chain_id))

    @classmethod
    def from_atom_arr(cls, atom_arr, filter_aa=True):
        """
        Create an instance of the class based on a biotite atom array
        :param atom_arr: biotite atom array
        :param filter_aa: bool, True to keep only the amino-acids.
        """
        assert isinstance(atom_arr, struc.AtomArray)

        #Get an atom array made of nucleotide only
        nt_arr = atom_arr[struc.filter_nucleotides(atom_arr)]
        aa_arr = atom_arr
        #Get an atom array of amino acids only.
        if filter_aa:
            aa_arr = atom_arr[struc.filter_amino_acids(atom_arr)]

        #Get the number of residues.
        num = 0
        if len(aa_arr) > 0:
            num += struc.get_residue_count(aa_arr)
        if len(nt_arr) > 0:
            for res in struc.residue_iter(nt_arr):
                valid_atoms = set(res.atom_name).intersection(NT_ATOMS)
                if len(valid_atoms) <= 0:
                    raise UserWarning(f"Nucleotides doesn't contain {' or '.join(NT_ATOMS)}.")
                else:
                    num += len(valid_atoms)
        meta = cls(num)

        def _update_res(tmp_res, kind="aa"):
            """
            Updates the information of a residue.
            :param tmp_res: atom_array corresponding to a specific residue.
            :param kind: str, either "aa" for amino acid or "nt" for nucleotide.
            """
            nonlocal pos

            if kind == "aa":
                using_atom_names = AA_ATOMS
                #Get only the backbones atoms.
                filtered_res = tmp_res[struc.filter_peptide_backbone(tmp_res)]
                filtered_res = tmp_res
            elif kind == "nt":
                using_atom_names = NT_ATOMS
                filtered_res = tmp_res
            else:
                raise NotImplemented

            #Filter out the valid atom names only, in the case of a protein we keep only the C_alpha atom.  
            valid_atom_names = set(tmp_res.atom_name).intersection(using_atom_names)
            #For each valid atom, we set its characteristics: its chain name, its residue id, , its residue name, its coordinates, its atom name and the average number of electrons
            #in that residue.
            for select_atom_name in valid_atom_names:
                meta[pos] = {
                    "chain_id": tmp_res.chain_id[0],
                    "res_id": tmp_res.res_id[0],
                    "res_name": tmp_res.res_name[0],
                    "coord": filtered_res[filtered_res.atom_name == select_atom_name].coord,
                    "atom_name": select_atom_name,
                    "element": filtered_res[filtered_res.atom_name == select_atom_name].element[0],
                    "num_electron": get_num_electrons(tmp_res) // len(valid_atom_names)
                }
                pos += 1

        def _update(tmp_arr, kind="aa"):
            """
            Given an biotite atom_array, compute the informations of this structure
            :param tmp_arr: biotite atom_array
            """
            nonlocal pos
            #Iterate over the chains
            for chain in struc.chain_iter(tmp_arr):
                #Iterate over the residues
                for tmp_res in struc.residue_iter(chain):
                    #Update the informations for that residue
                    _update_res(tmp_res, kind)

        pos = 0

        if len(aa_arr) > 0:
            _update(aa_arr, kind="aa")
        if len(nt_arr) > 0:
            _update(nt_arr, kind="nt")

        ## REMOVING ASSERT HERE !
        #assert pos == num
        return meta

    @classmethod
    def from_pdb(cls, file_path, filter_aa=True):
        """
        Returns a class instances from a pdb file
        :param file_path: str, path to the pdb file
        :param filter_aa: bool, True to keep only the amino-acids.
        """
        f = PDBFile.read(file_path)
        atom_arr_stack = f.get_structure()
        if atom_arr_stack.stack_depth() > 1:
            print("PDB file contains more than 1 models, select the 1st model")
        atom_arr = atom_arr_stack[0]
        return Polymer.from_atom_arr(atom_arr, filter_aa)

    def to_pdb(self, file_path):
        """
        Save the Polymer structure to pdb
        file_path: str, path to save the pdb file
        """
        file = PDBFile()
        file.set_structure(self.to_atom_arr())
        file.write(file_path)

    def to_atom_arr(self):
        """
        Creates an atom array from a polymer object.
        """
        num = len(self)
        atom_arr = struc.AtomArray(num)
        atom_arr.coord = self.coord

        for f in dataclasses.fields(self):
            if f.name != "coord" and f.name in atom_arr.get_annotation_categories():
                atom_arr.set_annotation(f.name, getattr(self, f.name))
        # atom_arr.atom_name[atom_arr.atom_name == "R"] = "CB"
        return atom_arr


    def translate_structure(self, translation_vector):
        """
        Translate the structure by the translation vector
        translation_vector: np.array(None, 3)
        """
        self.coord += translation_vector


