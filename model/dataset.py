import os
import torch
import mrcfile
import starfile
import numpy as np
from time import time
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
#from pytorch3d.transforms import euler_angles_to_matrix, axis_angle_to_matrix
from roma import rotvec_to_rotmat, euler_to_rotmat



class Mask(torch.nn.Module):

    def __init__(self, im_size, rad):
        super(Mask, self).__init__()

        mask = torch.lt(torch.linspace(-1, 1, im_size)[None]**2 + torch.linspace(-1, 1, im_size)[:, None]**2, rad**2)
        # float for pl ddp broadcast compatible
        self.register_buffer('mask', mask.float())
        self.num_masked = torch.sum(mask).item()

    def forward(self, x):
        return x * self.mask


def starfile_reader(starfile_path, apix):
    """
    Reads a RELION starfile for the poses
    :starfile_path: str, path to the starfile
    :apix: float, size of pixel
    :return: torch.tensor(N_particles, 3, 3) or rotation poses as matrices, torch.tensor(N_particles, 3) of translations.
    """
    particles_star = starfile.read(starfile_path)
    particles_df = particles_star
    if type(particles_star) is dict and "particles" in particles_star:
        particles_df = particles_star["particles"]

    euler_angles_degrees = particles_df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].values
    euler_angles_radians = euler_angles_degrees*np.pi/180
    #poses_rotations = euler_angles_to_matrix(torch.tensor(euler_angles_radians, dtype=torch.float32), convention="ZYZ")
    poses_rotations = euler_to_rotmat(convention = "ZYZ", angles=torch.tensor(euler_angles_radians, dtype=torch.float32))
    #Transposing because ReLion has a clockwise convention, while we use a counter-clockwise convention.
    poses_rotations = torch.transpose(poses_rotations, dim0=-2, dim1=-1)

    #Reading the translations. ReLion may express the translations divided by apix. So we need to multiply by apix to recover them in Å
    if "rlnOriginXAngst" in particles_df:
        shiftX = torch.from_numpy(np.array(particles_df["rlnOriginXAngst"], dtype=np.float32))
        shiftY = torch.from_numpy(np.array(particles_df["rlnOriginYAngst"], dtype=np.float32))
    else:
        shiftX = torch.from_numpy(np.array(particles_df["rlnOriginX"] * apix, dtype=np.float32))
        shiftY = torch.from_numpy(np.array(particles_df["rlnOriginY"] * apix, dtype=np.float32))

    poses_translation = torch.tensor(torch.vstack([shiftY, shiftX]).T, dtype=torch.float32)   
    assert poses_translation.shape[0] == poses_rotations.shape[0], "Rotation and translation pose shapes are not matching !"
    return poses_rotations, poses_translation


def cs_file_reader(cs_file_path, apix, abinit, hetrefine):
    """
    Reads a cs file for the poses
    :param c_file_path: str, path to the cs file
    :param apix: float, size of a pixel
    :param abinit: boolean, whether the poses are the result of an ab-initio reconstruction or not.
    :param hetrefine: boolean, whether or not the dataset is the result of heterogeneous refinment.
    """
    data = np.load(cs_file_path)
    # view the first row
    if abinit:
        RKEY = "alignments_class_0/pose"
        TKEY = "alignments_class_0/shift"
    else:
        RKEY = "alignments3D/pose"
        TKEY = "alignments3D/shift"

    #parse rotations
    rot = np.array([x[RKEY] for x in data])
    rot = torch.tensor(rot)
    #rot_matrix = axis_angle_to_matrix(rot)
    rot_matrix = rotvec_to_rotmat(rot)
    rot_matrix = torch.transpose(rot_matrix, dim0= -2, dim1=-1)

    #parse translations
    trans = np.array([x[TKEY] for x in data])
    trans *= apix
    if hetrefine:
        trans *= 2

    #convert translations from pixels to fraction
    trans = torch.tensor(trans, dtype=torch.float32)[:, [1, 0]]
    #write output
    return rot_matrix, trans



class ImageDataSet(Dataset):
    def __init__(self, apix, side_shape, star_cs_file_config, particles_path, down_side_shape=None, down_method="interp", rad_mask=None):
        """
        Create a dataset of images and poses
        :param apix: float, size of a pixel in Å.
        :param side_shape: integer, number of pixels on each side of a picture. So the picture is a side_shape x side_shape array
        :param particle_df: particles dataframe coming from a star file
        :particles_path: string, path to the folder containing the mrcs files. It is appended to the path present in the star file.
        :param down_side_shape: integer, number of pixels of the downsampled images. If no downampling, set down_side_shape = side_shape. 
        :param down_method: str, downsampling method to use if down_side_shape < side_shape. Currently only interp is supported.
        :param rad_mask: float, radius of the mask used on the input image. If None, no mask is used.
        """

        self.side_shape = side_shape
        self.down_method = down_method
        self.apix = apix
        self.particles_path = particles_path
        self.mask = None
        if rad_mask is not None:
            self.mask = Mask(down_side_shape if down_side_shape is not None else side_shape, rad_mask)

        self.pose_file_extension = os.path.splitext(star_cs_file_config["file"])[-1].replace(".", "")
        assert self.pose_file_extension in ["cs", "star"], "Pose file must be a starfile or a cs file."
        if self.pose_file_extension == "star":
            self.poses, self.poses_translation = starfile_reader(star_cs_file_config["file"], self.apix)
            self.particles_df = starfile.read(star_cs_file_config["file"])
            if type(self.particles_df) is dict and "particles" in self.particles_df:
                self.particles_df = self.particles_df["particles"]
        else:
            self.poses, self.poses_translation = cs_file_reader(star_cs_file_config["file"], self.apix, star_cs_file_config.get("abinit", False), 
                                                                star_cs_file_config.get("hetrefine", False))
            self.particles_df = np.load(star_cs_file_config["file"])


        print("Dataset size:", self.poses.shape[0], "apix:",self.apix)
        print("Normalizing training data")

        #If a downsampling is wanted, recompute the new apix and set the new down_side_shape
        self.down_side_shape = side_shape
        self.down_apix = apix
        if down_side_shape is not None:
            self.down_side_shape = down_side_shape
            self.down_apix = self.side_shape * self.apix /self.down_side_shape

        self.f_std = None
        self.f_mu = None
        self.estimate_normalization()

    def estimate_normalization(self):
        if self.f_mu is None and self.f_std is None:
            f_sub_data = []
            # I have checked that the standard deviation of 10/100/1000 particles is similar
            for i in range(0, len(self), len(self) // 100):
                _, _, _, _, fproj = self[i]
                f_sub_data.append(fproj)

            f_sub_data = torch.cat(f_sub_data, dim=0)
            self.f_mu = 0.0  # just follow cryodrgn
            self.f_std = torch.std(f_sub_data).item()
            print("Estimated std", self.f_std)
        else:
            raise Exception("The normalization factor has been estimated!")

    def standardize(self, images, device="cpu"):
        return (images - self.avg_image.to(device))/self.std_image.to(device)

    def __len__(self):
        return self.particles_df.shape[0]

    def __getitem__(self, idx):
        """
        #Return a batch of true images, as 2d array
        # return: the set of indexes queried for the batch, the corresponding images as a torch.tensor((batch_size, side_shape, side_shape)), 
        # the corresponding poses rotation matrices as torch.tensor((batch_size, 3, 3)), the corresponding poses translations as torch.tensor((batch_size, 2))
        # NOTA BENE: the convention for the rotation matrix is left multiplication of the coordinates of the atoms of the protein !!
        """
        #try:
        if self.pose_file_extension == "star":
            particles = self.particles_df.iloc[idx]
            mrc_idx, img_name = particles["rlnImageName"].split("@")
            mrc_idx = int(mrc_idx) - 1
        else:
            particles = self.particles_df[idx]
            mrc_idx = particles["blob/idx"]
            if type(idx) != list:
                img_name = particles["blob/path"].decode('ascii').replace(">", "")
            else:
                img_name = [s.decode('ascii').replace(">", "") for s in particles["blob/path"]]

        mrc_path = os.path.join(self.particles_path, img_name)
        with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
            if mrc.data.ndim > 2:
                proj = torch.from_numpy(np.array(mrc.data[mrc_idx])).float() #* self.cfg.scale_images
            else:
                # the mrcs file can contain only one particle
                proj = torch.from_numpy(np.array(mrc.data)).float() #* self.cfg.scale_images

        # get (1, side_shape, side_shape) proj
        if len(proj.shape) == 2:
            proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)
        else:
            assert len(proj.shape) == 3 and proj.shape[0] == 1  # some starfile already have a dummy channel

        if self.down_side_shape != self.side_shape:
            if self.down_method == "interp":
                proj = tvf.resize(proj, [self.down_side_shape, ] * 2, antialias=True)
            #elif self.down_method == "fft":
            #    proj = downsample_2d(proj[0, :, :], self.down_side_shape)[None, :, :]
            else:
                raise NotImplementedError            

        proj = proj[0]
        if self.mask is not None:
            proj = self.mask(proj)

        fproj = primal_to_fourier_2d(proj)
        if self.f_mu is not None:
            fproj = (fproj - self.f_mu) / self.f_std
            proj = fourier_to_primal_2d(fproj).real

        return idx, proj, self.poses[idx], self.poses_translation[idx]/self.down_apix, fproj



def primal_to_fourier_2d(images):
    """
    Computes the fourier transform of the images.
    images: torch.tensor(batch_size, N_pix, N_pix)
    return fourier transform of the images
    """
    r = torch.fft.ifftshift(images, dim=(-2, -1))
    fourier_images = torch.fft.fftshift(torch.fft.fft2(r, dim=(-2, -1), s=(r.shape[-2], r.shape[-1])), dim=(-2, -1))
    return fourier_images

def fourier_to_primal_2d(fourier_images):
    """
    Computes the inverse fourier transform
    fourier_images: torch.tensor(batch_size, N_pix, N_pix)
    return fourier transform of the images
    """
    f = torch.fft.ifftshift(fourier_images, dim=(-2, -1))
    r = torch.fft.fftshift(torch.fft.ifft2(f, dim=(-2, -1), s=(f.shape[-2], f.shape[-1])),dim=(-2, -1)).real
    return r



