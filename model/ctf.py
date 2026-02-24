import torch
import starfile
import numpy as np
import matplotlib.pyplot as plt


class CTF(torch.nn.Module):
	"""
	Class describing the ctf, built from starfile
	"""
	def __init__(self, side_shape, apix, defocusU, defocusV, defocusAngle, voltage, sphericalAberration, amplitudeContrastRatio, phaseShift=None ,scalefactor = None,
		bfactor= None, device="cpu"):
		"""
		side shape: number of pixels on a side.
		apix: size of a pixel in Å.
		defocusU: defocusU.
		defocusV: defocusV.
		defocusAngle: defocus angle in degrees.
		voltage: accelerating voltage in keV.
		sphericalAberration: spherical aberration in mm.
		AmplitudeContrastRatio: amplitude contrat ratio.
		phaseShift: phase shift in degrees.
		scalefactor: scalefactor.
		bfactor: bfactor.
		device: str, device to use.
		"""
		super().__init__()
		if phaseShift is None:
			phaseShift = torch.zeros(defocusU.shape, dtype=torch.float32, device=device)
		else:
			phaseShift = torch.tensor(phaseShift, dtype=torch.float32, device=device)

		if scalefactor is None:
			scalefactor = torch.ones(defocusU.shape, dtype=torch.float32, device=device)
		else:
			scalefactor = torch.tensor(scalefactor, dtype=torch.float32, device=device)

		if bfactor is None:
			bfactor = torch.zeros(defocusU.shape, dtype=torch.float32, device=device)
		else:
			bfactor = torch.tensor(bfactor, dtype=torch.float32, device=device)


		saved_args = locals()
		assert len(set({len(val) for arg_name,val in saved_args.items() if arg_name not in ["self", "__class__", "device"]})) == 1, "CTF values do not have the same shape."
		assert len(set(side_shape)) == 1, "All images must have the same number of pixels"
		assert len(set(apix)) == 1, "All images must have the same apix"

		self.register_buffer("Npix", torch.tensor(side_shape , dtype=torch.float32, device=device)[:, None])
		self.register_buffer("Apix", torch.tensor(apix, dtype=torch.float32, device=device)[:, None])
		self.register_buffer("dfU", torch.tensor(defocusU[:, None], dtype=torch.float32, device=device))
		self.register_buffer("dfV", torch.tensor(defocusV[:, None], dtype=torch.float32, device=device))
		self.register_buffer("dfang", torch.tensor(defocusAngle[:, None], dtype=torch.float32, device=device))
		self.register_buffer("volt", torch.tensor(voltage[:, None], dtype=torch.float32, device=device))
		self.register_buffer("cs", torch.tensor(sphericalAberration[:, None], dtype=torch.float32, device=device))
		self.register_buffer("w", torch.tensor(amplitudeContrastRatio[:, None], dtype=torch.float32, device=device))
		self.register_buffer("phaseShift", phaseShift[:, None])
		self.register_buffer("scalefactor", scalefactor[:, None])
		self.register_buffer("bfactor", bfactor[:, None])
		self.npix = int(side_shape[0])
		self.apix = apix[0]
		#In this stack, freqs[0, :] corresponds to constant x values, freqs[:, 0] corresponds to contant y values.
		ax = torch.fft.fftshift(torch.fft.fftfreq(self.npix, self.apix))
		mx, my = torch.meshgrid(ax, ax, indexing="xy")
		freqs = torch.stack([mx.flatten(), my.flatten()], 1)
		self.register_buffer("freqs", freqs)
		self.freqs = self.freqs.to(device)


	@classmethod
	def from_starfile(cls, file, device="cpu", **kwargs):
		"""
		Instantiate a CTF object from a starfile.
		:param file: path to the starfile containing the parameters of the ctf
		:param device: str, path to the starfile.
		"""
		df = starfile.read(file)

		overrides = {}

		#First we find the values of the CTF in the optics block of the star file.
		try:
			if "side_shape" in kwargs and "apix" in kwargs:
				raise Exception('User input', 'Apix and side_shape are input by user. Overrinding the ones in the starfile.')

			side_n_pix = int(df["optics"].loc[0, "rlnImageSize"])
			apix = df["optics"].loc[0, "rlnImagePixelSize"]
		except Exception:
		    assert "side_shape" in kwargs and "apix" in kwargs, "side_shape, apix must be provided."
		    side_n_pix = kwargs["side_shape"]
		    apix = kwargs["apix"]

		if "optics" in df:
		    assert len(df["optics"]) == 1, "Currently only support one optics group."
		    overrides["rlnVoltage"] = df["optics"].loc[0, "rlnVoltage"]
		    overrides["rlnSphericalAberration"] = df["optics"].loc[0, "rlnSphericalAberration"]
		    overrides["rlnAmplitudeContrast"] = df["optics"].loc[0, "rlnAmplitudeContrast"]

		#Second, if there are particles in the file we find defocus U, V and angle for each one of them. Otherwise we just find the 
		#values in the optics block and there is only one ctf. Notte that the CTF parameters in the optics groups have precedence on
		# the ones in the data block.

		if "particles" in df:
			df = df["particles"]

		num = len(df)
		ctf_params = np.zeros((num, 9))
		ctf_params[:, 0] = side_n_pix
		ctf_params[:, 1] = apix
		for i, header in enumerate([
		"rlnDefocusU",
		"rlnDefocusV",
		"rlnDefocusAngle",
		"rlnVoltage",
		"rlnSphericalAberration",
		"rlnAmplitudeContrast",
		"rlnPhaseShift",
		]):

			if header in overrides:
				ctf_params[:, i + 2] = overrides[header]
			else:
				ctf_params[:, i + 2] = df[header].values if header in df else None

		return cls(*ctf_params[:, :8].T, phaseShift=ctf_params[:, 8], device=device)

	@classmethod
	def from_cs_file(cls, cs_file,  device="cpu", **kwargs):
		"""
		Instantiate a CTF object from cs file
		:param cs_file: str, path the cs file
		:param device: torch device, device to use
		"""
		assert cs_file.endswith(".cs")
		metadata = np.load(cs_file)
		#Trying to get image size and pixel size from the cs file
		if "side_shape" in kwargs and "apix" in kwargs:
			side_n_pix = kwargs["side_shape"]
			apix = kwargs["apix"]
		else:
			try:
				side_n_pix = metadata["blob/shape"][0][0]
				apix = metadata["blob/psize_A"]
			except Exception:
			    assert "side_shape" in kwargs and "apix" in kwargs, "side_shape, apix must be provided."
	
		ctf_params = np.zeros((len(metadata), 9))
		ctf_params[:, 0] = side_n_pix
		ctf_params[:, 1] = apix
		fields = (
		    "ctf/df1_A",
		    "ctf/df2_A",
		    "ctf/df_angle_rad",
		    "ctf/accel_kv",
		    "ctf/cs_mm",
		    "ctf/amp_contrast",
		    "ctf/phase_shift_rad",
		)

		for i, f in enumerate(fields):
			ctf_params[:, i + 2] = metadata[f]
			if f in ("ctf/df_angle_rad", "ctf/phase_shift_rad"):  # convert to degrees
				ctf_params[:, i + 2] *= 180 / np.pi

		return cls(*ctf_params[:, :8].T, phaseShift=ctf_params[:, 8], device=device)

	@classmethod
	def create_ctf(cls, cs_star_config, device="cpu", **kwargs):
		"""
		Instantiates a CTF object from cs file or star file.
		:param cs_star_config: dictionnary, containing the path to the cs or star file, if it is the product of ab initio reconstruction and if its from heterogeneous refinement
		:param device: torch device, device to use
		"""
		file = cs_star_config["file"]
		assert file.endswith(".cs") or file.endswith(".star"), "The file for CTF must be a starfile or a cryosparc file."
		if file.endswith(".cs"):
			return cls.from_cs_file(file, device=device, apix = kwargs["apix"], side_shape = kwargs["side_shape"])

		return cls.from_starfile(file, device=device, apix = kwargs["apix"], side_shape = kwargs["side_shape"])

	def compute_ctf(self, indexes
		) -> torch.Tensor:
		"""
		Compute the 2D CTF
		This code is based on the cryoDRGN code.

		Input:
		    indexes: torch.tensor(batch_size) of indexes of the images in this batch.
		"""
		# convert units
		volt = self.volt[indexes]
		cs = self.cs[indexes]
		dfu = self.dfU[indexes]
		dfv = self.dfV[indexes]
		dfang = self.dfang[indexes]
		w = self.w[indexes]
		phase_shift = self.phaseShift[indexes]
		bfactor = self.bfactor[indexes]
		scalefactor = self.scalefactor[indexes]

		volt = volt * 1000
		cs = cs * 10 ** 7
		dfang = dfang * np.pi / 180
		phase_shift = phase_shift * np.pi / 180

		# lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
		lam = 12.2639 / torch.sqrt(volt + 0.97845e-6 * volt ** 2)
		bsz = len(indexes)
		freqs = self.freqs.repeat(bsz, 1, 1)
		x = self.freqs[..., 0]
		y = self.freqs[..., 1]
		#Since we take arctan between y and x and not x and y, we are still in x ordering but x is the second coordinate now !
		ang = torch.arctan2(y, x)
		s2 = x ** 2 + y ** 2
		df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
		gamma = (
		        2 * torch.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam ** 3 * s2 ** 2)
		        - phase_shift
		)
		ctf = torch.sqrt(1 - w ** 2) * torch.sin(gamma) - w * torch.cos(gamma)
		if scalefactor is not None:
			scalefactor = self.scalefactor[indexes]
			ctf *= scalefactor
		if bfactor is not None:
			bfactor = self.bfactor[indexes]
			ctf *= torch.exp(-bfactor / 4 * s2)


		#But in this project, the images are (y_coords, x_coords), see renderer.project so we transpose:
		ctf = ctf.reshape((len(indexes), self.npix, self.npix))
		return ctf
  






