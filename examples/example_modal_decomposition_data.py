import matplotlib.pyplot as plt
import numpy as np
import skimage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.longitudinal.gaussian_profile import GaussianLongitudinalProfile
from lasy.profiles.transverse.hermite_gaussian_profile import (
    HermiteGaussianTransverseProfile,
)
from lasy.profiles.transverse.transverse_profile_from_data import (
    TransverseProfileFromData,
)
from lasy.utils.mode_decomposition import hermite_gauss_decomposition

# Define the transverse profile of the laser pulse
img_url = "https://user-images.githubusercontent.com/27694869/228038930-d6ab03b1-a726-4b41-a378-5f4a83dc3778.png"
intensityData = skimage.io.imread(img_url)
intensityData[intensityData < 2.1] = 0
pixel_calib = 0.186e-6
lo = (
    -intensityData.shape[0] / 2 * pixel_calib,
    -intensityData.shape[1] / 2 * pixel_calib,
)
hi = (
    intensityData.shape[0] / 2 * pixel_calib,
    intensityData.shape[1] / 2 * pixel_calib,
)
energy = 0.5
pol = (1, 0)
transverse_profile = TransverseProfileFromData(intensityData, lo, hi)

# Define longitudinal profile of the laser pulse
wavelength = 800e-9
tau = 30e-15
t_peak = 0.0
longitudinal_profile = GaussianLongitudinalProfile(wavelength, tau, t_peak)

# Combine into full laser profile
profile = CombinedLongitudinalTransverseProfile(
    wavelength, pol, energy, longitudinal_profile, transverse_profile
)

# Calculate the decomposition into hermite-gauss modes
m_max = 20
n_max = 20
modeCoeffs, waist = hermite_gauss_decomposition(
    transverse_profile, wavelength, m_max=m_max, n_max=n_max, res=0.5e-6
)

# Reconstruct the pulse using a series of hermite-gauss modes
for i, mode_key in enumerate(list(modeCoeffs)):
    tmp_transverse_profile = HermiteGaussianTransverseProfile(
        waist, waist, mode_key[0], mode_key[1], wavelength
    )
    if i == 0:
        reconstructedProfile = modeCoeffs[
            mode_key
        ] * CombinedLongitudinalTransverseProfile(
            wavelength, pol, energy, longitudinal_profile, tmp_transverse_profile
        )
    else:
        reconstructedProfile += modeCoeffs[
            mode_key
        ] * CombinedLongitudinalTransverseProfile(
            wavelength, pol, energy, longitudinal_profile, tmp_transverse_profile
        )

# Plotting the results
# Plot the original and denoised profiles
# Create a grid for plotting
x = np.linspace(-5 * waist, 5 * waist, 500)
X, Y = np.meshgrid(x, x)

# Determine the figure parameters
fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
fig.suptitle(
    "Hermite-Gauss Reconstruction using m_max = %i, n_max = %i" % (m_max, n_max)
)

# Plot the original profile
pltextent = np.array([np.min(x), np.max(x), np.min(x), np.max(x)]) * 1e6  # in microns
prof1 = np.abs(profile.evaluate(X, Y, 0)) ** 2
maxInten = np.max(prof1)
prof1 /= maxInten
divider0 = make_axes_locatable(ax[0])
ax0_cb = divider0.append_axes("right", size="5%", pad=0.05)
pl0 = ax[0].imshow(prof1, cmap="magma", extent=pltextent, vmin=0, vmax=np.max(prof1))
cbar0 = fig.colorbar(pl0, cax=ax0_cb)
cbar0.set_label("Intensity (norm.)")
ax[0].set_xlabel("x ($ \\mu m $)")
ax[0].set_ylabel("y ($ \\mu m $)")
ax[0].set_title("Original Profile")

# Plot the reconstructed profile
prof2 = np.abs(reconstructedProfile.evaluate(X, Y, 0)) ** 2
prof2 /= maxInten
divider1 = make_axes_locatable(ax[1])
ax1_cb = divider1.append_axes("right", size="5%", pad=0.05)
pl1 = ax[1].imshow(prof2, cmap="magma", extent=pltextent, vmin=0, vmax=np.max(prof1))
cbar1 = fig.colorbar(pl1, cax=ax1_cb)
cbar1.set_label("Intensity (norm.)")
ax[1].set_xlabel("x ($ \\mu m $)")
ax[1].set_ylabel("y ($ \\mu m $)")
ax[1].set_title("Reconstructed Profile")

# Plot the error
prof3 = (prof1 - prof2) / np.max(prof1)  # Normalized error
divider2 = make_axes_locatable(ax[2])
ax2_cb = divider2.append_axes("right", size="5%", pad=0.05)
pl2 = ax[2].imshow(100 * np.abs(prof3), cmap="magma", extent=pltextent)
cbar2 = fig.colorbar(pl2, cax=ax2_cb)
cbar2.set_label("|Intensity Error| (%)")
ax[2].set_xlabel("x ($ \\mu m $)")
ax[2].set_ylabel("y ($ \\mu m $)")
ax[2].set_title("Error")

plt.show()

fig2, ax2 = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
ax2[0].plot(
    x * 1e6,
    prof2[int(len(x) / 2), :],
    label="Reconstructed Profile",
    color=(1, 0.5, 0.5),
    lw=2.5,
)
ax2[0].plot(
    x * 1e6,
    prof1[int(len(x) / 2), :],
    label="Original Profile",
    color=(0.3, 0.3, 0.3),
    lw=1.0,
)
ax2[0].legend()
ax2[0].set_xlim(pltextent[0], pltextent[1])
ax2[0].set_xlabel("x ($ \\mu m $)")
ax2[0].set_ylabel("Intensity (norm.)")

ax2[1].plot(
    x * 1e6,
    prof2[int(len(x) / 2), :],
    label="Reconstructed Profile",
    color=(1, 0.5, 0.5),
    lw=2.5,
)
ax2[1].plot(
    x * 1e6,
    prof1[int(len(x) / 2), :],
    label="Original Profile",
    color=(0.3, 0.3, 0.3),
    lw=1.0,
)
ax2[1].legend()
ax2[1].set_xlim(pltextent[0], pltextent[1])
ax2[1].set_yscale("log")
ax2[1].set_xlabel("x ($ \\mu m $)")
ax2[1].set_ylabel("Intensity (norm.)")

plt.show()
