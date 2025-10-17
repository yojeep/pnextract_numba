from email.mime import image
import matplotlib.pyplot as plt
import numpy as np

image_classical_pne = np.load(r"image_500_500_500_classical_pne.npz")["arr_0"]
image_python_pne = np.load(r"image_500_500_500_python_pne.npz")["arr_0"]

slices = [0, 100, 200, 300, 400]

fig, axs = plt.subplots(len(slices), 2, figsize=(5, 10))

for i, slice_ in enumerate(slices):
    axs[i, 0].axis("off")
    axs[i, 1].axis("off")
    axs[i, 0].imshow(image_classical_pne[slice_])
    axs[i, 0].set_title(f"Classical PNE, Slice {slice_}", fontsize=9)
    axs[i, 1].imshow(image_python_pne[slice_])
    axs[i, 1].set_title(f"Pnextract_numba, Slice {slice_}", fontsize=9)

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout(pad=0.5)
plt.savefig("pnextract_numba.png", dpi=300)
plt.show()
