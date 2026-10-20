"""Plot a transmission spectrum written by a completed fit.

    python examples/plot_spectrum.py WASP-39_SOSS_ORDER1/WASP-39_NIRISS_SOSS_order1_Rreference.csv
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

spectrum = pd.read_csv(sys.argv[1])

fig, ax = plt.subplots(figsize=(9, 4))
ax.errorbar(
    spectrum["wavelength"],
    spectrum["depth_ppm00"],
    xerr=spectrum["wavelength_err"],
    yerr=spectrum["depth_err_ppm00"],
    fmt=".",
    elinewidth=1,
)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel("Transit depth [ppm]")
ax.set_title(sys.argv[1])
fig.tight_layout()
plt.show()
