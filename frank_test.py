import numpy as np
import matplotlib.pyplot as plt

import frank
from frank.io import load_uvtable, save_fit
from frank.radial_fitters import FrankFitter
from frank.geometry import FitGeometryGaussian
from frank.make_figs import make_quick_fig


as209_dat = np.load('AS209_continuum.npz')
u, v, vis, weights = [as209_dat[k] for k in ['u', 'v', 'V', 'weights']]

FF = FrankFitter(Rmax=1.6, N=250, geometry=FitGeometryGaussian(),
                 alpha=1.05, weights_smooth=1e-4)

sol = FF.fit(u, v, vis, weights)

fig, axes = make_quick_fig(u, v, vis, weights, sol, bin_widths=[1e3, 5e4], force_style=True)
plt.savefig('as209_dat_frank_fit_quick.png', dpi=200)

