import matplotlib
import numpy as np

# Visualization
def error_to_color(errors, clipping_error=None):
    if clipping_error is not None:
        errors_norm = np.clip(errors / float(clipping_error), 0., 1.)
    else:
        errors_norm = (errors - errors.min()) / (errors.max() - errors.min())

    hsv = np.ones((errors.shape[-1], 3))
    hsv[:, 0] = (1. - errors_norm) / 3.

    return matplotlib.colors.hsv_to_rgb(hsv)
