import numpy as np

def dict_in(varname, average, std, default):
	"""Creates dictionary entry containing scale and shift parameters"""
	return {"name": varname, "shift": average, "scale":std, "default": default}

def Get_Shift_Scale(vec, w, varname):
	"""Calculates the weighted average and std for vector vec and weight w."""
	nans = np.isnan(vec)

	w_without_nan = w[~nans]
	vec_without_nan = vec[~nans]
	default = np.ma.average(vec_without_nan, weights = w_without_nan)

	vec[nans] = default
	average = np.ma.average(vec, weights=w)
	std = np.sqrt(np.average((vec-average)**2, weights=w))
	return [varname, average, std, default]

def Gen_default_dict(scale_dict):
    """Generates default value dictionary from scale/shift dictionary."""
    default_dict = {}
    for elem in scale_dict:
        default_dict[elem['name']] = elem['default']
    return default_dict
