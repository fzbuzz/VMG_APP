from helper_functions import illumination_normalization, load_mats, CameraSensitivityPCA

def setup():
    params = load_mats("util")

    params['mu'], params['PC'], params['EVpca'] = CameraSensitivityPCA(params['cmf'])

    # Normalization of illumniaton parameters to solve scale ambiguity between the 
    # overall intensity of the light source and the diffuse/specular shading
    illumFNorm, illumDNorm, illumA = illumination_normalization(params['illF'], params['illumDmeasured'], params['illumA'])
    params['illumFNorm'], params['illumDNorm'], params['illumA'] =  illumFNorm, illumDNorm, illumA 

    params['wavelength'] = 33

    return params








