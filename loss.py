import torch

def bioface_loss(sRGBim, actualImages, actualMask, actualShading, specularities, predictedShading, b, mu_image):
    # Preprocessing, scale for shading
    scaleRGB = sRGBim*255
    Y1 = torch.ones(mu_image.shape).cuda()
    Y1 = Y1*mu_image
    rgbim = scaleRGB - Y1

    actualShading =torch.sum(actualShading, dim=1, keepdim=True)
    scale_numerator = torch.sum(actualShading*predictedShading*actualMask, dim=(2,3), keepdim=True)
    scale_denominator = torch.sum(torch.pow(predictedShading, 2)*actualMask, dim=(2,3), keepdim=True)
    scale = scale_numerator / scale_denominator
    alpha = (actualShading - predictedShading*scale)*actualMask

    # Hypers
    blossweight = 1e-4
    appweight = 1e-3
    shadingweight = 1e-5 
    sparseweight = 1e-5


    appearance = appearance_loss(rgbim, actualImages, actualMask, appweight)
    prior = prior_loss(b, blossweight)
    sparsity = sparsity_loss(specularities, sparseweight)
    shading = shading_loss(alpha, shadingweight)
    return appearance + prior + sparsity + shading



def appearance_loss(rgbim, images,actualMask, appweight):
    delta = (images - rgbim)*actualMask
    return torch.sum(torch.pow(delta,2))/(64**2)*appweight

def prior_loss(b, blossweight):
    return torch.sum(torch.pow(b,2))*blossweight

def sparsity_loss(specularities, sparseweight):
    return torch.sum(specularities)*sparseweight

def shading_loss(alpha, shadingweight):
    return torch.sum(torch.pow(alpha,2))*shadingweight