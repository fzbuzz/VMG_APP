from scipy.io import loadmat
from sklearn.decomposition import PCA
import os
import numpy as np
import torch

def illumination_normalization(illF, illumDmeasured, illumA):  
    illF = np.expand_dims(illF, axis=0)
    illumDmeasured = illumDmeasured.T
    illumDmeasured = np.expand_dims(illumDmeasured, axis = (0,1))
    illumA = np.divide(illumA, np.sum(illumA))
    illumDNorm = np.zeros((1,1,33,22))
    for i in range(0,22):
        illumDNorm[0,0,:,i] =  np.divide(illumDmeasured[0,0,:,i], np.sum(illumDmeasured[0,0,:,i]))

    illumFNorm = np.zeros((1,1,33,12))
    for i in range(0,12):
        illumFNorm[0,0,:,i] = np.divide(illF[0,0,:,i], np.sum(illF[0,0,:,i]))

    return illumFNorm, illumDNorm, illumA


def load_mats(relative_path_to_util):
    parameters = {}
    parameters['cmf'] = loadmat(os.path.join(relative_path_to_util,'rgbCMF.mat'))['rgbCMF']
    parameters['illF'] = loadmat(os.path.join(relative_path_to_util,'illF.mat'))['illF']
    parameters['illumA'] = loadmat(os.path.join(relative_path_to_util,'illumA.mat'))['illumA']
    parameters['illumDmeasured'] = loadmat(os.path.join(relative_path_to_util,'illumDmeasured.mat'))['illumDmeasured']
    parameters['newskincolour'] = loadmat(os.path.join(relative_path_to_util,'Newskincolour.mat'))['Newskincolour']
    parameters['Tmatrix'] = loadmat(os.path.join(relative_path_to_util,'Tmatrix.mat'))['Tmatrix']

    return parameters



def CameraSensitivityPCA(cmf):
    X = np.zeros((99,28))
    Y = np.zeros((99,28))

    redS = cmf[0,0]
    greenS = cmf[0,1]
    blueS = cmf[0,2]

    for i in range(28):
        Y[0:33, i]  = np.divide(redS[:,i], np.sum(redS[:,i]))
        Y[33:66, i] = np.divide(greenS[:,i], np.sum(greenS[:,i]))
        Y[66:99, i] = np.divide(blueS[:,i], np.sum(blueS[:,i]))

    Y = Y.T
    pca = PCA()
    pca = pca.fit(Y)
    PC, EV, explained, mu = pca.components_, pca.explained_variance_, pca.explained_variance_ratio_*100, pca.mean_
    PC = PC.T
    PC = np.matmul(PC[:,0:2],np.diag(np.sqrt(EV[0:2])))
    PC[:,1] *= -1
    mu = mu.T
    EV = EV[0:2]
    return mu, PC, EV


def separate_and_scale_output(z,p):
    light_params = p[:,0:15,:,:]
    b = torch.reshape(p[:,15:,:,:],(-1,2))
    fmel = z[:,0:1,:,:]
    fblood = z[:,1:2,:,:]
    shading = z[:,2:3,:,:]
    specmask = z[:,3:4,:,:]

    CCT = light_params[:,14:15,:,:]
    CCT = (22-1)/(1+torch.exp(-1*CCT)) + 1

    light_weights = torch.softmax(light_params[:,:14,:,:], dim=1)
    weightA = light_weights[:,0:1,:,:]
    weightD = light_weights[:,1:2,:,:]
    Fweights = light_weights[:,2:14,:,:]

    b = torch.mul(6,torch.sigmoid(b))-3
    bGrid = b.unsqueeze(2).unsqueeze(2) # B x 1 x 1 x 2
    bGrid = bGrid/3

        
    fmel = torch.sigmoid(fmel)*2-1
    fblood = torch.sigmoid(fblood)*2-1
    predictedShading = torch.exp(shading)
    specmask = torch.exp(specmask)

    return weightA,weightD,CCT,Fweights,b,bGrid,fmel,fblood,predictedShading,specmask


