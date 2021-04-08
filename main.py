import dataset
from model import BioFace_FCN
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import separate_and_scale_output
from setup import setup
from decoder import ModelBasedDecoder
from loss import bioface_loss
import time

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = setup()
    celebA50k = dataset.CelebA50k("data/celebA50k.hdf5")
    loader = torch.utils.data.DataLoader(celebA50k, batch_size=64, shuffle=True, drop_last=True)

    model = BioFace_FCN([32,64,128,256,512])
    model.to(device)
    decoder = ModelBasedDecoder(params)
    decoder.to(device)

    celebA_avgImg = torch.tensor([129.1863,104.7624,93.5940]).cuda()
    mu_image = torch.reshape(celebA_avgImg,(1,3,1,1))

    #definitely not best optimizer, paper uses this due to autonn matlab limiations most likely
    optim = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

    total_start_time = time.time()
    for epoch in range(10):  #PAPER DID 200 EPOCHS
        epoch_start_time = time.time()
        Bio_losses = []
        i = 0

        for batch, x in enumerate(loader):
            optim.zero_grad()
            i += 1
            x = x.to(device)
            x_img, x_mask, x_shading = x[:,0:3,:,:], x[:,6:7,:,:], x[:,3:6,:,:]

            #TRAINABLE ENC/DEC setup
            z,p = model(x_img)
            #seperating maps and scaling
            weightA,weightD,CCT,Fweights,b,bGrid,fmel,fblood,predictedShading,specmask = separate_and_scale_output(z,p)
            #Physics-Based Decoder
            sRGBim, specularities = decoder(weightA,weightD,CCT,Fweights,b,bGrid,fmel,fblood,predictedShading,specmask)
            
            #CONSIDER MOVING LOSS TO DEVICE
            loss = bioface_loss(sRGBim, x_img, x_mask, x_shading, specularities, predictedShading, b, mu_image)
            loss.backward()
            optim.step()

            if i%100 == 0:
                print('[Epoch %d] %d/%d' % (epoch, i*64,len(celebA50k)))
                print('Elapsed Time since Epoch Started: %f' % (time.time()-epoch_start_time))


    # # SAVE THE MODEL -- need to implement
    return







if __name__ == "__main__":
    main()