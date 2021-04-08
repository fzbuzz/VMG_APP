import torch
from torch import nn

class ModelBasedDecoder(nn.Module):
    def __init__(self, params):
        super(ModelBasedDecoder, self).__init__()
        self.illumFNorm = torch.from_numpy(params['illumFNorm']).cuda()
        self.illumDNorm = torch.from_numpy(params['illumDNorm']).cuda()
        self.illumA =  torch.from_numpy(params['illumA']).cuda()
        self.mu = torch.from_numpy(params['mu']).cuda()
        self.PC = torch.from_numpy(params['PC']).type(torch.cuda.FloatTensor).cuda()

        #256 x 256 x 33 -> 64 x 33 x 256 x 256 (64 is batch size)
        self.newskincolour = torch.from_numpy(params['newskincolour']).unsqueeze(0).expand(64,256,256,33).permute(0,3,1,2).cuda()
        self.Tmatrix = torch.from_numpy(params['Tmatrix']).permute(2,0,1).unsqueeze(0).expand(64,9,128,128).cuda()

        # self.t = torch.randn(1)
        # t.requires_grad = True

        self.Txyzrgb = torch.tensor([3.2406, -1.5372, -0.4986,-0.9689, 1.8758, 0.0415,0.0557, -0.2040, 1.057]).cuda()
        self.Txyzrgb.requires_grad = False




    def forward(self,weightA,weightD,CCT,Fweights,b,bGrid,fmel,fblood,predictedShading,specmask):
        e = self.illumniationModel(weightA, weightD, Fweights, CCT, self.illumA, self.illumDNorm, self.illumFNorm)
        Sr, Sg, Sb =                    self.cameraModel(self.mu, self.PC, b)
        lightcolour =                   self.computeLightColour(e, Sr, Sg, Sb)
        specularities =                 self.computeSpecularities(specmask, lightcolour)
        R_total =                       self.BioToSpectralRef(fmel, fblood, self.newskincolour)
        rawAppearance, diffuseAlbedo =  self.ImageFormation(R_total, Sr, Sg, Sb, e, specularities, predictedShading)
        imwhiteBalanced =               self.whiteBalance(rawAppearance, lightcolour)
        T_RAW2XYZ =                     self.findT(self.Tmatrix, bGrid)
        sRGBim =                        self.fromRawTosRGB(imwhiteBalanced, T_RAW2XYZ)

        return sRGBim, specularities


        

    def illumniationModel(self,weightA,weightD,Fweights,CCT,illumA,illumDNorm,illumFNorm):
        illuminantA =  illumA.unsqueeze(0)*weightA
        # CCT # correlated color temperature, 1 x 1 x 1 x B

        # % illumination D: 
            # illumDlayer = Layer.fromFunction(@vl_nnillumD);
            # illD   = illumDlayer(CCT,illumDNorm); %  (1 x 1 x 1 x B) x  1 x 1 x 33 x 22
            # illuminantD = illD.*weightD; 

        # some Fully-Connected Layer of CCT (1) x illumDNorm (1 x 1 x 33 x 22) -> (1 x 1 x 33 x 1)? 
        # I OMITTED this part because i truly had no idea where vl_nnillumD referenced was
        
        illumFNorm = illumFNorm.permute(0,3,2,1)
        illuminantF = Fweights*illumFNorm
        illuminantF = torch.sum(illuminantF, dim=1,keepdim=True)
        illuminantF = illuminantF.permute(0,1,3,2)

        e = illuminantA  + illuminantF # + illuminantD

        e = e/torch.sum(e, dim=3,keepdim=True)
        return e

    def cameraModel(self,mu,PC,b):
        S = torch.matmul(b, PC.T) + mu
        S = nn.ReLU()(S)
        Sr = S[:,:33].unsqueeze(1).unsqueeze(1)
        Sg = S[:,33:66].unsqueeze(1).unsqueeze(1)
        Sb = S[:,66:].unsqueeze(1).unsqueeze(1)
        return Sr, Sg, Sb

    def computeLightColour(self,e,Sr,Sg,Sb):
        lc1  = torch.sum(Sr*e, dim=3, keepdim=True)
        lc2  = torch.sum(Sg*e, dim=3, keepdim=True)
        lc3  = torch.sum(Sb*e, dim=3, keepdim=True)
        lightcolour = torch.cat((lc1, lc2, lc3), dim=1) # 64(B) x 3 x 1 x 1
        return lightcolour

    def computeSpecularities(self, specmask,lightcolour):
        specularities = specmask*lightcolour
        return specularities

    def BioToSpectralRef(self, fmel, fblood, newskincolour):
        biophysicalmaps = torch.cat((fmel, fblood),dim=1).permute(0,2,3,1)
        R_total = nn.functional.grid_sample(newskincolour,biophysicalmaps) # produces B x 33 x H x W
        return R_total

    def ImageFormation(self, R_total, Sr, Sg, Sb, e, specularities, predictedShading):
        R_total = R_total.permute(0,2,3,1)
        spectraRef = R_total*(e)
        rChannel = torch.sum(spectraRef*Sr, dim=3, keepdim=True)
        gChannel = torch.sum(spectraRef*Sg, dim=3, keepdim=True)
        bChannel = torch.sum(spectraRef*Sb, dim=3, keepdim=True)

        diffuseAlbedo = torch.cat((rChannel, gChannel, bChannel),dim=3).permute(0,3,1,2) 
        rawAppearance = diffuseAlbedo * predictedShading + specularities # B x 3 x 64 x 64

        return rawAppearance, diffuseAlbedo

    def whiteBalance(self,rawAppearance, lightcolour):
        #I think you could do this in one operation, fix later
        WBrCh = rawAppearance[:,0,:,:]/lightcolour[:,0,:,:]
        WBgCh = rawAppearance[:,1,:,:]/lightcolour[:,1,:,:]
        WBbCh = rawAppearance[:,2,:,:]/lightcolour[:,2,:,:]

        imwhiteBalanced = torch.cat((WBrCh.unsqueeze(1), WBgCh.unsqueeze(1), WBbCh.unsqueeze(1)), dim=1)

        return imwhiteBalanced


    def findT(self,Tmatrix,bGrid):
        bGrid = bGrid.permute(0,2,3,1)
        T_RAW2XYZ = nn.functional.grid_sample(Tmatrix,bGrid)
        return T_RAW2XYZ

    def fromRawTosRGB(self,imwhiteBalanced,T_RAW2XYZ):
        #I'm almost positive you can do both of these with torch.matmul()--investigate later

        Ix = T_RAW2XYZ[:,0,0,0]*imwhiteBalanced[:,0,:,:] + T_RAW2XYZ[:,3,0,0]*imwhiteBalanced[:,1,:,:] + T_RAW2XYZ[:,6,0,0]*imwhiteBalanced[:,2,:,:] 
        Iy = T_RAW2XYZ[:,1,0,0]*imwhiteBalanced[:,0,:,:] + T_RAW2XYZ[:,4,0,0]*imwhiteBalanced[:,1,:,:] + T_RAW2XYZ[:,7,0,0]*imwhiteBalanced[:,2,:,:]
        Iz = T_RAW2XYZ[:,2,0,0]*imwhiteBalanced[:,0,:,:] + T_RAW2XYZ[:,5,0,0]*imwhiteBalanced[:,1,:,:] + T_RAW2XYZ[:,8,0,0]*imwhiteBalanced[:,2,:,:] 

        Ix.shape
        Ixyz = torch.cat((Ix.unsqueeze(1),Iy.unsqueeze(1),Iz.unsqueeze(1)), dim=1)
        Ixyz.shape

        R = self.Txyzrgb[0]*Ixyz[:,0,:,:] + self.Txyzrgb[3]*Ixyz[:,1,:,:] + self.Txyzrgb[6]*Ixyz[:,2,:,:]
        G = self.Txyzrgb[1]*Ixyz[:,0,:,:] + self.Txyzrgb[4]*Ixyz[:,1,:,:] + self.Txyzrgb[7]*Ixyz[:,2,:,:]
        B = self.Txyzrgb[2]*Ixyz[:,0,:,:] + self.Txyzrgb[5]*Ixyz[:,1,:,:] + self.Txyzrgb[8]*Ixyz[:,2,:,:]

        sRGBim = torch.nn.ReLU()(torch.cat((R.unsqueeze(1),G.unsqueeze(1),B.unsqueeze(1)), dim=1))
        return sRGBim