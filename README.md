# VMG_APP

suggested usage:

Run/Modify first two cells of train.ipynb

QUICK OVERVIEW:
util/
    - mat files included in original BioFace paper 
helpers/
    - combine_h5s.py : 
        Download Partial Dataset from NeuralFaceEditing Paper: https://drive.google.com/drive/folders/1UMiaw36z2E1F-tUBSMKNAjpx0o2TePvF
        Run function to generate celebA50k
dataset.py 
    Torch dataset class for celebA50k
decoder.py
    Physics Based Decoder
helper_functions.py
    misc helpers for scaling output, illumination normalization, camera sensitive, and some preprocessings
loss.py
    BioFace Loss consisting of 4 parts
model.py
    BioFace "U-Net" initial model to extract the 4 components
setup.py
    setup parameters and pre-process
main.py - WIP
    run to train, still need to save / checkpoint model
train.ipynb
    jupyter to control training

*not included:*
data/
    - celeba50k.hdf5 :
        Combined partial dataset, 50k images.
experiment.ipynb
experiment2.ipynb
    experimentation stuff

        