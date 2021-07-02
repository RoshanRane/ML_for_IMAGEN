import numpy as np
import os
import time as t
import nibabel as nib
from os.path import join, basename, dirname
import re
import matplotlib.pyplot as plt
import ants
import multiprocessing as mp
import pandas as pd

"""
This class PREP is created to preprocess raw MRI images by
1. Normalization with ANTs.
"""
def register_ants_py(mri, dest, mask):
    
    file_name = basename(mri).split(".")[0]

    # Create folders for the output files
    if not os.path.exists(dirname(dest)): os.makedirs(dirname(dest))
        
    print("File {}: starting MNI registration".format(file_name))
    fixed = ants.image_read(mask)
    moving = ants.image_read(mri)
    
    start = t.time()
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN', outprefix=file_name)
    warped_moving = mytx["warpedmovout"]
    warped_moving.to_file(dest)
    print("File {}: registration complete. That took {} seconds.\nSaved at {}".format(file_name, t.time()-start, dest))
#     # ANTs creates these files in the running dir that we won't need any longer
#     os.system("rm -f *1Warp.nii.gz")
#     os.system("rm -f *1InverseWarp.nii.gz")
#     os.system("rm -f *0GenericAffine.mat")



def show_slice(file_path, x, y, z):
    """Function to display row of image slices"""
    img = nib.load(file_path)
    img = img.get_fdata()
    print("The scan has dimensions {}.".format(img.shape))
    slice_0 = img[x, :, :]
    slice_1 = img[:, y, :]
    slice_2 = img[:, :, z]
    slices = [slice_0, slice_1, slice_2]
    fig, axes = plt.subplots(1, len(slices), figsize=[12, 4])
    for i, sli in enumerate(slices):
        axes[i].imshow(sli.T, cmap="gray", origin="lower")


if __name__ == "__main__":
    DATA_DIR = "/ritter/share/data/IMAGEN/IMAGEN_BIDS"
    mask = "/ritter/share/misc/masks/FSL_atlases/MNI/standard/MNI152_T1_1mm_brain.nii.gz"

    # Only do FU3
    ppts = pd.read_csv(join(DATA_DIR, "participants_FU3.csv"), index_col="ID")
    paths = ppts["path_fs-r1_T1w"].dropna().tolist()
    new_paths = ppts["path_fs-r1_T1w"].dropna().str.replace("rec-fs-r1_T1w.mgz", "rec-fs-r1-MNInl_T1w.nii.gz").tolist()

    def wrapper(i):
        if not os.path.isfile(new_paths[i]):
            register_ants_py(paths[i], new_paths[i], mask)
            #prep.skullstrip(frac)
        else:
            print(new_paths[i], "already exists.")
            pass
        
    with mp.Pool(15) as p:
        p.map(wrapper,range(len(paths)))
        
    # ANTs creates these files in the current working dir that we won't need any longer
    os.system("rm -f *1Warp.nii.gz")
    os.system("rm -f *1InverseWarp.nii.gz")
    os.system("rm -f *0GenericAffine.mat")