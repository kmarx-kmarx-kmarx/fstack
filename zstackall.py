import pandas as pd
import cv2
import numpy as np
from skimage.morphology import white_tophat
from skimage.morphology import disk
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
import gcsfs
from utils import *
from fstack_cu import fstack_cu_images
import os
import time
t0 = time.time()

# script settings
remove_background = False
subtract_background = True
invert_contrast = True
save_individual_images = False
save_stack = True
shift_registration = True
use_color = False
channels = [405,488,561,638]
colors = {'0':[255,255,255],'1':[255,200,0],'2':[30,200,30],'3':[0,0,255]} # BRG

# FOV setting
imax = 5 # row of the scan
jmax = 5 # column of the scan
kmax = 9 # z stack layer

savepath = "/home/prakashlab/Documents/images/cycle/"
os.makedirs(savepath, exist_ok=True)


# crop settings
crop_start = 0
crop_end = 3000
a = crop_end - crop_start

# gcs setting
gcs_project = 'soe-octopi'
gcs_token = '/home/prakashlab/Documents/fstack/codex-20220324-keys.json'
gcs_settings = {}
gcs_settings['gcs_project'] = gcs_project
gcs_settings['gcs_token'] = gcs_token
fs = gcsfs.GCSFileSystem(project=gcs_project,token=gcs_token)

# dataset ID
bucket_source = 'gs://octopi-codex-data'
bucket_destination = 'gs://octopi-codex-data-processing'
experiment_id = '20220601_20x_75mm'

with fs.open( bucket_source + '/' + experiment_id + '/' + 'index.csv', 'r' ) as f:
    database_df = pd.read_csv(f)
print(database_df)

n = database_df.shape[0] # n is the number of cycles
for i in range(imax + 1):
    for j in range(jmax + 1):
        if use_color:
            M = np.zeros((a,a,3),'uint8')
        else:
            M = np.zeros((a,a),'uint8')
        if save_stack:
            I_stack = np.zeros((a,a,len(channels),n))
        cycle_counter = 0
        I_zs = np.zeros((kmax,a,a))
        for index, row in database_df.iterrows():
            print(row['Round'])
            print(row['Acquisition_ID'])
            # for l in range(1):
            for l in range(len(channels)):
                # get the color for the channel
                color = colors[str(l)]
                if l == 0 and invert_contrast:
                    color = [0,0,0]
                # get the channel
                channel = channels[l] 
                for k in range(kmax):
                    filename = row['Acquisition_ID'] + '/0/' + str(i) + '_' + str(j) + '_' + str(k) + '_Fluorescence_' + str(channel) + '_nm_Ex.bmp'
                    I = imread_gcsfs(fs,bucket_source + '/' + experiment_id + '/'+ filename)
                    I = I[crop_start:crop_end,crop_start:crop_end]
                            
                    I = I.astype('float')
                    I_zs[k,:,:] = I

                #   We now have a full stack of images
                I = fstack_cu_images(I_zs, list(range(kmax)))

                
                if remove_background:
                    selem = disk(30) 
                    I = white_tophat(I,selem)
                
                if subtract_background:
                    I = I - np.amin(I)
                
                # normalize
                I = I.astype('float')
                I = 255*I/np.amax(I)
                # registration

                if shift_registration:
                    if cycle_counter == 0:
                        if l == 0:
                            I0 = I
                    else:
                        if l == 0:
                            # get shift
                            shift, error, diffphase = phase_cross_correlation(I, I0, upsample_factor = 5)
                            print(shift)
                            transform = AffineTransform(translation=(shift[1],shift[0]))
                            I = warp(I, transform)
                        else:
                            # apply shift
                            I = warp(I, transform)
                    # cv2.imwrite(str(row['Round'])+'_'+str(channel)+'.png',I)
                
                if use_color:
                    for m in range(3):
                        if invert_contrast:
                            M[:,:,m] = 255 - I*(1-color[m]/255.0)
                        else:
                            M[:,:,m] = I*color[m]/255.0
                else:
                    if invert_contrast:
                        M = 255 - I
                    else:
                        M = I
                
                # save images
                if save_individual_images:
                    cv2.imwrite(str(cycle_counter) + '_' + str(l) + '.png',M)
                if save_stack:
                    I_stack[:,:,l,cycle_counter] = I

            cycle_counter = cycle_counter + 1



        data = xr.DataArray(I_stack,dims=['y','x','c','t'])
        data = data.transpose('t','c','y','x')
        data = (data).astype('uint8')
        path = savepath + str(i) + '/' + str(j) + '/'
        os.makedirs(path, exist_ok=True)
        for tidx, cyx in enumerate(data):
            for cidx, yx in enumerate(cyx):
                impath  = path + str(tidx) + "_" + str(cidx) + ".png"
                yx = np.array(yx)
                cv2.imwrite(impath, yx)

t1 = time.time()
filepath = savepath + "time_stacking.txt"
with open(filepath, 'w') as f:
    f.write(str(t1-t0))