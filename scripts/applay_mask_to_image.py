import sys
import os
import h5py
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.ndimage
from scipy import misc

from skimage import measure, morphology
from PIL import Image, ImageDraw
from skimage.morphology import ball, binary_erosion, binary_dilation
from skimage.measure import label, regionprops

from optparse import OptionParser
import re
import multiprocessing

parser = OptionParser()
parser.add_option("-i", "--input", dest="input",
                  help="input image path")
parser.add_option("-m", "--masks", dest="masks",
                  help="Path with output masks predicted by nn")
                  
parser.add_option("-s", "--image_size", dest="image_size",
                  help="Output images size")
                  
parser.add_option("-o", "--output", dest="output",
                  help="Output path to store processed images")
                  
(options, args) = parser.parse_args()

IMAGES_BASE_PATH = options.input
OUTPUT_IMAGES_PATH = options.output
MASKS_INPUT_PATH = options.masks


# read masks from mask hdf5 file
def read_mask(mask_path):
    h5 = h5py.File(mask_path, 'r')
    mask = np.array(h5['mask'])
    h5.close()
    return mask

#extract region from masked file
def extract_regions(mask):
    out_regions = []
    b_masks = np.array(mask >= 0.5, dtype=np.int8)
    d_ball  = ball(3)
    b_masks = binary_dilation(b_masks, d_ball)
    
    labels = label(b_masks)
    regions = regionprops(labels)
    slice_reg = None
    max_area = 0 
    for ir, region in enumerate(regions):
        area = region['area']
        if area > max_area:
            slice_reg = (region['centroid'], region['bbox'], region['area'])
    return b_masks, slice_reg
 
# extract mask from image 
def extract_region_from_image(mask, image, image_name, out_size=224):
    mask = np.transpose(mask,(1,2,0))
    b_mask, reg = extract_regions(mask)
    if reg is not None:
      reg = reg[1]
      m  = mask[reg[0]:reg[2], reg[1]:reg[3]]
      m = np.concatenate((m, m, m), 2)
      im = image[reg[0]:reg[2], reg[1]:reg[3]]
    else:
      print "No mask " + image_name
      im = image
      m  = 1
    return scipy.misc.imresize(im / 255.0 * m, (out_size, out_size))

# write masked images
def write_masked_image(params):
     im = misc.imread(params[0])
     if len( im.shape ) == 0: return
     mask = read_mask(params[1])
     mim = extract_region_from_image(mask, im, params[0], int(options.image_size))
     if mim is not None:
        misc.imsave(params[2], mim)

if __name__ == "__main__":
   if not os.path.exists(OUTPUT_IMAGES_PATH):
      os.makedirs(OUTPUT_IMAGES_PATH)
   image_class_path = os.listdir(IMAGES_BASE_PATH)
   image_class_path.sort()
   for cp in image_class_path:
       if re.match(r'Type_\d+', cp) is None: continue
       print 'Processing: ' + cp
       im_cp = IMAGES_BASE_PATH + '/' + cp
       m_cp  = MASKS_INPUT_PATH + '/' + cp
       o_cp  = OUTPUT_IMAGES_PATH + '/' + cp
       if not os.path.exists(o_cp):
          os.makedirs(o_cp)
          
       images = os.listdir(im_cp)
       params = []
       for imf in images:
           params.append((im_cp + '/' + imf, m_cp + '/' + imf + '.h5', o_cp + '/' + imf))
           
       pool = multiprocessing.Pool(processes=20)
       pool.map(write_masked_image, params, chunksize=20)
       
       pool.terminate()
       pool.close()

       print ''