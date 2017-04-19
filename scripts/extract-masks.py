import sys
import os
import h5py
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.ndimage
from scipy import misc
from skimage import measure, morphology

import json
import re

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="input",
                  help="input image path")
parser.add_option("-m", "--masks", dest="masks",
                  help="Path with masks markers files")
                  
parser.add_option("-o", "--output", dest="output",
                  help="Output path to store h5 files with masks")
                  
(options, args) = parser.parse_args()

IMAGES_BASE_PATH = options.input
OUTPUT_MASK_PATH = options.output
MARKERS_INPUT_PATH = options.masks
markers = os.listdir(MARKERS_INPUT_PATH)
markers.sort()

def ellipse(w, h, dtype=np.uint8):
    X, Y = np.meshgrid(range(-w, w + 1), range(-h, h + 1))
    return np.array(((h*X) ** 2 + (w*Y) ** 2) <= (w*h) ** 2, dtype=np.uint8)
    
def mark_image(ImageType, ImageName, ShapeAttributes):
    im  = misc.imread(IMAGES_BASE_PATH + '/' + ImageType + '/' + ImageName)
    if len(im.shape) == 0: return None, None
    im_shape = im.shape
    mask = np.zeros([im_shape[0], im_shape[1]])    
    cx = ShapeAttributes['cx']
    cy = ShapeAttributes['cy']
    rx = 0
    ry = 0
    fig = None
    if ShapeAttributes['name'] == 'ellipse':
        rx = ShapeAttributes['rx']
        ry = ShapeAttributes['ry']
        fig = ellipse(rx, ry)
    else: 
        if ShapeAttributes['name'] == 'circle':
            rx = ShapeAttributes['r']
            ry = ShapeAttributes['r']
            fig = morphology.disk(rx)
    sx = max(0, cx-rx)
    ex = min(mask.shape[1], cx+rx+1)
    sy = max(0, cy-ry)
    ey = min(mask.shape[0], cy+ry+1)
    
    fsx = max(0, 0-(cx-rx))
    fex = min(fig.shape[1], fig.shape[1]- ((cx+rx+1)-mask.shape[1]))
    fsy = max(0, 0-(cy-ry))
    fey = min(fig.shape[0], fig.shape[0]- ((cy+ry+1)-mask.shape[0]))
    mask[sy:ey, sx:ex] += fig[fsy:fey, fsx:fex]
    return mask, im

if __name__ == "__main__":
    for mf in markers:
        marker_file = pd.read_csv(MARKERS_INPUT_PATH + '/' + mf)
        mf = re.search(r'^(.+?)\.txt', mf).group(1)
        print mf
        for i, j in enumerate(marker_file['region_shape_attributes']):
            o = json.loads(j)
            if 'name' in o:
                f = marker_file['#filename'][i]
                mask, image = mark_image( mf , f,o)
                if mask is None:
                   print "Miss: " + f
                   continue
                image = scipy.misc.imresize(image, (512, 512, 3)).transpose((2, 0, 1))
                mask  = scipy.misc.imresize(mask, (512, 512))
                ofile = h5py.File(OUTPUT_MASK_PATH + '/' + mf+'_'+ f + '.h5', 'w')
                ofile.create_dataset('mask',data=mask, compression='gzip')
                ofile.create_dataset('image', data=image, compression='gzip')
                ofile.close()
            sys.stderr.write('\r%d'% i)