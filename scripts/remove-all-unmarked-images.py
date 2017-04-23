## Removes all unmarked images in input mark failes
## Input - file with marks
##       - images catalog
## if failed exists in input marked file but has not any regions it file will be removed
##
import sys
import os
import pandas as pd
import json
import re

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="input",
                  help="input image path")
parser.add_option("-m", "--masks", dest="masks",
                  help="Path with masks markers files")
                  
                  
(options, args) = parser.parse_args()

IMAGES_BASE_PATH = options.input
MARKERS_INPUT_PATH = options.masks
markers = os.listdir(MARKERS_INPUT_PATH)
markers.sort()

if __name__ == "__main__":
    for mf in markers:
        marker_file = pd.read_csv(MARKERS_INPUT_PATH + '/' + mf)
        mf = re.search(r'^(.+?)\.txt', mf).group(1)
        print mf
        for i, j in enumerate(marker_file['region_shape_attributes']):
            o = json.loads(j)
            if 'name' not in o:
                f = marker_file['#filename'][i]
                fname = IMAGES_BASE_PATH + '/' + mf + '/' + f
                if os.path.isfile(fname) == True:
                   print fname
                   os.remove(fname)