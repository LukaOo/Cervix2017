import sys
import os
import pandas as pd
import json
from shutil import move

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="input",
                  help="input image path")
parser.add_option("-o", "--output", dest="output",
                  help="Move output image path")
                  
parser.add_option("-m", "--markers", dest="markers",
                  help="Path with  markers files")

parser.add_option("-p", "--preffix", dest="preffix", default ='',
                  help="Prefix for input files")

                  
(opt, args) = parser.parse_args()

IMAGES_BASE_PATH = opt.input
MARKERS_INPUT_PATH = opt.markers
markers = os.listdir(MARKERS_INPUT_PATH)
markers.sort()

if __name__ == "__main__":
    for mf in markers:
        print 'Process: ' + mf
        marker_file = pd.read_csv(MARKERS_INPUT_PATH + '/' + mf)
        st = mf.split('.')[0]
        if 'Type_' in st:
            st = st.split('_')[1]
        for row in marker_file.iterrows():
            attr = json.loads( row[1]['file_attributes'])
            fname= opt.preffix + row[1]['#filename']
            if 'type' in attr:
                st = attr['type']
            if 'bad' in attr and attr['bad'] == '1':  
                    
                infile = opt.input + '/' + st + '/' + fname
                ofile  = opt.output + '/' + st + '/'
                if not os.path.exists(ofile):
                   os.makedirs(ofile)
                ofile += fname
                
                #print (infile, ofile)
                if os.path.exists(infile) == True:
                   move (infile, ofile)
  