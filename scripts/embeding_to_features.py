import sys
import os
import h5py
import numpy as np  # linear algebra
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="input",
                  help="Input embedings path")
                                    
parser.add_option("-o", "--output", dest="output",
                  help="Output feature file")
                  

(options, args) = parser.parse_args()

EMBEDING_BASE_PATH = options.input
OUTPUT_EMBEDING_PATH = options.output
classes = {}

# read masks from mask hdf5 file
def read_embeding(embeding_file):
    h5 = h5py.File(embeding_file, 'r')
    embeding = np.array(h5['embeding'])
    h5.close()
    return embeding
    
def sample_dist_from_class(emb_src, IN_PATH, cl, cnt):
    global classes
    global EMBEDING_BASE_PATH
    l= len(classes[cl])
    dist = []
    for i in range(0, cnt): 
        a = np.random.randint(0, l)
        f = classes[cl][a]
        emb = read_embeding(IN_PATH + '/' + f)
        emb = emb.reshape(emb.shape[1])
        dist.append(np.linalg.norm(emb_src-emb))
    dist = np.asarray(dist)
    return np.mean(dist)
    
header = False
def write_embedings_into_file(IN_PATH, cl, out):
   global header;
   global classes
   flist = os.listdir(IN_PATH)
   for i, f in enumerate(flist):
       emb = read_embeding(IN_PATH + '/' + f)
       emb = emb.reshape(emb.shape[1])
       if i == 0 and header == False:
          h = 'id;class;' + ';'.join( [str(k) for k in range(0, emb.shape[0])] ) #+ ';d_1;d_2;d_3'
          print >> out, h
          header = True
       sout = f+';' +str(int(cl)-1) + ';' + ';'.join(map(str, emb))
       #clist = classes.keys()
       #clist.sort()
       #for c in clist:
       #  a_dist = sample_dist_from_class(emb, IN_PATH, cl, 10)
       #  sout += ';%f'%a_dist
         
       print >> out, sout
    
if __name__ == "__main__":

   with open(OUTPUT_EMBEDING_PATH, 'w') as out:
     e_class_path = os.listdir(EMBEDING_BASE_PATH)
     e_class_path.sort()
    
     #for cp in e_class_path:
     #    cl = cp.split('_')[1]
     #    classes[cl] = os.listdir(EMBEDING_BASE_PATH + '/' + cp)
         
     for cp in e_class_path:
         print "Process: " + cp
         IN_PATH = EMBEDING_BASE_PATH + '/' + cp
         write_embedings_into_file(IN_PATH, cp.split('_')[1], out)
    

                  