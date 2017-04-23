import sys
import os
from shutil import copyfile

# cat <file with images> | python copy_images_from_list.py <output path>

OUTPUT_PATH = sys.argv[1]
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
   
for line in sys.stdin:
    line = line.strip()
    if len(line) > 0:
      fname = line.split()[-1]
      if os.path.isfile(fname) == True:
         type_dir = fname.split('/')[-2]
         out_file = fname.split('/')[-1]
         out_dir  = OUTPUT_PATH + '/' + type_dir
         if not os.path.exists(out_dir):
            os.makedirs(out_dir)
         copyfile(fname, out_dir + '/' + out_file)