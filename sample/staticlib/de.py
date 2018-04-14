import re, os, os.path,shutil
import time
import glob
import random


#extense = ['.jpg', '.png', '.jpeg', '.bmp','.gif']
list=glob.glob(r'static/*.*')
sub=random.sample(list, 1000)
with open('database.txt', 'w') as f:
  for filename in sub:
    #if os.path.split(filename)[1][0:7] =="placard":
    f.write('/home/ckp/Imagedetection/sift/sample/staticlib/' + filename + '\n')