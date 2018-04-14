# -*-coding: utf-8 -*-
from imageRetrieval import ImageRetrieval


import re, os, os.path
import time
import glob
import ConfigParser
import string

#cf = ConfigParser.ConfigParser()
#cf.read('staticlib/mark.ini')


def getSimpleResult():

	IR = ImageRetrieval("config.cfg")
	IR.buildDictionary()
	IR.buildFeaturePool()
	IR.saveFeaturePool('features.yml')
	IR.loadFeaturePool('features.yml')

	

	total=0
	starttime = time.time()
        i=0
	for file_name in glob.glob(r'/home/ckp/data/siftfeature/*.sift'):
		#print num, os.path.split(file_name)[-1]
		result = IR.retrievalImage(file_name, 50)
	        if result.imagePath1 == "" :
			print "bad query image"
			continue;
		
		match_frame_name1=result.imagePath1
		match_frame_name2=result.imagePath2
		match_frame_name3=result.imagePath3
		match_frame_name4=result.imagePath4
		
		#for i in range(4):
		#    match_frame_name[i]=(*result.imagePath)[i]
		#    print match_frame_name[i]
		
		
		idtest=string.atoi((os.path.splitext(os.path.split(file_name)[1])[0])[7:12])
		print idtest
		idtrain=string.atoi((os.path.splitext(os.path.split(match_frame_name1)[1])[0])[7:12])
		print idtrain
			
                if idtest/4==idtrain/4:
                    total=total+1			
		
        
		idtrain=string.atoi((os.path.splitext(os.path.split(match_frame_name2)[1])[0])[7:12])
		print idtrain	
                if idtest/4==idtrain/4:
                    total=total+1	
        
		
		idtrain=string.atoi((os.path.splitext(os.path.split(match_frame_name3)[1])[0])[7:12])
		print idtrain	
                if idtest/4==idtrain/4:
                    total=total+1	
        
		idtrain=string.atoi((os.path.splitext(os.path.split(match_frame_name4)[1])[0])[7:12])
		print idtrain	
                if idtest/4==idtrain/4:
                    total=total+1	
		print i," ",total
                i=i+1	
		 #if os.path.splitext(cf.get('static',os.path.split(file_name)[1]))[0] == os.path.splitext(os.path.split(match_frame_name)[1])[0]:
			 #correct+=1			
		 #num += 1
		#print "shot & frame = ",cf.get('static',os.path.split(file_name)[-1]),"->",os.path.split(match_frame_name)[-1],result.score
	score=total/10200.
	endtime = time.time()
	print "score:",score
	#print "correct/num = ",correct, "/", num
	print "match time: ",(endtime - starttime)
	

if __name__ == "__main__":
	getSimpleResult()
