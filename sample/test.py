# -*-coding: utf-8 -*-
from imageRetrieval import ImageRetrieval

import re, os, os.path
import time
import glob
import ConfigParser

cf = ConfigParser.ConfigParser()
cf.read('staticlib/mark.ini')


def getSimpleResult():

	IR = ImageRetrieval("config.cfg")
	#IR.buildDictionary()
	#IR.buildFeaturePool()
	#IR.saveFeaturePool('features.yml')
	IR.loadFeaturePool('features.yml')
	
	if False:
		IR.deleteFeatureFromPool('./staticlib/blacklist.txt')

		correct = 0
		num = 0
		starttime = time.time()
		for file_name in glob.glob(r'/home/ckp/Imagedetection/sift/sample/staticlib/static_test/*.jpg'):
			#print num, os.path.split(file_name)[-1]
			result = IR.retrievalImage(file_name, 1)
			#if result.imagePath == "" :
			#	print "bad query image"
			#	continue;
			match_frame_name=result.imagePath
				
			if os.path.splitext(cf.get('static',os.path.split(file_name)[1]))[0] == os.path.splitext(os.path.split(match_frame_name)[1])[0]:
				correct+=1			
			num += 1
			#print "shot & frame = ",cf.get('static',os.path.split(file_name)[-1]),"->",os.path.split(match_frame_name)[-1],result.score
		
		endtime = time.time()
		print "correct/num = ",correct, "/", num
		print "match time: ",(endtime - starttime)

		IR.addFeatureIntoPool('./staticlib/blacklist.txt')

	correct = 0
	num = 0
	starttime = time.time()
	#print "p1"
	for file_name in glob.glob(r'/home/ckp/Imagedetection/sift/sample/staticlib/static_test/*.jpg'):
		#print num, os.path.split(file_name)[-1]
		result = IR.retrievalImage(file_name, 50)
		#if result.imagePath == "" :
		#	print "bad query image"
		#	continue;
		match_frame_name=result.imagePath
		#print "p4"
		print len(match_frame_name)
		print match_frame_name[0]
		print match_frame_name[1]
		print match_frame_name[2]
		
			
		#if os.path.splitext(cf.get('static',os.path.split(file_name)[1]))[0] == os.path.splitext(os.path.split(match_frame_name)[1])[0]:
		for i in range(3):
		   if os.path.splitext(cf.get('static',os.path.split(file_name)[1]))[0] == os.path.splitext(os.path.split(match_frame_name[i])[1])[0]:
			   correct+=1			
		   num += 1
		print "shot & frame = ",cf.get('static',os.path.split(file_name)[-1]),"->",os.path.split(match_frame_name[i])[-1],result.score
	
	endtime = time.time()
	print "correct/num = ",correct, "/", num
	print "match time: ",(endtime - starttime)
	

if __name__ == "__main__":
	getSimpleResult()
