#ifndef __IMAGERETRIEVAL_H
#define __IMAGERETRIEVAL_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "utils.h"
extern "C"{
#include "vl/kdtree.h"
}

typedef boost::shared_mutex Lock;
typedef boost::unique_lock< Lock > WriteLock;
typedef boost::shared_lock< Lock > ReadLock;

using namespace std;
using namespace cv;

struct CodeBook {
    Mat dictionary;
	Mat dictionaryVlad;
	Mat means;
    Mat covariances;
	Mat priors;
};

struct Feature {
	Mat frame;
    Mat word;
	Mat descriptor;
};

struct DbFeature {
	vector<String> imageName;
    vector<Feature> imageFeature;
};


struct RetrievalResult{
	//const char *imagePath;
	vector<string> imagePath;
	float score;
};

struct Forest{
	VlKDForest *kdtree;
  float *data;
};

struct ImageRetrievalParam{
	String retrievalType;
	int dictDim1;
	int dictDim2;
	bool usePCA;
	int pcaDim;
	String encodeType;
	String basePath;
	String databasePath;
	ImageRetrievalParam(){
		retrievalType = "STATIC";
		dictDim1 = 1200;
		dictDim2 = 50;
		usePCA = true;
		pcaDim = 50;;
		encodeType = "FV";
		basePath = "./data/static/FV/";
		databasePath = "./staticlib/static/";
	}
};

class ImageRetrieval{
public:
	ImageRetrieval(const char *configFile){	
		if (!isAvailability()) {
			cerr << "The library out of data. Please contact the author\n";  
			exit(1);
		}
		getParam(configFile);
	}
	
	~ImageRetrieval(){
		if (forest.kdtree) {
      delete forest.data;
			vl_kdforest_delete(forest.kdtree);
		}
	}

	void buildDictionary();
	void buildFeaturePool();
	void addFeatureIntoPool(char* appendListText);
	void deleteFeatureFromPool(char* blackListText);
	RetrievalResult retrievalImage(const char *imagePath, int k = 100);

	void saveFeaturePool(const char *featureFile);
	void loadFeaturePool(const char *featureFile);

private:
	void init();
	int getParam(const char *configFile);
	Feature getFeature(const char *path);
	void generateUnclusterFeatures(Mat &featuresUnclustered);
	Forest buildIndex(vector<Feature>& appendFramesFeature);
	void replaceMembers(vector<string>& appendFramesPath, vector<Feature>& appendFramesFeature, Forest forest_);
	
	PCA pca;
	CodeBook codebook;
	DbFeature db;
	Forest forest;
	ImageRetrievalParam par;
	Lock m_Lock;
	//Lock m_search_Lock;
};

#endif 
