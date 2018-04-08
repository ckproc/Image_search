// -*- c++ -*- 

%module imageRetrieval;

%{
#include "imageRetrieval.h"
%}

struct RetrievalResult{
	const char *imagePath;
	float score;
};

class ImageRetrieval
{
public:
	ImageRetrieval(const char *configFile);
	~ImageRetrieval();
	void buildDictionary();
	void buildFeaturePool();
	void addFeatureIntoPool(char* dir);
	void deleteFeatureFromPool(char* blackListText);
	RetrievalResult retrievalImage(const char *imagePath, int k = 100);

	void saveFeaturePool(const char *featureFile);
	void loadFeaturePool(const char *featureFile);

};