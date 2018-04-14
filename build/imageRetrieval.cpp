#include <iostream>
#include <fstream>
#include <iterator>
#include <cstdlib>
#include <map>
#include <config4cpp/Configuration.h>
#include "imageRetrieval.h"
//#include "hesaff.h"
#include "time.h"
#include "getorb.h"
#define ln 2.718282
extern "C"{
#include "vl/vlad.h"
#include "vl/gmm.h"
#include "vl/fisher.h"
}

using namespace config4cpp;


int ImageRetrieval::getParam(const char *configFile){
	Configuration *cfg = Configuration::create();
	const char *scope = "";
	try {
        cfg->parse(configFile);
		par.retrievalType = String(cfg->lookupString(scope, "retrievalType"));
        par.dictDim1 = cfg->lookupInt(scope, "dictDim1");
        par.dictDim2 = cfg->lookupInt(scope, "dictDim2");
        par.usePCA = cfg->lookupBoolean(scope, "usePCA");
		par.pcaDim = cfg->lookupInt(scope, "pcaDim");
		par.encodeType = String(cfg->lookupString(scope, "encodeType"));
		par.basePath = String(cfg->lookupString(scope, "basePath"));
		par.databasePath = String(cfg->lookupString(scope, "databasePath"));
    } catch(const ConfigurationException & ex) {
        cerr << ex.c_str() << endl;
        cfg->destroy();
        return 1;
    }
	cfg->destroy();
    cout << "-----------------------------------------------" << endl
		 << "-----------------------------------------------" << endl
		 << "retrieval type : " << par.retrievalType << endl
		 << "dimension of dictionary 1 : " << par.dictDim1 << endl
		 << "dimension of dictionary 2 : " << par.dictDim2 << endl
		 << "use pca to sift descriptor or not : " << (par.usePCA ? "yes" : "no") << endl
		 << "dimension of pca sift descriptor : " << par.pcaDim << endl
		 << "method of encode sift descriptor : " << par.encodeType << endl
		 << "path of codebook and others : " << par.basePath << endl
		 << "path of database : " << par.databasePath << endl
		 << "-----------------------------------------------" << endl
		 << "-----------------------------------------------" << endl;
    return 0;
}


void ImageRetrieval::init(){
	String dictionaryPath = par.basePath + "dictionary.yml";
	FileStorage fs(dictionaryPath, FileStorage::READ);
	fs["dictionary"] >> codebook.dictionary;
	if(!par.encodeType.compare("VLAD")){
		fs["dictionaryVlad"] >> codebook.dictionaryVlad;
		if (par.usePCA){
			fs["vectors"] >> pca.eigenvectors;
			fs["values"] >> pca.eigenvalues;
			fs["mean"] >> pca.mean;
		}
	}
	if(!par.encodeType.compare("FV")){
		fs["means"] >> codebook.means;
		fs["covariances"] >> codebook.covariances;
		fs["priors"] >> codebook.priors;
		if (par.usePCA){
			fs["vectors"] >> pca.eigenvectors;
			fs["values"] >> pca.eigenvalues;
			fs["mean"] >> pca.mean;
		}
	}
    fs.release();
}


void ImageRetrieval::generateUnclusterFeatures(Mat &featuresUnclustered){
	//to store the current input image
	//Mat input;
	int len = 0;
	const char *dir = par.databasePath.c_str();
	vector<string> filenames = list_all_image_in_file(dir);
	vector<int> ra1(filenames.size());
	for (size_t i = 0; i < filenames.size(); ++i)
		ra1[i] = i;
	size_t numFile = min((int)filenames.size(), 8000);
	if (numFile == 8000)
		random_shuffle( ra1.begin(), ra1.end() );
	cout << numFile << " image featute will be added !" << endl;

	/*float IMAGE_SQUARE;
	if (!par.retrievalType.compare("STATIC")){
		IMAGE_SQUARE = 307200.0;
	}
	else if (!par.retrievalType.compare("DYNAMIC")){
		IMAGE_SQUARE = 200000.0;
	}
	else {
		cerr << "there is not this retrieval type !" << endl;
		exit(1);
	}
	*/
    //ofstream in;
	//in.open("numkeypoints.txt",ios::trunc);
	for (size_t i=0; i < numFile; ++i){
		string filename = filenames[ra1[i]];
		/*
		input = imread(filename, CV_LOAD_IMAGE_COLOR); //Load as grayscale GRAYSCALE
		if (input.empty() || input.rows <= 0 || input.cols <= 0) 
			continue;
		
		double scale = sqrt(IMAGE_SQUARE/(float)(input.cols*input.rows));
		resize(input, input, Size(), scale, scale);
		printf("%s %d %d\n", filename.c_str(), input.cols, input.rows);
        */
		Mat descriptor_AffineHessian;
		Mat frame; //unused
		//getAffineHessianDescriptor(input, descriptor_AffineHessian, frame);
		getAffineHessianDescriptor(filename, descriptor_AffineHessian);
		if(descriptor_AffineHessian.rows==0||descriptor_AffineHessian.cols==0)
			continue;
		//in<<descriptor_AffineHessian.rows<<"\n";
		//put the all feature descriptors in a single Mat object
		vector<int> ra2(descriptor_AffineHessian.rows);
		for (size_t j = 0; j < descriptor_AffineHessian.rows; ++j)
			ra2[j] = j;
		size_t numDescriptor = min((int)descriptor_AffineHessian.rows, 200);
		if (numDescriptor == 200)
			random_shuffle( ra2.begin(), ra2.end() );
		for (size_t j = 0; j < numDescriptor; ++j)
			featuresUnclustered.push_back(descriptor_AffineHessian.row(ra2[j]));
		cout << (++len) << " images feature added ! ( " << ra1[i] << " , " << numDescriptor << " )" << endl;
	}
	//in.close();
	featuresUnclustered.convertTo(featuresUnclustered, CV_32F);
}

void kMeans(const Mat &featuresUnclustered, Mat &dictionary, int k){
	//retries number
	int retries = 1;
	//necessary flags
	int flags = KMEANS_PP_CENTERS;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER, 1000, 0.00001);
	//the number of bags
	int dictionarySize = k;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	//cluster the feature vectors
	dictionary = bowTrainer.cluster(featuresUnclustered); //1200*128
}

void GMM(const Mat &featuresUnclustered, Mat &means, Mat &covariances, Mat &priors, int k){
	VlGMM *gmm;
	vl_size numClusters = k;
	vl_size dimension = featuresUnclustered.cols;
	vl_size numData = featuresUnclustered.rows;
	float *data = (float *)featuresUnclustered.data;
	// create a new instance of a GMM object for float data
	gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numClusters);
	// cluster the data, i.e. learn the GMM
	vl_gmm_cluster(gmm, data, numData);
	// get the means, covariances, and priors of the GMM
	Mat means_tmp(numClusters, dimension, CV_32F, (float *)vl_gmm_get_means(gmm));
	Mat covariances_tmp(numClusters, dimension, CV_32F, (float *)vl_gmm_get_covariances(gmm));
	Mat priors_tmp(1, numClusters, CV_32F, (float *)vl_gmm_get_priors(gmm));
	means_tmp.copyTo(means);
	covariances_tmp.copyTo(covariances);
	priors_tmp.copyTo(priors);
	vl_gmm_delete(gmm);
}

void ImageRetrieval::buildDictionary(){

	//To store all the descriptors that are extracted from all the images.
	Mat featuresUnclustered;
	generateUnclusterFeatures(featuresUnclustered);
	
	printf("start building dictionary. Please wait a minute \n");
	//store the vocabulary
	creatDir(par.basePath.c_str());
	String dictionaryPath = par.basePath + "dictionary.yml";
	FileStorage fs(dictionaryPath, FileStorage::WRITE);
	if (!par.encodeType.compare("BOW")){
		kMeans(featuresUnclustered, codebook.dictionary, par.dictDim1);
	}
	else if(!par.encodeType.compare("VLAD")){
		kMeans(featuresUnclustered, codebook.dictionary, par.dictDim1);
		if (par.usePCA){
			pca = PCA(featuresUnclustered, Mat(), 0, par.pcaDim); 
			Mat pcaSIFT = pca.project(featuresUnclustered);
			kMeans(pcaSIFT, codebook.dictionaryVlad, par.dictDim2);
			fs << "vectors" << pca.eigenvectors;
			fs << "values" << pca.eigenvalues;
			fs << "mean" << pca.mean;
		}
		else
			kMeans(featuresUnclustered, codebook.dictionaryVlad, par.dictDim2);	
		fs << "dictionaryVlad" << codebook.dictionaryVlad;
	} 
	else if(!par.encodeType.compare("FV")){
		kMeans(featuresUnclustered, codebook.dictionary, par.dictDim1);
		if (par.usePCA){
			pca = PCA(featuresUnclustered, Mat(), 0, par.pcaDim); 
			Mat pcaSIFT = pca.project(featuresUnclustered);
			GMM(pcaSIFT, codebook.means, codebook.covariances, codebook.priors, par.dictDim2);
			fs << "vectors" << pca.eigenvectors;
			fs << "values" << pca.eigenvalues;
			fs << "mean" << pca.mean;
		}
		else
			GMM(featuresUnclustered, codebook.means, codebook.covariances, codebook.priors, par.dictDim2);	
		fs << "means" << codebook.means;
		fs << "covariances" << codebook.covariances;
		fs << "priors" << codebook.priors;
	}
	else{
		cerr << "there is not this encode method !" << endl;
		exit(1);
	}
	
	fs << "dictionary" << codebook.dictionary;
	fs.release();
	printf("\ndone\n");

}

Forest ImageRetrieval::buildIndex(vector<Feature>& appendFramesFeature)
{
  Forest forest;
	cout << endl << "start buld Feature Pool,please wait ......" << endl;

	//*********************** build kdtree **********************//
	int nPts = appendFramesFeature.size();
	int dim = appendFramesFeature[0].descriptor.cols;

  forest.data = new float[nPts * dim];
	for (size_t i = 0; i < nPts; i++){
		for (size_t j = 0; j < dim; j++){
			forest.data[i * dim + j] = appendFramesFeature[i].descriptor.ptr<float>(0)[j];
		}
	}
	forest.kdtree = vl_kdforest_new(VL_TYPE_FLOAT, dim, 1, VlDistanceL1);
	vl_kdforest_build(forest.kdtree, nPts, forest.data);

	cout << "build done !!! " << endl;
	return forest;
}

void ImageRetrieval::replaceMembers(vector<string>& appendFramesPath, vector<Feature>& appendFramesFeature, Forest forest_)
{
	WriteLock w_lock(m_Lock);
	
	if (forest.kdtree) {
    delete forest.data;
    vl_kdforest_delete(forest.kdtree);
		forest.data = NULL;
    forest.kdtree = NULL;
	}
	forest = forest_;
	
	db.imageName = appendFramesPath;
	db.imageFeature = appendFramesFeature;
}

void ImageRetrieval::addFeatureIntoPool(char* image_txt)
{
	vector<string> filenames = list_all_image_in_file(image_txt);

	vector<string> tmp_filenames;
	vector<Feature> tmp_featureDb;
	{
		ReadLock r_lock(m_Lock);
		tmp_filenames = db.imageName;
		tmp_featureDb = db.imageFeature;
	}
	
	for (size_t i=0; i<filenames.size(); ++i){
		String filename = filenames[i];
		cout << i <<"  "<< filename <<endl;
		Feature fea = getFeature(filename.c_str());	
		if (!fea.descriptor.empty()){
			tmp_filenames.push_back(filenames[i]);
			tmp_featureDb.push_back(fea);
		}
	}

	Forest forest_ = buildIndex(tmp_featureDb);
	replaceMembers(tmp_filenames, tmp_featureDb, forest_);
}

void ImageRetrieval::deleteFeatureFromPool(char* blackListText)
{
	vector<string> blackList = list_all_image_in_file(blackListText);

	map<string, int> hm;
	for (int i = 0; i < blackList.size(); ++i)
	{
		hm[blackList[i]] = 1;
	}

	vector<string> tmp_filenames;
	vector<Feature> tmp_featureDb;
	{
		ReadLock r_lock(m_Lock);

		for (int i = 0; i < db.imageName.size(); ++i)
		{
			if (hm.find(db.imageName[i]) == hm.end()) 
			{
				tmp_filenames.push_back(db.imageName[i]);
				tmp_featureDb.push_back(db.imageFeature[i]);
			}
		}
	}

	Forest forest_ = buildIndex(tmp_featureDb);
	replaceMembers(tmp_filenames, tmp_featureDb, forest_);

}
void cdm(int kk,float thres,DbFeature &db){
	int N=db.imageName.size();
	float S1=0;
	float S2=0;
	Mat deta(1,N,CV_32FC1,Scalar::all(1.0));
	Mat Distance( N, N, CV_32FC1, Scalar::all(0.0));
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			float distance=0.;
			for(int p=0;p<db.imageFeature[0].descriptor.cols;p++){
			distance=distance+abs(db.imageFeature[i].descriptor.ptr<float>(0)[p]-db.imageFeature[j].descriptor.ptr<float>(0)[p]);
			}
			Distance.ptr<float>(i)[j]=distance;
		}
	}
	Mat Dis;
	Distance.copyTo(Dis);
	int m=2;
	//iteration 1
	  vector<float> r1(N,0);
	  for(int i=0;i<N;i++){
		   vector<float> ith1(N);
		  for(int j=0;j<N;j++){
			  ith1[j]=Distance.ptr<float>(i)[j];
		  }
		  int max = (int)(max_element(ith1.begin(), ith1.end()) - ith1.begin());
		  for(int j=0;j<kk;j++){
			  int min=(int)(min_element(ith1.begin(), ith1.end()) - ith1.begin());
			  r1[i]=r1[i]+ith1[min];
			  ith1[min]=ith1[max];
		  }
		  r1[i]=r1[i]/kk;
	  }
	
	  float rr1=0.0;
	  for(int i=0;i<N;i++){
		  rr1=rr1+log(r1[i]);
	  }
	  rr1=rr1/N;
	  rr1=pow((float)ln, (float)rr1);
	  //cout<<rr1<<endl;
	  //cout<<"rr1"<<endl;

	  for(int i=0;i<N;i++){
		  deta.at<float>(0,i)=deta.at<float>(0,i)*sqrt(rr1/r1[i]);
	  }
      Mat diag(N,N,CV_32FC1, Scalar::all(0.0));
	  for(int i=0; i<N; i++) {
		  diag.ptr<float>(i)[i]=deta.ptr<float>(0)[i];
	  }
	  Distance=diag*Dis*diag;
	  for(int i=0;i<N;i++){
		  S1=S1+abs(r1[i]-rr1);
	  }
	  	  
	  //iteration 2
	  vector<float> r2(N,0);
	  for(int i=0;i<N;i++){
		   vector<float> ith2(N);
		  for(int j=0;j<N;j++){
			  ith2[j]=Distance.ptr<float>(i)[j];
		  }
		  int max = (int)(max_element(ith2.begin(), ith2.end()) - ith2.begin());
		  for(int j=0;j<kk;j++){
			  int min=(int)(min_element(ith2.begin(), ith2.end()) - ith2.begin());
			  r2[i]=r2[i]+ith2[min];
			  ith2[min]=ith2[max];
		  }
		  r2[i]=r2[i]/kk;
	  }
	  float rr2=0.0;
	  for(int i=0;i<N;i++){
		  rr2=rr2+log(r2[i]);
	  }
	  rr2=rr2/N;
	  rr2=pow((float)ln, (float)rr2);
	  for(int i=0;i<N;i++){
		  deta.at<float>(0,i)=deta.at<float>(0,i)*sqrt(rr2/r2[i]);
	  }
      // diag(N,N,CV_32FC1, Scalar::all(0.0))
	  for(int i=0;i<N;i++){
		  diag.ptr<float>(i)[i]=deta.ptr<float>(0)[i];
	  }
	  Distance=diag*Dis*diag;
	  for(int i=0;i<N;i++){
		  S2=S2+abs(r2[i]-rr2);
	  }
  
	while((S1-S2)>=thres)
	{	
	  float S=0;
	  m=m+1;
	  cout<<"iteration"<<m<<endl;
	  vector<float> r(N,0);
	  for(int i=0;i<N;i++){
		  vector<float> ith(N);
		  for(int j=0;j<N;j++){
			  ith[j]=Distance.ptr<float>(i)[j];
		  }
		  int max = (int)(max_element(ith.begin(), ith.end()) - ith.begin());
		  for(int j=0;j<kk;j++){
			  int min=(int)(min_element(ith.begin(), ith.end()) - ith.begin());
			  r[i]=r[i]+ith[min];
			  ith[min]=ith[max];
		  }
		  r[i]=r[i]/kk;
		  
		}	  
	  float rr=0.0;
	  for(int i=0;i<N;i++){
		  rr=rr+log(r[i]);
	  }
	  rr=rr/N;
	  rr=pow((float)ln, (float)rr);
	  
	  for(int i=0;i<N;i++){
		  deta.at<float>(0,i)=deta.at<float>(0,i)*sqrt(rr/r[i]);
	  }
      //Mat diag(N,N,CV_32FC1, Scalar::all(0.0))
	  for(int i=0;i<N;i++){
		  diag.ptr<float>(i)[i]=deta.ptr<float>(0)[i];
	  }
	  Distance=diag*Dis*diag;
	  for(int i=0;i<N;i++){
		  S=S+abs(r[i]-rr);
	  }
	  S1=S2;
	  S2=S;	  
	}
	 
	for(int i=0;i<N;i++){
		db.imageSigma.push_back(deta.ptr<float>(0)[i]);
	}
}

void ImageRetrieval::buildFeaturePool(){
	
	if (codebook.dictionary.empty())
		init();
	
	clock_t starttime = clock();
	vector<string> filenames = list_all_image_in_file(par.databasePath.c_str());
	for (size_t i=0; i<filenames.size(); ++i) {
		string filename = filenames[i];
		cout << i <<"  "<< filename <<endl;
		Feature fea = getFeature(filename.c_str());	
		
		if (!fea.descriptor.empty()) {
			db.imageName.push_back(filenames[i]);
			db.imageFeature.push_back(fea);
		}
	}
	clock_t endtime = clock();
	cout << "Size of database :  " << db.imageName.size() << endl;
	cout << "Extract feature time: " << (double)(endtime-starttime) / CLOCKS_PER_SEC << endl;
		
	int kk=10;
	float thres=0.00001;
	cdm(kk,thres,db);
	
	//*********************** build kdtree **********************//
    forest = buildIndex(db.imageFeature);
}


float geometricVerification(const Feature &fea1, const Feature &fea2){
	
	const Feature *p = 0;
	const Feature *q = 0;
	
	//sample
	Feature fea;
	fea.frame = Mat( 3, 800, CV_32F, Scalar::all(0.0) );
	fea.word = Mat( 1, 800, CV_32F, Scalar::all(0.0) );
	if (fea1.frame.cols > 800){
		vector<int> ra(fea1.frame.cols);
		for (size_t i = 0; i < fea1.frame.cols; i++)
			ra[i] = i;
		random_shuffle ( ra.begin(), ra.end() );
		for (size_t i = 0; i < 800; i++){
			fea.frame.ptr<float>(0)[i] = fea1.frame.ptr<float>(0)[ra[i]];
			fea.frame.ptr<float>(1)[i] = fea1.frame.ptr<float>(1)[ra[i]];
			fea.frame.ptr<float>(2)[i] = fea1.frame.ptr<float>(2)[ra[i]];
			fea.word.ptr<float>(0)[i] = fea1.word.ptr<float>(0)[ra[i]];
		}
		p = &fea; q = &fea2;
	}
	else{
		p = &fea1; q = &fea2;
	}
	
	//matchwords
	vector<vector<int>> matches(2);
	for (size_t i = 0; i < (p->word).cols; i++){
		for (size_t j = 0; j < (q->word).cols; j++){
			if ((p->word).ptr<float>(0)[i] == (q->word).ptr<float>(0)[j]){
				matches[0].push_back(i);
				matches[1].push_back(j);
				break;
			}
		}
	}
	
	//geometricVerification
	int numMatches = matches[0].size();
	if (numMatches == 0)
		return 0;
	vector<vector<int>> inliers(numMatches);
	Mat frame1(3, numMatches, CV_32F);
	Mat frame2(3, numMatches, CV_32F);
	for (size_t i = 0; i < numMatches; i++){
		frame1.ptr<float>(0)[i] = (p->frame).ptr<float>(0)[matches[0][i]];
		frame1.ptr<float>(1)[i] = (p->frame).ptr<float>(1)[matches[0][i]];
		frame1.ptr<float>(2)[i] = (p->frame).ptr<float>(2)[matches[0][i]];
		frame2.ptr<float>(0)[i] = (q->frame).ptr<float>(0)[matches[1][i]];
		frame2.ptr<float>(1)[i] = (q->frame).ptr<float>(1)[matches[1][i]];
		frame2.ptr<float>(2)[i] = (q->frame).ptr<float>(2)[matches[1][i]];
	}
	Mat x1hom = frame1.clone();
	Mat x2hom = frame2.clone();
	for (size_t i = 0; i < numMatches; i++){
		x1hom.ptr<float>(2)[i] = 1;
		x2hom.ptr<float>(2)[i] = 1;
	}
	
	Mat x1p;
	float tol;
	for (size_t m = 0; m < numMatches; ++m){
		for (size_t t = 0; t < 3; t++){
			if (t == 0){
				float m1[3][3] = { { frame1.ptr<float>(2)[m], 0, frame1.ptr<float>(0)[m] },
				{ 0, frame1.ptr<float>(2)[m], frame1.ptr<float>(1)[m] }, { 0, 0, 1 } };
				float m2[3][3] = { { frame2.ptr<float>(2)[m], 0, frame2.ptr<float>(0)[m] },
				{ 0, frame2.ptr<float>(2)[m], frame2.ptr<float>(1)[m] }, { 0, 0, 1 } };
				Mat A1 = Mat(3, 3, CV_32F, m1);
				Mat A2 = Mat(3, 3, CV_32F, m2);
				Mat A21 = A2 * A1.inv(DECOMP_SVD);
				x1p = A21.rowRange(0, 2)*x1hom;
				tol = 20 * sqrt(determinant(A21.rowRange(0, 2).colRange(0, 2)));
			}
			else{
				Mat A1(3, inliers[m].size(), CV_32F), A2(2, inliers[m].size(), CV_32F);
				for (size_t i = 0; i < inliers[m].size(); i++){
					A1.ptr<float>(0)[i] = (x1hom.ptr<float>(0)[inliers[m][i]]);
					A1.ptr<float>(1)[i] = (x1hom.ptr<float>(1)[inliers[m][i]]);
					A1.ptr<float>(2)[i] = (x1hom.ptr<float>(2)[inliers[m][i]]);
					A2.ptr<float>(0)[i] = (x2hom.ptr<float>(0)[inliers[m][i]]);
					A2.ptr<float>(1)[i] = (x2hom.ptr<float>(1)[inliers[m][i]]);
				}
				Mat A21 = A2 * A1.inv(DECOMP_SVD);
				x1p = A21 * x1hom;
				tol = 20 * sqrt(determinant(A21.rowRange(0, 2).colRange(0, 2)));
				inliers[m].clear();
			}
			Mat delta = x2hom.rowRange(0, 2) - x1p;
			for (size_t i = 0; i < numMatches; i++){
				if ((delta.ptr<float>(0)[i]*delta.ptr<float>(0)[i] + delta.ptr<float>(1)[i]*delta.ptr<float>(1)[i]) < tol*tol)
					inliers[m].push_back(i);
			}
			if (inliers[m].size() < 6)
				break;
			if (inliers[m].size() > 0.7 * numMatches)
				break;
		}
	}

	vector<int> score(numMatches);
	for (size_t i = 0; i < numMatches; i++){
		score[i] = inliers[i].size();
	}
	int best = (int)(max_element(score.begin(), score.end()) - score.begin());
	
	return (float)inliers[best].size()/(p->frame).cols;
}



RetrievalResult ImageRetrieval::retrievalImage(const char *imagePath, int k){
	
	RetrievalResult result;
	Feature fea = getFeature(imagePath);
	if (fea.descriptor.empty()){
		cout << "get feature error!\t";
		result.imagePath1 = "";
		result.imagePath2 = "";
		result.imagePath3 = "";
		result.imagePath4 = "";
		result.score = 0;
		return result;
	}
	
	
	
	//int N=db.imageName.size();
	//int total1=0;
	//int total2=0;
	//int total3=0;
	//int total4=0;
	
	for(size_t i=0;i<N;i++){
		float *queryPt = (float *)db.imageFeature[i].descriptor.data;
	    VlKDForestSearcher *searcher = vl_kdforest_new_searcher(forest.kdtree);
	    VlKDForestNeighbor *neighbours = new VlKDForestNeighbor[k];
	
	    int nvisited = vl_kdforestsearcher_query(searcher, neighbours, k, queryPt);
	  vector<float> score(k);
      for (size_t j = 0; j < k; j ++){
		//float scores = geometricVerification(fea,db.imageFeature[neighbours[i].index]);
		float dis=0;
		   for(int p=0;p<db.imageFeature[0].descriptor.cols;p++){
			   dis=dis+abs(db.imageFeature[neighbours[j].index].descriptor.ptr<float>(0)[p]-db.imageFeature[i].descriptor.ptr<float>(0)[p]);
			}
		
		   score[j]=dis*db.imageSigma[neighbours[j].index];
	  }
	  int queryid=atoi(db.imageName[i].substr(34,5).c_str());
	  int max=(int)(max_element(score.begin(), score.end()) - score.begin());
	  float value=score[max];
	  //cout<<"min"<<distances[min]<<endl;
	  int best = (int)(min_element(score.begin(), score.end()) - score.begin());
	  score[best]=value;
      result.imagePath1=db.imageName[neighbours[best].index].c_str();
	  if(atoi(db.imageName[neighbours[best].index].substr(34, 5).c_str())/4==queryid/4){
		  total1=total1+1;
		  total2=total2+1;
		  total3=total3+1;
		  total4=total4+1;
	  }
	  //cout<<db.imageName[i]<<"---"<<db.imageName[neighbours[best].index]<<endl;
	  
	  best = (int)(min_element(score.begin(), score.end()) - score.begin());
	  score[best]=value;
      result.imagePath2=db.imageName[neighbours[best].index].c_str();
	  if(atoi(db.imageName[neighbours[best].index].substr(34, 5).c_str())/4==queryid/4){
		  //total1=total1+1;
		  total2=total2+1;
		  total3=total3+1;
		  total4=total4+1;
	  }
	   //cout<<db.imageName[i]<<"---"<<db.imageName[neighbours[best].index]<<endl;
	   
	  best = (int)(min_element(score.begin(), score.end()) - score.begin());
	  score[best]=value;
      result.imagePath3=db.imageName[neighbours[best].index].c_str();
	   //cout<<db.imageName[i]<<"---"<<db.imageName[neighbours[best].index]<<endl;
	  if(atoi(db.imageName[neighbours[best].index].substr(34, 5).c_str())/4==queryid/4){
		  //total1=total1+1;
		  //total2=total2+1;
		  total3=total3+1;
		  total4=total4+1;
	  } 
	  best = (int)(min_element(score.begin(), score.end()) - score.begin());
	  score[best]=value;
      result.imagePath4=db.imageName[neighbours[best].index].c_str();
	  if(atoi(db.imageName[neighbours[best].index].substr(34, 5).c_str())/4==queryid/4){
		  //total1=total1+1;
		  //total2=total2+1;
		  //total3=total3+1;
		  total4=total4+1;
	  }
	  //cout<<db.imageName[i]<<"---"<<db.imageName[neighbours[best].index]<<endl;
	 
	  cout<<"search image:"<<i<<endl;
	  delete[] neighbours;
	  vl_kdforestsearcher_delete(searcher);
	}
	
	float *queryPt = (float *)fea.descriptor.data;
	VlKDForestSearcher *searcher = vl_kdforest_new_searcher(forest.kdtree);
	VlKDForestNeighbor *neighbours = new VlKDForestNeighbor[k];
	int nvisited = vl_kdforestsearcher_query(searcher, neighbours, k, queryPt);
	vector<float> score(k);
    for (size_t i = 0; i < k; i ++){
		//float scores = geometricVerification(fea,db.imageFeature[neighbours[i].index]);
		float dis=0;
		   for(int p=0;p<db.imageFeature[0].descriptor.cols;p++){
			   dis=dis+abs(db.imageFeature[neighbours[i].index].descriptor.ptr<float>(0)[p]-fea.descriptor.ptr<float>(0)[p]);
			}
		
		score[i]=dis*db.imageSigma[neighbours[i].index];
	}
	
      	
	  int max=(int)(max_element(score.begin(), score.end()) - score.begin());
	  float value=score[max];
	  //cout<<"min"<<distances[min]<<endl;
	  int best = (int)(min_element(score.begin(), score.end()) - score.begin());
	  score[best]=value;
      result.imagePath1=db.imageName[neighbours[best].index].c_str();
	  
	   best = (int)(min_element(score.begin(), score.end()) - score.begin());
	  score[best]=value;
      result.imagePath2=db.imageName[neighbours[best].index].c_str();
	   
	  best = (int)(min_element(score.begin(), score.end()) - score.begin());
	  score[best]=value;
      result.imagePath3=db.imageName[neighbours[best].index].c_str();
	   
	  best = (int)(min_element(score.begin(), score.end()) - score.begin());
	  score[best]=value;
      result.imagePath4=db.imageName[neighbours[best].index].c_str();   
	 
	   
	  result.score=0;
    delete[] neighbours;
	vl_kdforestsearcher_delete(searcher);
	
 	
	
	return result;
}


void getBOW(const Mat &word, const Mat &dictionary, Mat &bowDescriptor){
	
	int clusterCount = dictionary.rows;
	bowDescriptor = Mat( 1, clusterCount, CV_32F, Scalar::all(0.0) );
	float *dptr = (float*)bowDescriptor.data;
	for(size_t i = 0; i < word.cols; i++){
		int index = word.ptr<float>(0)[i];
		dptr[index] += 1.f;
	}
	// Normalize image descriptor.
	double norm_l2 = norm(bowDescriptor, NORM_L2, noArray());
	if ( norm_l2 != 0)
		bowDescriptor /= norm_l2;
}

void getVLAD(const Mat &descriptors, const Mat &dictionary, Mat &vladDescriptor){
	
	int clusterCount = dictionary.rows;
	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    matcher->add(vector<Mat>(1, dictionary));
	// Match keypoint descriptors to cluster center (to dictionary)
	vector<DMatch> matches;
	matcher->match(descriptors, matches);		
	
	float *assignments = (float *)vl_malloc(sizeof(float) * clusterCount * descriptors.rows);
	memset(assignments, 0, sizeof(float) * clusterCount * descriptors.rows);
	for(size_t i = 0; i < matches.size(); i++){
		int queryIdx = matches[i].queryIdx;
		int trainIdx = matches[i].trainIdx; // cluster index
		CV_Assert(queryIdx == (int)i);
		assignments[queryIdx * clusterCount + trainIdx] = 1.f;
	}

	vladDescriptor = Mat(1, clusterCount * descriptors.cols, CV_32F);
	vl_vlad_encode((float *)vladDescriptor.data, VL_TYPE_FLOAT,
			(void *)dictionary.data, descriptors.cols, clusterCount,
			(float *)descriptors.data, descriptors.rows,
			assignments,
			0);	

	vl_free(assignments);
}

void getFV(const Mat &descriptors, const Mat &means, const Mat &covariances, const Mat &priors, Mat &fvDescriptor){
	//fisher vector
	// allocate space for the encoding
	int dimension = means.cols;
	int numClusters = means.rows;
	fvDescriptor = Mat( 1 , 2 * numClusters * dimension, CV_32F, Scalar::all(0.0) );
	float *dataToEncode = (float *)descriptors.data;
	int numDataToEncode = descriptors.rows;
	// run fisher encoding
	vl_fisher_encode((float *)fvDescriptor.data, VL_TYPE_FLOAT,
			(void *)means.data, dimension, numClusters,
			(void *)covariances.data,
			(void *)priors.data,
			dataToEncode, numDataToEncode,
			VL_FISHER_FLAG_IMPROVED);
}

Feature ImageRetrieval::getFeature(const char *path){
	
	Feature feature;
    //To store the image file name
    char filename[1024];
    sprintf(filename, path);
    //read the image
	
    /*
	Mat img=imread(filename,CV_LOAD_IMAGE_COLOR);//GRAYSCALE
	
	if (img.empty() || img.rows <= 0 || img.cols <= 0){
		cout << "This image is null!" << endl;
        return feature;
	}
	
	float IMAGE_SQUARE;
	if (!par.retrievalType.compare("STATIC")){
		IMAGE_SQUARE = 307200.0;
	}
	
	else if (!par.retrievalType.compare("DYNAMIC")){
		IMAGE_SQUARE = 200000.0;
		bool detected = true;
		img = detectTV(img, detected);
	}
	else {
		cerr << "there is not this retrieval type !" << endl;
		exit(1);
	}
	
    double scale = sqrt(IMAGE_SQUARE/(float)(img.cols*img.rows));
    resize(img, img, Size(), scale, scale);
   */
	//To store the BoW (or BoF) representation of the image
    Mat descriptors;
    //getAffineHessianDescriptor(img, descriptors, feature.frame);
	getAffineHessianDescriptor(filename, descriptors);
	if (descriptors.rows==0){
		cout << "This image is null!" << endl;
        return feature;
	}
	
	//create a nearest neighbor matcher
	
	//Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
   // matcher->add( vector<Mat>(1, codebook.dictionary) );
	// Match keypoint descriptors to cluster center (to dictionary)
	//vector<DMatch> matches;
	//matcher->match( descriptors, matches );
	
	//feature.word = Mat( 1, descriptors.rows, CV_32F, Scalar::all(0.0) );
	//float *wptr = (float*)feature.word.data;
	
	//for( size_t i = 0; i < matches.size(); i++ ){
	//	int queryIdx = matches[i].queryIdx;
	//	int trainIdx = matches[i].trainIdx; // cluster index
	//	CV_Assert( queryIdx == (int)i );
	//	wptr[queryIdx] = trainIdx;
	//}
	
	if(!par.encodeType.compare("BOW")){
			getBOW(feature.word, codebook.dictionary, feature.descriptor);
	}
	else if(!par.encodeType.compare("VLAD")){
		if (par.usePCA){
			Mat pcaDescriptors = pca.project(descriptors);
			getVLAD(pcaDescriptors, codebook.dictionaryVlad, feature.descriptor);
		}
		else
			getVLAD(descriptors, codebook.dictionaryVlad, feature.descriptor);
	}
	else if(!par.encodeType.compare("FV")){
		if (par.usePCA){
			
			Mat pcaDescriptors = pca.project(descriptors);
			
			getFV(pcaDescriptors, codebook.means, codebook.covariances, codebook.priors, feature.descriptor);
			
		}
		else
			getFV(descriptors, codebook.means, codebook.covariances, codebook.priors, feature.descriptor);
	}
	
	else{
		cerr << "there is not this encode method !" << endl;
		exit(1);
	}
	
    return feature;						

}


void ImageRetrieval::saveFeaturePool(const char *featureFile) {
	FileStorage fs(featureFile, FileStorage::WRITE);
	fs << "imageName" <<"[";
	for (size_t i = 0; i < db.imageName.size(); ++i)
		fs << db.imageName[i];
	fs << "]" << "imageFeature" << "{" << "descriptor" << "[";
	for (size_t i = 0; i < db.imageFeature.size(); ++i)
		fs << db.imageFeature[i].descriptor;
	fs << "]" << "frame" << "[";
	for (size_t i = 0; i < db.imageFeature.size(); ++i)
		fs << db.imageFeature[i].frame;
	fs << "]" << "word" << "[";
	for (size_t i = 0; i < db.imageFeature.size(); ++i)
		fs << db.imageFeature[i].word;
	fs << "]" << "}";
	fs << "imageSigma" <<"[";
	for(size_t i = 0; i < db.imageSigma.size(); ++i)
		fs << db.imageSigma[i];
	fs << "]";
	fs.release();
}


void ImageRetrieval::loadFeaturePool(const char *featureFile) {
	
	if (codebook.dictionary.empty())
		init();
	
	db.imageName.clear();
	db.imageFeature.clear();
	db.imageSigma.clear();
	cout << endl << "start load feature,please wait ......" << endl;
	FileStorage fs(featureFile, FileStorage::READ);
	FileNode fn1 = fs["imageName"]; 
	FileNodeIterator it1 = fn1.begin(), it1_end = fn1.end(); 
	for (; it1 != it1_end; ++it1)
		db.imageName.push_back((String)*it1);
	
	db.imageFeature.resize(db.imageName.size());
	FileNode fn2 = fs["imageFeature"]["descriptor"]; 
	FileNodeIterator it2 = fn2.begin(), it2_end = fn2.end(); 
	for (size_t i = 0; it2 != it2_end; ++it2, ++i)
		(*it2) >> db.imageFeature[i].descriptor;
	FileNode fn3 = fs["imageFeature"]["frame"]; 
	FileNodeIterator it3 = fn3.begin(), it3_end = fn3.end(); 
	for (size_t i = 0; it3 != it3_end; ++it3, ++i)
		(*it3) >> db.imageFeature[i].frame;
	FileNode fn4 = fs["imageFeature"]["word"]; 
	FileNodeIterator it4 = fn4.begin(), it4_end = fn4.end(); 
	for (size_t i = 0; it4 != it4_end; ++it4, ++i)
		(*it4) >> db.imageFeature[i].word;
	
	FileNode fn5 = fs["imageSigma"]; 
	FileNodeIterator it5 = fn5.begin(), it5_end = fn5.end(); 
	for (; it5 != it5_end; ++it5)
		db.imageSigma.push_back((float)*it5);
	
	fs.release();
	cout << "finish load feature ." << endl;

	
	//*********************** build kdtree **********************//
     forest = buildIndex(db.imageFeature);
}






