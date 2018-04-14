#include <iostream>
#include <fstream> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "getorb.h"

#include<fstream>    // 文件流 
#include<iostream>   // 标准流 
#include<string>     // 字符串 
#include <sstream>


using namespace cv;
using namespace std;


void getAffineHessianDescriptor(string ImagePath, Mat& descriptor){
	
	//char drive[128];
	//char dir[128];
	//char fname[32];
	//char ext[_MAX_EXT];
	//char targetfilename[_MAX_EXT];
	//_splitpath_s(ImagePath.c_str(), drive, dir, fname, ext);
	//string sfname = fname;
	//string path="/home/ckp/data/siftfeature/"+sfname+".sift";
	ifstream inFile;
	inFile.open(ImagePath);  // 打开文件 

	string str;    	// 字符串 
	float a[133];
	int i = 0;        // 列 
	getline(inFile, str);
	getline(inFile, str);
	istringstream sstr(str);
	sstr >> a[0];
	int numkey = a[0];
	descriptor=Mat(numkey, 128, CV_32FC1, Scalar::all(0.0));
	//frame = Mat( 3, numkey, CV_32FC1, Scalar::all(0.0) );
	int j = 0;
	if (inFile.is_open())
	{     // 若成功打开文件 
		
		
		while (!inFile.eof())
		{ // 若未到文件结束 
			i = 0;
			getline(inFile, str); // 读取一行内容，存入存str中 

			istringstream istr(str);
			while (!istr.eof())
			{

				istr >> a[i];
				if (i >= 5){
					descriptor.ptr<float>(j)[i - 5] = a[i];
				}
				i++;
			}
			j++;
		}
		
		inFile.close();  // 关闭文件
	}
	//FileStorage fs("C:/Users/123/Desktop/img1.xml", FileStorage::WRITE);
	//fs << "Mat" << des;
	//fs.release();
	//cout << "done" << endl;		
}
