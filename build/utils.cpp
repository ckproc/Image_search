#include "utils.h"
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static String sEndTime ="2018-1-1";


void creatDir(const char *dir){
	String s_dir(dir);
	int len = s_dir.length();
	if (s_dir[len - 1] != '/'){
		s_dir = s_dir + '/';
		len += 1;
	}
	for (int i = 0; i < len; ++i){
		if (s_dir[i] == '/'){
			const char *subDir = s_dir.substr(0,i).c_str();
			if (opendir(subDir) == 0){
				mkdir(subDir, 0755);
			}
		}
	}
}

bool isInclude(const char* filename, vector<String> extense){
	for (size_t i = 0; i < extense.size(); ++i){
		if (strstr(filename, extense[i].c_str()) != NULL)
			return true;
	}
	return false;
}

vector<string> list_all_file_in_folder(const char * pfold){
	printf("Target folder : %s \n", pfold);
	vector<String> extense{".jpg",".png",".bmp",".JPG",".PNG",".jpeg"};
	vector<String> fileNames = vector<String>();
	DIR* dp;
	struct dirent *dirp;
	if ((dp = opendir(pfold)) == NULL) {
		cout<<"ERROR:" << String(pfold) << "cannot opening" << endl;
		return fileNames;
	}
	
	while ((dirp = readdir(dp)) != NULL) {
		if (extense.size() == 0 || isInclude(dirp->d_name, extense)) {
			fileNames.push_back(String(dirp->d_name));		
		}	
	}
	return fileNames; 
}

vector<string> list_all_image_in_file(const char * images_txt) {
	vector<string> imagesList;

	ifstream myfile(images_txt);
	copy(istream_iterator<string>(myfile), istream_iterator<string>(), back_inserter(imagesList));

	return imagesList;
}

int getLogoId(string filename){
	int i = 0;
	while(filename[i] >= 48 && filename[i] <= 57)
		i ++;
	return atoi(filename.substr(0,i).c_str());
}

time_t str_to_time_t(const string& ATime, const string& AFormat="%d-%d-%d");
time_t NowTime();
bool IsValidTime(const time_t& AEndTime, const time_t& ANowTime );
bool isAvailability() {
    string sTemp;
    time_t t_Now = NowTime();
    time_t t_End = str_to_time_t(sEndTime);
    return IsValidTime(t_End, t_Now);
}

time_t str_to_time_t(const string& ATime, const string& AFormat)  {  
    struct tm tm_Temp;  
    time_t time_Ret;  
    try 
    {
        int i = sscanf(ATime.c_str(), AFormat.c_str(),// "%d/%d/%d %d:%d:%d" ,       
            &(tm_Temp.tm_year),   
            &(tm_Temp.tm_mon),   
            &(tm_Temp.tm_mday),  
            &(tm_Temp.tm_hour),  
            &(tm_Temp.tm_min),  
            &(tm_Temp.tm_sec),  
            &(tm_Temp.tm_wday),  
            &(tm_Temp.tm_yday));  
        tm_Temp.tm_year -= 1900;  
        tm_Temp.tm_mon --;  
        tm_Temp.tm_hour=0;  
        tm_Temp.tm_min=0;  
        tm_Temp.tm_sec=0;  
        tm_Temp.tm_isdst = 0;
        time_Ret = mktime(&tm_Temp);  
        return time_Ret;  
    } catch(...) {
        return 0;
    }
}  

time_t NowTime(){
    time_t t_Now = time(0);
    struct tm* tm_Now = localtime(&t_Now);
    tm_Now->tm_hour =0;
    tm_Now->tm_min = 0;
    tm_Now->tm_sec = 0;
    return  mktime(tm_Now);  
}

bool IsValidTime(const time_t& AEndTime, const time_t& ANowTime ){
    return (AEndTime >= ANowTime);
}

Mat transformToRectangle(const Mat & image, const vector<Point> sq)
{
	Point cent = Point( (float)(sq[0].x+sq[1].x+sq[2].x+sq[3].x)/4,
		(float)(sq[0].y+sq[1].y+sq[2].y+sq[3].y)/4 );
	float top=0, bot=0, rig=0, lef=0;

	Point2f src[4];

	for (size_t i=0; i<4; ++i)
	{
		if ( sq[i].x < cent.x )
			lef += sq[i].x;
		else
			rig += sq[i].x;

		if ( sq[i].y < cent.y )
			top += sq[i].y;
		else
			bot += sq[i].y;

		if ( sq[i].x<cent.x & sq[i].y<cent.y )
			src[0] = sq[i];
		if ( sq[i].x>cent.x & sq[i].y<cent.y )
			src[1] = sq[i];
		if ( sq[i].x>cent.x & sq[i].y>cent.y )
			src[2] = sq[i];
		if ( sq[i].x<cent.x & sq[i].y>cent.y )
			src[3] = sq[i];
	}

	lef /= 2;
	rig /= 2;
	bot /= 2;
	top /= 2;

	Point2f dst[4];
	dst[0].x = lef;
	dst[0].y = top;
	dst[1].x = rig;
	dst[1].y = top;
	dst[2].x = rig;
	dst[2].y = bot;
	dst[3].x = lef;
	dst[3].y = bot;

	Mat t;		// the transformation matrix
	t = getPerspectiveTransform(src,dst);  

	Mat image_wrap;
	warpPerspective(image, image_wrap, t, Size(image.cols, image.rows) );

	Mat tv = Mat( image_wrap, Range(top,bot), Range(lef,rig) );

	return tv;
}


float meanVal(const Mat & m)
{
	double v = 0;
	for (size_t r = 0; r < m.rows; ++r)
	{
		for (size_t c = 0; c < m.cols; ++c)
		{
			v += m.at<unsigned char>(r,c);
		}
	}
	v /= (double)( m.rows*m.cols );
	return v;
}

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double angle( Point pt1, Point pt2, Point pt0 )
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

Rect fitRect(const vector<Point> & sq)
{
	int minx, miny, maxx, maxy;
	for(size_t j=0; j<sq.size(); ++j)
	{
		int helpx = sq[j].x;
		int helpy = sq[j].y;
		if(0 == j)
		{
			minx = maxx = helpx;
			miny = maxy = helpy;
		}
		else
		{
			minx = min(helpx,minx);
			miny = min(helpy,miny);
			maxx = max(helpx,maxx);
			maxy = max(helpy,maxy);
		}
	}
	Rect r(minx, miny, maxx-minx+1, maxy-miny+1);

	return r;
}

// filter TV screen squre, 3 conditions are constraint:
// 1. any corner of TV screen don't locate near the ordinate origin
// 2. TV target always contain the mid point of img;
// 3. H/W radio range
static bool isTvScreen(const vector<Point>& square, Point & ptMid)
{
	Rect rect = fitRect(square);
	double ratio = (double)max(rect.width,rect.height) / min(rect.width,rect.height);

	//Check the W/H radio, common radio are 4:3 and 16:9. we extend the range to match more scene.
	if( ! (ratio >=1.2 && ratio <= 2) )
	{
		return false;
	}

	
	if (! (ptMid.x>rect.x+rect.width*0.2  & ptMid.x<rect.x+rect.width*0.8 
		 & ptMid.y>rect.y+rect.height*0.2 & ptMid.y<rect.y+rect.height*0.8) )
	{
		return false;
	}

	return true;

}

// find TV screen in a binary image
void findSquaresCore(const Mat& bw, vector<vector<Point> >& squares )
{
	vector<vector<Point> > contours;
	findContours(bw, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	for( int i = 0; i < contours.size(); i++ )
	{
		vector<Point> hull;
		convexHull( Mat(contours[i]), hull, false ); 

		vector<Point> approx;
		approxPolyDP(Mat(hull), approx, arcLength(Mat(hull), true)*0.1, true);

		float area = fabs(contourArea(Mat(approx)));
		if (approx.size() == 4 && // convexHullArea(approx) > 1000)
			area > 10000 && 
			area < 0.8*bw.rows*bw.cols )
		{
			double maxCosine = 0;

			for( int j = 2; j < 5; j++ )
			{
				// find the maximum cosine of the angle between joint edges
				double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
				maxCosine = MAX(maxCosine, cosine);
			}

			// if cosines of all angles are small
			// (all angles are ~90 degree) then write quandrange
			// vertices to resultant sequence
			if( maxCosine < 0.1 )
			{
				Point point = Point(bw.cols/2,bw.rows/2);
				if( isTvScreen(approx, point) )	
				{
					squares.push_back(approx);
				}
			}

		}
	}
}


typedef pair<int,float> ivpair;
bool ivcompare( const ivpair& l, const ivpair& r)
{ 
	return l.second > r.second; 
}

// get squares from the image
vector<Point> getTopSquare(const vector<vector<Point>> squares, const Mat image)
{
	vector<ivpair> areas;

	// fit a rectangle for every square
	for(size_t i=0; i<squares.size(); ++i)
	{
		ivpair helppair;
		helppair.first = i;
		helppair.second = fabs(contourArea(Mat(squares[i])));
		areas.push_back( helppair );
	}

	// sort the areas of squares in a descend order
	sort(areas.begin(), areas.end(), ivcompare);
	int besti = areas[0].first;

	return squares[besti];

}

/*!
 ÌáÈ¡canny±ßÔµ
 */
Mat cannyEdge(const Mat & gray)
{
	float t_canny = meanVal(gray);
	Mat ed;
	Canny(gray, ed, t_canny*0.66, t_canny*1.33);
	return ed;
}

cv::Mat detectTV(const cv::Mat &src, bool& isDetect) {

	int bdev = 0;

	Mat image = src;

	float s = (float) 600 / ( image.rows>image.cols ? image.rows : image.cols );
	resize( image, image, Size((int)image.cols*s,(int)image.rows*s) );

	// a container for the detected squares
	vector<vector<Point> > squares;
	bool useOtsu = true;
	bool useCanny = true;
	bool useSobel = true;
	bool useLocalCanny = true;
	bool useMultiThresh = true;

	Mat img2;
	cvtColor( image, img2, CV_BGR2HSV );
	Mat gray;
	cvtColor( image, gray, CV_RGB2GRAY );
	// GaussianBlur( gray, gray, Size(3,3), 0, 0, BORDER_DEFAULT );
	equalizeHist( gray, gray );

	// compute gradient
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad;
	Sobel( gray, grad_x, CV_16S, 1, 0 );
	Sobel( gray, grad_y, CV_16S, 0, 1 );
	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	Mat bw;
	threshold(gray, bw, 0, 255, THRESH_OTSU);

	if (useOtsu)
	{
		findSquaresCore( bw, squares );
	}

	if (useCanny)
	{
		Mat ed = cannyEdge(gray);
		dilate( ed, ed, Mat() );

		findSquaresCore( ed, squares );
	}

	if (useSobel)
	{
		Mat ed_sobel;
		ed_sobel = grad > meanVal(grad);

		findSquaresCore( ed_sobel, squares );
	}

	if (useLocalCanny)
	{
		int step = 40;
		Mat ed3(gray.rows, gray.cols, CV_8U);

		for( int r=0; r<gray.rows-step; r+=step )
		{
			for (int c=0; c<gray.cols-step; c+=step )
			{
				Range row_range(r, r+step);
				Range col_range(c, c+step);

				Mat small_image(gray, row_range, col_range);
				Mat small_ed = cannyEdge(small_image);

				for (int i=0; i<step; ++i)
				{
					for (int j=0; j<step; ++j)
					{
						ed3.at<uchar>(row_range.start+i, col_range.start+j)
							= small_ed.at<uchar>(i,j);
					}
				}
			}
		}
		dilate( ed3, ed3, Mat() );
		erode( ed3, ed3, Mat() );

		findSquaresCore( ed3, squares );
	}

	if (useMultiThresh)
	{
		int M = 8;
		for (int i=0; i<M; ++i )
		{
			Mat helpbw = gray > (255*(float)i/((float)(M+1)));
			findSquaresCore( helpbw, squares );
		}
	}

	if ( squares.size() >= 1 )
	{
		isDetect = true;
		vector<Point> topSquare;
		topSquare = getTopSquare(squares, image);
		
		// do a perspective transformation, and return the tv window
		Mat tv = transformToRectangle(image, topSquare);
		return tv;
	}
	else
	{
		isDetect = false;
		return image;
	}

}




