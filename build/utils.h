#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

Mat detectTV(const Mat &src, bool& isDetect);
void creatDir(const char *dir);
vector<string> list_all_file_in_folder(const char * pfold);
vector<string> list_all_image_in_file(const char * images_txt);
int getLogoId(String filename);
bool isAvailability();