library:imageRetrieval_wrap.cxx
	/usr/bin/g++ -fPIC -shared -std=c++0x -I/usr/include/python2.6/ -I/usr/local/include -I/usr/local/include/opencv -I/usr/local/include/opencv2 -I. -Ihesaff -I./vlfeat-0.9.20 -Iconfig/include -L/usr/local/lib -L./vlfeat-0.9.20/bin/glnxa64/ -Lconfig/lib/ hesaff/*.cpp utils.cpp imageRetrieval.cpp imageRetrieval_wrap.cxx -O2 -o _imageRetrieval.so -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_features2d -lopencv_legacy -lopencv_nonfree -lpng -ljpeg -lpthread -lvl -lconfig4cpp -lboost_thread
	cp _imageRetrieval.so ../sample/
	cp imageRetrieval.py ../sample/

imageRetrieval_wrap.cxx:imageRetrieval.i
	swig -python -c++ imageRetrieval.i