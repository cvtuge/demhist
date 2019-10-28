#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <stdio.h>
#include <vector>
#include <dirent.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <boost/filesystem.hpp>
#include <sys/stat.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define BIN_THRESHOLD 20
#define sigma 1.5  //定义sigma的大小越大越模糊
#define PI 3.14159
#define N 5 //定义卷积核大小

using namespace boost::filesystem;

using namespace cv;
using namespace std;


directory_iterator end_iter;

int _brightness = 100;
int _contrast = 100;

int hist_size = 64;
float range_0[]={0,256};
float* ranges[] = { range_0 };
uchar lut[256];
CvMat* lut_mat;

int update_brightcont(int _brightness, int _contrast, IplImage *src_image, IplImage *dst_image, CvHistogram *hist, IplImage *hist_image)
{
    int brightness = _brightness - 100;
    int contrast = _contrast - 100;
    int i, bin_w;
    float max_value = 0;

    if( contrast > 0 )
    {
        double delta = 127.*contrast/100;
        double a = 255./(255. - delta*2);
        double b = a*(brightness - delta);
        for( i = 0; i < 256; i++ )
        {
            int v = cvRound(a*i + b);
            if( v < 0 )
                v = 0;
            if( v > 255 )
                v = 255;
            lut[i] = (uchar)v;
        }
    }
    else
    {
        double delta = -128.*contrast/100;
        double a = (256.-delta*2)/255.;
        double b = a*brightness + delta;
        for( i = 0; i < 256; i++ )
        {
            int v = cvRound(a*i + b);
            if( v < 0 )
                v = 0;
            if( v > 255 )
                v = 255;
            lut[i] = (uchar)v;
        }
    }

    cvLUT( src_image, dst_image, lut_mat );

	return 1;
    //cvCalcHist( &dst_image, hist, 0, NULL );

    //cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );
    //cvScale( hist->bins, hist->bins, ((double)hist_image->height)/max_value, 0 );

    //cvSet( hist_image, cvScalarAll(255), 0 );
    //bin_w = cvRound((double)hist_image->width/hist_size);
	//int NonZeroBinsNum = 0;
	//for (i = 0; i < hist_size; i++){
		//cvRectangle(hist_image, cvPoint(i*bin_w, hist_image->height),
			//cvPoint((i + 1)*bin_w, hist_image->height - cvRound(cvGetReal1D(hist->bins, i))),
			//cvScalarAll(0), -1, 8, 0);
		//if (cvGetReal1D(hist->bins, i) > 10)
			//NonZeroBinsNum++;
	//}
	//if (NonZeroBinsNum > BIN_THRESHOLD)
		//return 1;
	//else
		//return 0;
}

void SplitString(const string& s, vector<string>& v, const string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

void getJustCurrentDir(string path,vector<string>& dirs)
{
	directory_iterator end_iter1;
	for(directory_iterator img_itr(path); img_itr != end_iter1; img_itr++)
    {
      string dir = img_itr->path().string();
      if(is_directory(dir))
      {
        //cout << dir << endl;
        dirs.push_back(dir);
      }
    }
}

void getJustCurrentfile(string path,vector<string>& files)
{
	DIR* dir=opendir(path.c_str());
	dirent* p = NULL;
	while((p = readdir(dir)) != NULL)
    {       
        if(p->d_name[0] != '.')
        {
            //string name = dir+ "/" + string(p->d_name);
			string name = string(p->d_name);
			files.push_back(name);
            //cout<<name<<endl;
        }
    }
	closedir(dir);
}

/*
 * 参考链接：基于opencv2利用卷积算子实现高斯模糊：https://blog.csdn.net/ling_robe/article/details/79970084
 */
int img_blur(Mat& src, Mat& dst)
{
	float liv_conv[N][N] = {0}; //定义卷积核 
	cv::Mat lMv_gray = cv::Mat::zeros(src.rows, src.cols, CV_8UC3); //定义灰度图
	cv::cvtColor(src,lMv_gray,CV_BGR2GRAY);//将彩色图转换灰度图
 
	int liv_n = N/2; //获取半径
	float all = 0.0; 
 
	/**************
	产生卷积核
	***************/
	for(int i = 0; i<N; i++)
	{
		for(int j = 0; j<N; j++)
		{
			liv_conv[i][j] = exp(-((i-liv_n)*(i-liv_n)+(j-liv_n)*(j-liv_n))/(2.0*sigma*sigma))/(2*PI*sigma*sigma);//二维正态分布公式
			all+=liv_conv[i][j];
		}
	}
	/******************
	逐个像素进行卷积
	*******************/
	for(int i = 0; i<lMv_gray.rows-N; i++)
	{
		for(int j = 0; j<lMv_gray.cols-N; j++)
		{
			float lfv_sum_b = 0.0, lfv_sum_g = 0.0, lfv_sum_r = 0.0;
			for(int y = 0; y<N; y++)
			{
				for(int x = 0; x<N; x++)
				{
					lfv_sum_b+=src.at<cv::Vec3b>(i+y,j+x)[0]*liv_conv[y][x];//相称求和
					lfv_sum_g+=src.at<cv::Vec3b>(i+y,j+x)[1]*liv_conv[y][x];
					lfv_sum_r+=src.at<cv::Vec3b>(i+y,j+x)[2]*liv_conv[y][x];
				}
			}
			dst.at<cv::Vec3b>(i,j)[0] = lfv_sum_b/all;
			dst.at<cv::Vec3b>(i,j)[1] = lfv_sum_g/all;
			dst.at<cv::Vec3b>(i,j)[2] = lfv_sum_r/all;
		}
	}

	return 0;
}

 
int main(int argc,char* argv[])
{
    //cout << "Hello OpenCV " << CV_VERSION << endl;
	int flag;
	string src_image_path;
	string faces_dir = argv[1];
	string save_root = argv[2];
	//string faces_dir="/data/xlm/Origin_data/201609/";
	//string save_root="/data/xlm/Precess_data/201609/";
    vector<string> dirs,files,str1;
    
    getJustCurrentDir(faces_dir,dirs);
    for(int i=0;i<dirs.size();i++)
    {
		//cout << dirs[i] << endl;
		SplitString(dirs[i], str1, "/");
		string foldername = str1[str1.size()-1];
		string newdir=save_root+foldername;
		mkdir(newdir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		//cout << foldername << endl;
		str1.clear();
        getJustCurrentfile(dirs[i],files);
		for(int j=0;j<files.size();j++)
		{
			CvHistogram *hist;
			IplImage *src_image = 0, *dst_image = 0, *hist_image = 0;
			src_image_path = dirs[i] + "/" + files[j];
			//Mat src_gray = cv::imread(src_image_path , 0);
			//imwrite((newdir + "/gray_" + files[j]).c_str(), src_gray);
			
			src_image = cvLoadImage(src_image_path.c_str(), 1);
			dst_image = cvCloneImage(src_image);
			hist_image = cvCreateImage(cvSize(320, 200), 8, 1);
			hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
			lut_mat = cvCreateMatHeader(1, 256, CV_8UC1);
			cvSetData(lut_mat, lut, 0);
			_brightness = 150;
			_contrast = 100;
			flag = update_brightcont(_brightness, _contrast, src_image, dst_image, hist, hist_image);
			if (flag)
			{
				imwrite((newdir + "/1_" + files[j]).c_str(), cv::cvarrToMat(dst_image));//cvSaveImage
				//cout << newdir + "/1_" + files[j] << endl;
			}
			
			_brightness = 50;
			_contrast = 100;
			flag = update_brightcont(_brightness, _contrast, src_image, dst_image, hist, hist_image);
			if (flag)
			{
				imwrite((newdir + "/2_" + files[j]).c_str(), cv::cvarrToMat(dst_image));
				//cout << newdir + "/2_" + files[j] << endl;
			}
			
			_brightness = 100;
			_contrast = 150;
			flag = update_brightcont(_brightness, _contrast, src_image, dst_image, hist, hist_image);
			if (flag)
			{
				imwrite((newdir + "/3_" + files[j]).c_str(), cv::cvarrToMat(dst_image));
				//cout << newdir + "/3_" + files[j] << endl;
			}
			
			_brightness = 100;
			_contrast = 50;
			flag = update_brightcont(_brightness, _contrast, src_image, dst_image, hist, hist_image);
			if (flag)
			{
				imwrite((newdir + "/4_" + files[j]).c_str(), cv::cvarrToMat(dst_image));
				//cout << newdir + "/4_" + files[j] << endl;
			}

			//增加原图
			Mat src_mat = cv::cvarrToMat(src_image);
			imwrite((newdir + "/0_" + files[j]).c_str(), src_mat);

			//增加高斯模糊图片
			cv::Mat dst_mat = cv::Mat::zeros(src_mat.rows, src_mat.cols, CV_8UC3);
			img_blur(src_mat,dst_mat);
			imwrite((newdir + "/5_" + files[j]).c_str(), dst_mat);
	
			cvReleaseImage(&src_image);
			cvReleaseImage(&dst_image);
			cvReleaseImage(&hist_image);
			cvReleaseHist(&hist);
			
			//cv::imshow("test",src_gray);
			//waitKey(0);
			//cout<<files[j]<<endl;
		}
		files.clear();
    }
    return 0;
}
