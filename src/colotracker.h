#ifndef COLOTRACKER_H
#define COLOTRACKER_H

#include "cv.h"
#include "highgui.h"
#include "region.h"
#include "histogram.h"
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/background_segm.hpp>

//#define SHOWDEBUGWIN

#define BIN_1 16
#define BIN_2 16
#define BIN_3 16

class ColorTracker
{
private:
    BBox lastPosition;
    int _maxdatanum{3000};
    int _maxiter{10};
    double _orig_qb_simi{0};
    double _cur_batta_q{0};
    double _cur_batta_b{0};
    bool _record_batta{false};
    bool _lose_target{false};
    int _lose_count{0};
    int _bodyratio{1};
    double _backvalid_thresh{0.05};
    cv::Mat _img_old;
    bool _saveoldimg{false};
    double segment_u{0.15};
    double segment_d{0.6};

    cv::Mat im1;
    cv::Mat im2;
    cv::Mat im3;

    cv::Mat im1_old;
    cv::Mat im2_old;
    cv::Mat im3_old;

    Histogram q_hist;
    Histogram q_orig_hist;
    Histogram b_hist;

    double defaultWidth;
    double defaultHeight;

    double wAvgBg;
    double bound1;
    double bound2;

    cv::Point histMeanShift(double x1, double y1, double x2, double y2);
    cv::Point histMeanShiftIsotropicScale(double x1, double y1, double x2, double y2, double * scale, int * msIter = NULL);
    cv::Point histMeanShiftIsotropicScale_c(const cv::Mat &target,double x1, double y1, double x2, double y2, double * scale,int maxiter=4, int * msIter = NULL);
    cv::Point histMeanShiftAnisotropicScale(double x1, double y1, double x2, double y2, double * width, double * height);

    void preprocessImage(cv::Mat & img);
    void extractBackgroundHistogram(int x1, int y1, int x2, int y2, Histogram &hist);
    void extractForegroundHistogram(int x1, int y1, int x2, int y2, Histogram &hist);
    void extractForegroundHistogram_c(const cv::Mat &target,int x1, int y1, int x2, int y2, Histogram &hist);
    void extractForegroundHistogram(const cv::Mat &target, Histogram &hist);
    void extractBackgroundHistogram(const cv::Mat &target,int x1, int y1, int x2, int y2, Histogram &hist);

    bool reSearchTarget(cv::Mat& img,BBox* box);//global search
    double calcProbDist(double q,double b);//calculate probability distance to target object

    void tgColorDistribution(int x1, int y1, int x2, int y2, Histogram hist);


    inline double kernelProfile_Epanechnikov(double x)
        { return (x <= 1) ? (2.0/3.14)*(1-x) : 0; }
        //{ return (x <= 1) ? (1-x) : 0; }
    inline double kernelProfile_EpanechnikovDeriv(double x)
        { return (x <= 1) ? (-2.0/3.14) : 0; }
        //{ return (x <= 1) ? -1 : 0; }
public:
    int frame;
    int sumIter;

	// Init methods
    void init(cv::Mat & img, int x1, int y1, int x2, int y2,int maxiter=10);
    void setMaxNum(int maxnum){
        _maxdatanum = maxnum;
    }
    void setBCT(double val){
        _backvalid_thresh = val;
    }
    void print_info(){
        std::cout<<"Tracker Info: ---------------"<<std::endl;
        std::cout<<"max data num is :" <<_maxdatanum<<std::endl;
        std::cout<<"max iter num is :" <<_maxiter<<std::endl;
        std::cout<<"back validation threshold is :" <<_backvalid_thresh<<std::endl;
        std::cout<<"------------------------------"<<std::endl;
    }

    // Set last object position - starting position for next tracking step
    inline void setLastBBox(int x1, int y1, int x2, int y2)
	{
        lastPosition.setBBox(x1, y1, x2-x1, y2-y1, 1, 1);
    }

    inline BBox * getBBox()
    {
        BBox * bbox = new BBox();
        *bbox = lastPosition;
        return bbox;
    }

	// frame-to-frame object tracking
    BBox * track(cv::Mat & img, cv::Mat & img_old, double x1, double y1, double x2, double y2, int * iter = NULL);
    inline BBox * track(cv::Mat & img,cv::Mat & img_old,int * iter = NULL)
    {
        return track(img,img_old,lastPosition.x, lastPosition.y, lastPosition.x + lastPosition.width, lastPosition.y + lastPosition.height,iter);
    }
};

#endif // COLOTRACKER_H
