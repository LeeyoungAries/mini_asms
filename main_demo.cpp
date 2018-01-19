#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "highgui.h"
#include "colotracker.h"
#include "region.h"
#include <string>
#include <time.h>
#include <stdio.h>
using namespace cv;

cv::Point g_topLeft(0,0);
cv::Point g_botRight(0,0);
cv::Point g_botRight_tmp(0,0);
bool plot = false;
bool g_trackerInitialized = false;
ColorTracker * g_tracker = NULL;
cv::VideoCapture webcam;
int tracker_maxiter=10;
int tracker_maxnum = 3000;
double tracker_bvt = 0.05;

static void onMouse( int event, int x, int y, int, void* param)
{
    if( event == cv::EVENT_LBUTTONDOWN && !g_trackerInitialized){
        g_topLeft = Point(x,y);
        plot = true;
    }else if (event == cv::EVENT_LBUTTONUP && !g_trackerInitialized){
        g_botRight = Point(x,y);
        plot = false;
        if (g_tracker != NULL)
            delete g_tracker;

        g_tracker = new ColorTracker();
        cv::Mat imgt;
        webcam>>imgt;
        g_tracker->setMaxNum(tracker_maxnum);
        g_tracker->setBCT(tracker_bvt);
        g_tracker->init(imgt, std::min(g_topLeft.x,g_botRight.x), std::min(g_topLeft.y,g_botRight.y), std::max(g_topLeft.x,g_botRight.x), std::max(g_topLeft.y,g_botRight.y),tracker_maxiter);
        g_trackerInitialized = true;
    }else if (event == cv::EVENT_MOUSEMOVE && !g_trackerInitialized){
        //plot bbox
        g_botRight_tmp = Point(x,y);
        // if (plot){
        //     cv::rectangle(img, g_topLeft, current, cv::Scalar(0,255,0), 2);
        //     imshow("output", img);
        // }
    }
}


int main(int argc, char **argv) 
{
    bool rtmode = true;
    auto show_prompt = [](){
        std::cout<<"run './mini_asms' or './mini_asms [maxiter] [maxdata] [backvalid_threshold]' for camera mode. "<<std::endl;
        std::cout<<"run './mini_asms [video filename]' or './mini_asms [video filename] [maxiter] [maxdata] [backvalid_threshold]' for video mode. "<<std::endl;
    };
    std::string videofile = "";
    if(argc == 4){
        tracker_maxiter = atoi(argv[1]);
        tracker_maxnum = atoi(argv[2]);
        tracker_bvt = atof(argv[3]);
    }else if(argc == 2){
        videofile = argv[1];
        rtmode = false;
    }else if(argc == 5){
        videofile = argv[1];
        tracker_maxiter = atoi(argv[2]);
        tracker_maxnum = atoi(argv[3]);
        tracker_bvt = atof(argv[4]);
        rtmode = false;
    }else if(argc != 1){
        show_prompt();
        return 0;
    }
    BBox * bb = NULL;
    cv::Mat img,img_old;
    int captureDevice = 0;

    if(rtmode)
         webcam= cv::VideoCapture(captureDevice);
    else
         webcam = cv::VideoCapture(videofile);

    webcam.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    webcam.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    if (!webcam.isOpened()){
        webcam.release();
        if(rtmode){std::cerr << "Error during opening capture device!" << std::endl;}
        else{std::cerr << "Error during opening video!" << std::endl;}
        show_prompt();
        return 1;
    }
    cv::namedWindow( "output", 0 );
    cv::setMouseCallback( "output", onMouse);
    webcam >> img;
    int frame_count = 0;
    double frame_sum = 0;
    for(;;){
        if(rtmode){
            img.copyTo(img_old);
            webcam>>img;
        }
        int c = waitKey(10);
        if( (c & 255) == 27 ) {
            std::cout << "Exiting ..." << std::endl;
            break;
        }
        //some control
        switch( (char)c ){
        case 'i':
            g_trackerInitialized = false;
            g_topLeft = cv::Point(0,0);
            g_botRight_tmp = cv::Point(0,0);
            break;
        default:;

        }

        if (g_trackerInitialized && g_tracker != NULL){
            if(!rtmode){
                img.copyTo(img_old);
                webcam>>img;
            }
            timespec start, end, diff;
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
            bb = g_tracker->track(img,img_old);/////////
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
            diff.tv_sec = ( end.tv_sec - start.tv_sec );
            diff.tv_nsec = ( end.tv_nsec - start.tv_nsec );
            if (diff.tv_nsec < 0) {
                diff.tv_sec--;
                diff.tv_nsec += 1000000000;
            }
            int usec = diff.tv_nsec + diff.tv_sec * 1000000000;
            double resultTime = (double)usec / 1000000000;
            frame_sum += resultTime;
            if(frame_count%10 == 0){
                printf("%f\n",frame_sum*1000/10);
                frame_sum = 0;
            }
            frame_count++;
        }

        if (!g_trackerInitialized && plot && g_botRight_tmp.x > 0){
            cv::rectangle(img, g_topLeft, g_botRight_tmp, cv::Scalar(255,0,0), 3);
        }
        if (bb != NULL){
            cv::rectangle(img, Point2i(bb->x, bb->y), Point2i(bb->x + bb->width, bb->y + bb->height), Scalar(255, 0, 0), 3);
            delete bb; bb = NULL;
        }
        cv::imshow("output", img);
        waitKey(10);
    }

    if (g_tracker != NULL)
        delete g_tracker;
    return 0;
}
