#include "colotracker.h"
#include <iostream>
#include <fstream>

void ColorTracker::init(cv::Mat & img, int x1, int y1, int x2, int y2,int maxiter)
{
    _maxiter = maxiter;
    img.copyTo(_img_old);
    im1 = cv::Mat( img.rows, img.cols, CV_8UC1 );
    im2 = cv::Mat( img.rows, img.cols, CV_8UC1 );
    im3 = cv::Mat( img.rows, img.cols, CV_8UC1 );

    //boundary checks
    y1 = std::max(0, y1);
    y2 = std::min(img.rows-1, y2);
    x1 = std::max(0, x1);
    x2 = std::min(img.cols-1, x2);

    preprocessImage(img);
    extractForegroundHistogram(x1, y1, x2, y2, q_hist);
    q_orig_hist = q_hist;

    extractBackgroundHistogram(x1, y1, x2, y2, b_hist);

    Histogram b_weights = b_hist;

    b_weights.transformToWeights();
    q_hist.multiplyByWeights(&b_weights);

    lastPosition.setBBox(x1, y1, x2-x1, y2-y1, 1, 1);
    defaultWidth = x2-x1;
    defaultHeight = y2-y1;
    sumIter = 0;
    frame = 0;

    double w2 = (x2-x1)/2.;
    double h2 = (y2-y1)/2.;
    double cx = x1 + w2;
    double cy = y1 + h2;
    double wh = w2+5.;
    double hh = h2+5.;

    double Sbg = 0, Sfg = 0;
    for(int i = y1; i < y2+1; i++) {
        const uchar *Mi1 = im1.ptr<uchar>(i);
        const uchar *Mi2 = im2.ptr<uchar>(i);
        const uchar *Mi3 = im3.ptr<uchar>(i);
        double tmp_y = std::pow((cy - i) / hh, 2);
        for (int j = x1; j < x1 + 1; j++) {
            double arg = std::pow((cx - j) / wh, 2) + tmp_y;
            if (arg > 1)
                continue;
            //likelihood weights
            double wqi = 1.0;
            double wbi = sqrt(b_hist.getValue(Mi1[j], Mi2[j], Mi3[j]) /
                              q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
            Sbg += (wqi < wbi) ? q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])
                               : 0.0;
            Sfg += q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);
        }
    }

    //wAvgBg = 0.5
    wAvgBg = std::max(0.1, std::min(Sbg/Sfg, 0.5));
    bound1 = 0.05;
    bound2 = 0.1;

    _orig_qb_simi = q_orig_hist.computeSimilarity(&b_hist);
    tgColorDistribution(x1, y1, x2, y2, q_hist);
    print_info();
    return;
}

cv::Point ColorTracker::histMeanShift(double x1, double y1, double x2, double y2)
{
    int maxIter = 10;

    double w2 = (x2-x1+1)/2;
    double h2 = (y2-y1+1)/2;

    double borderX = 5;
    double borderY = 5;

    double cx = x1 + w2;
    double cy = y1 + h2;

    Histogram y1hist;

    int ii;
    for (ii = 0; ii < maxIter; ++ii){

        double wh = (w2+borderX);
        double hh = (h2+borderY);
        int rowMin = std::max(0, (int)(cy-hh));
        int rowMax = std::min(im1.rows, (int)(cy+hh));
        int colMin = std::max(0, (int)(cx-wh));
        int colMax = std::min(im1.cols, (int)(cx+wh));

        extractForegroundHistogram(colMin, rowMin, colMax, rowMax, y1hist);

        double batta_q = y1hist.computeSimilarity(&q_orig_hist);
        double batta_b = y1hist.computeSimilarity(&b_hist);


        //MeanShift Vector
        double m0 = 0, m1x = 0, m1y = 0;
        for(int i = rowMin; i < rowMax; i++){
            const uchar* Mi1 = im1.ptr<uchar>(i);
            const uchar* Mi2 = im2.ptr<uchar>(i);
            const uchar* Mi3 = im3.ptr<uchar>(i);
            double tmp_y = std::pow((cy-(float)i)/hh,2);
            for(int j = colMin; j < colMax; j++){
                double arg = std::pow((cx-(float)j)/wh,2) + tmp_y;
                if (arg>1)
                    continue;

                double wqi = sqrt(q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
                double wbi = sqrt(b_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
                
                double wg = std::max(wqi/batta_q - wbi/batta_b, 0.0)*(-kernelProfile_EpanechnikovDeriv(arg));     

                m0 += wg;
                m1x += (j-cx)*wg;
                m1y += (i-cy)*wg;
            }
        }

        double xn_1 = m1x/m0 + cx;
        double yn_1 = m1y/m0 + cy;

        if (std::pow(xn_1 - cx,2) + std::pow(yn_1 - cy,2) < 1)
            break;

        if (m0==m0 && !std::isinf(m0) && m0 > 0){
            cx = xn_1;
            cy = yn_1;
        }
    }

    return  cv::Point(cx, cy);
}

cv::Point ColorTracker::histMeanShiftIsotropicScale(double x1, double y1, double x2, double y2, double * scale, int * iter)
{
    int maxIter = 10;
    double w2 = (x2-x1)/2;
    double h2 = (y2-y1)/2;
    double borderX = 5;
    double borderY = 5;
    double cx = x1 + w2;
    double cy = y1 + h2;
    double h0 = 1;
    int ii;

    for (ii = 0; ii < maxIter; ++ii){
        double wh = h0*w2+borderX;
        double hh = h0*h2+borderY;
        int rowMin = std::max(0, (int)(cy-hh));
        rowMin = std::min(rowMin,im1.rows-1);
        int rowMax = std::min(im1.rows-1, (int)(cy+hh));
        rowMax = std::max(0,rowMax);

        int colMin = std::max(0, (int)(cx-wh));
        colMin = std::min(colMin,im1.cols-1);
        int colMax = std::min(im1.cols-1, (int)(cx+wh));
        colMax = std::max(0,colMax);

        Histogram y1hist;
        if(rowMax==rowMin){
            if(rowMax == im1.rows-1)
                rowMin--;
            else
                rowMax++;
        }
        if(colMax==colMin){
            if(colMax == im1.cols-1)
                colMin--;
            else
                colMax++;
        }
        extractForegroundHistogram(colMin, rowMin, colMax, rowMax, y1hist);

        double batta_q = y1hist.computeSimilarity(&q_orig_hist);
        double batta_b = y1hist.computeSimilarity(&b_hist);

        //MeanShift Vector
        double m0 = 0, m1x = 0, m1y = 0;
        double wg_dist_sum = 0, wk_sum = 0, Sbg = 0, Sfg = 0;
        double rescale = 1;

        if(w2*h2*4 > _maxdatanum)
            rescale = sqrt(wh*hh*4/_maxdatanum);
        int sw = (int)(wh/rescale);
        int sh = (int)(hh/rescale);
        double xstep = wh/sw;
        double ystep = hh/sh;

        for(int y = 0; y < 2*sh; y++){
            int i = floor(ystep*y)+rowMin;
            if(i < rowMin){i = rowMin;}
            else if(i > rowMax){i = rowMax;}
            const uchar* Mi1 = im1.ptr<uchar>(i);
            const uchar* Mi2 = im2.ptr<uchar>(i);
            const uchar* Mi3 = im3.ptr<uchar>(i);
            double tmp_y = std::pow((cy-i)/hh,2);
            for(int x = 0; x < 2*sw; x++){
                int j = floor(xstep*x)+colMin;
                if(j < colMin){j = colMin;}
                else if(j > colMax){j = colMax;}
                double arg = std::pow((cx-j)/wh,2) + tmp_y;
                if (arg>1)
                    continue;
                double y1hist_elem =y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]);
                if(y1hist_elem<=0)
                    y1hist_elem = 1;

                //likelihood weights
                double wqi = sqrt(q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist_elem);
                double wbi = sqrt(b_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist_elem);
                double w = std::max(wqi/batta_q - wbi/batta_b, 0.0);

                //weights based on "Robust mean-shift tracking with corrected background-weighted histogram"
                // double w = sqrt(q_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));

                //orig weights
                // double w = sqrt(q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));

                double wg = w*(-kernelProfile_EpanechnikovDeriv(arg));     
                double dist = std::sqrt(std::pow((j-cx)/w2,2) + std::pow((i-cy)/h2,2));

                wg_dist_sum += wg*dist;
                wk_sum += w*(kernelProfile_Epanechnikov(arg));

                //orig weights
                // Sbg += (q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]) == 0) ? y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]) : 0;
                // Sfg += q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);

                //likelihood
                Sbg += (wqi < wbi) ? y1hist_elem : 0;
                Sfg += q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);

                //SCIA
                // Sbg += (w == 0) ? y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]) : 0;
                // Sfg += q_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);

                m0 += wg;
                m1x += (j-cx)*wg;
                m1y += (i-cy)*wg;
            }            
        }
        double xn_1 = m1x/m0 + cx;
        double yn_1 = m1y/m0 + cy;

        //Rebularization
        double reg2 = 0, reg1 = 0;
        reg1 = (wAvgBg - Sbg/Sfg);
        if (std::abs(reg1) > bound1)
            reg1 = reg1 > 0 ? bound1 : -bound1;

        reg2 = -(log(h0));
        if (std::abs(reg2) > bound2)
            reg2 = reg2 > 0 ? bound2 : -bound2;
        
        double h_tmp = (1.0 - wk_sum/m0)*h0 + (1.0/h0)*(wg_dist_sum/m0) + reg1 + reg2;
                
        if (std::pow(xn_1 - cx,2) + std::pow(yn_1 - cy,2) < 1)
            break;

        if (m0==m0 && !std::isinf(m0) && m0 > 0){
            cx = xn_1;
            cy = yn_1;
            h0 = 0.7*h0 + 0.3*h_tmp;
            if(borderX >5){
                borderX /= 3;
                borderY /= 3;
            }
        }else if (ii == 0){     
            //if in first iteration is m0 not valid => fail (maybe too fast movement) 
            //  try to enlarge the search region
            borderX = 3*borderX;
            borderY = 3*borderY;
        }
    }
    *scale = h0;
    if (iter != NULL)
        *iter = ii;
    return  cv::Point(cx, cy);
}
cv::Point ColorTracker::histMeanShiftIsotropicScale_c(const cv::Mat &img,
                                                      double x1, double y1, double x2, double y2,
                                                      double * scale, int maxIter, int * iter)
{
    double w2 = (x2-x1)/2;
    double h2 = (y2-y1)/2;

    double borderX = 5;
    double borderY = 5;

    double cx = x1 + w2;
    double cy = y1 + h2;

    double h0 = 1;
    int ii;
    double batta_q=0.9;
    double batta_b=0.8;
    for (ii = 0; ii < maxIter; ++ii){
        double wh = h0*w2+borderX;
        double hh = h0*h2+borderY;
        int rowMin = std::max(0, (int)(cy-hh));
        rowMin = std::min(rowMin,im1.rows-1);
        int rowMax = std::min(im1.rows-1, (int)(cy+hh));
        rowMax = std::max(0,rowMax);

        int colMin = std::max(0, (int)(cx-wh));
        colMin = std::min(colMin,im1.cols-1);
        int colMax = std::min(im1.cols-1, (int)(cx+wh));
        colMax = std::max(0,colMax);

        Histogram y1hist;
        if(rowMax==rowMin){
            if(rowMax == im1.rows-1)
                rowMin--;
            else
                rowMax++;
        }
        if(colMax==colMin){
            if(colMax == im1.cols-1)
                colMin--;
            else
                colMax++;
        }
        extractForegroundHistogram_c(img,colMin, rowMin, colMax, rowMax, y1hist);
        if(y1hist.dimSize==0)
            continue;

        batta_q = y1hist.computeSimilarity(&q_orig_hist);
        batta_b = y1hist.computeSimilarity(&b_hist);

        //MeanShift Vector
        double m0 = 0, m1x = 0, m1y = 0;
        double wg_dist_sum = 0, wk_sum = 0, Sbg = 0, Sfg = 0;
        double rescale = 1;

        if(w2*h2*4 > _maxdatanum)
            rescale = sqrt(wh*hh*4/_maxdatanum);
        int sw = (int)(wh/rescale);
        int sh = (int)(hh/rescale);
        double xstep = wh/sw;
        double ystep = hh/sh;
        double segline1 = 2*(double)sh*segment_u;
        double segline2 = 2*(double)sh*segment_d;

        //spirse the center

        for(int y = 0; y < 2*sh; y++){
            if(y>segline1 && y<=segline2 && ((y-(int)segline1)%_bodyratio!=0))
                continue;
            int i = floor(ystep*y)+rowMin;
            if(i < rowMin){
                i = rowMin;
            }else if(i > rowMax){
                i = rowMax;
            }
            const cv::Vec3b* Mi = img.ptr<cv::Vec3b>(i);
            double tmp_y = std::pow((cy-i)/hh,2);
            for(int x = 0; x < 2*sw; x++){
                int j = floor(xstep*x)+colMin;
                if(j < colMin){
                    j = colMin;
                }else if(j > colMax){
                    j = colMax;
                }
                double arg = std::pow((cx-j)/wh,2) + tmp_y;
                if (arg>1)
                    continue;
                double y1hist_elem =y1hist.getValue(Mi[j][0], Mi[j][1], Mi[j][2]);
                if(y1hist_elem<=0)
                    y1hist_elem = 1;
                double q_hist_elem = q_orig_hist.getValue(Mi[j][0], Mi[j][1], Mi[j][2]);
                double b_hist_elem = b_hist.getValue(Mi[j][0], Mi[j][1], Mi[j][2]);

                //likelihood weights
                double wqi = sqrt(q_hist_elem/y1hist_elem);
                double wbi = sqrt(b_hist_elem/y1hist_elem);
                double w = std::max(wqi/batta_q - wbi/batta_b, 0.0);

                //weights based on "Robust mean-shift tracking with corrected background-weighted histogram"
                // double w = sqrt(q_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));

                //orig weights
                // double w = sqrt(q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));

                double wg = w*(-kernelProfile_EpanechnikovDeriv(arg));
                double dist = std::sqrt(std::pow((j-cx)/w2,2) + std::pow((i-cy)/h2,2));

                wg_dist_sum += wg*dist;
                wk_sum += w*(kernelProfile_Epanechnikov(arg));

                //orig weights
                // Sbg += (q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]) == 0) ? y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]) : 0;
                // Sfg += q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);

                //likelihood
                Sbg += (wqi < wbi) ? y1hist_elem : 0;
                Sfg += q_hist_elem;

                //SCIA
                // Sbg += (w == 0) ? y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]) : 0;
                // Sfg += q_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);

                m0 += wg;
                m1x += (j-cx)*wg;
                m1y += (i-cy)*wg;
            }
        }
        double xn_1 = m1x/m0 + cx;
        double yn_1 = m1y/m0 + cy;

        //Rebularization
        double reg2 = 0, reg1 = 0;
        reg1 = (wAvgBg - Sbg/Sfg);
        if (std::abs(reg1) > bound1)
            reg1 = reg1 > 0 ? bound1 : -bound1;

        reg2 = -(log(h0));
        if (std::abs(reg2) > bound2)
            reg2 = reg2 > 0 ? bound2 : -bound2;

        double h_tmp = (1.0 - wk_sum/m0)*h0 + (1.0/h0)*(wg_dist_sum/m0) + reg1 + reg2;

        if (std::pow(xn_1 - cx,2) + std::pow(yn_1 - cy,2) < 1)
            break;

        if (m0==m0 && !std::isinf(m0) && m0 > 0){
            cx = xn_1;
            cy = yn_1;
            h0 = 0.7*h0 + 0.3*h_tmp;
            if(borderX >5){
                borderX /= 3;
                borderY /= 3;
            }
        }else if (ii == 0){
            //if in first iteration is m0 not valid => fail (maybe too fast movement)
            //  try to enlarge the search region
            borderX = 3*borderX;
            borderY = 3*borderY;
        }
    }
    *scale = h0;
    if (iter != NULL)
        *iter = ii;
    if(_record_batta){
        _cur_batta_b = batta_b;
        _cur_batta_q = batta_q;
        _record_batta = false;
    }

    return  cv::Point(cx, cy);
}
cv::Point ColorTracker::histMeanShiftAnisotropicScale(double x1, double y1, double x2, double y2, double * width, double * height)
{
    int maxIter = 5;

    double w2 = (x2-x1)/2.;
    double h2 = (y2-y1)/2.;

    double borderX = 5.;
    double borderY = 5.;

    double cx = x1 + w2;
    double cy = y1 + h2;
    double h0_1 = 1.;
    double h0_2 = 1.;

    double wh;
    double hh;

    Histogram y1hist;

    int ii;
    for (ii = 0; ii < maxIter; ++ii){

        wh = w2*h0_1+borderX;
        hh = h2*h0_2+borderY;
        int rowMin = std::max(0, (int)(cy-hh));
        int rowMax = std::min(im1.rows-1, (int)(cy+hh));
        int colMin = std::max(0, (int)(cx-wh));
        int colMax = std::min(im1.cols-1, (int)(cx+wh));

        extractForegroundHistogram(colMin, rowMin, colMax, rowMax, y1hist);

        double batta_q = y1hist.computeSimilarity(&q_orig_hist);
        double batta_b = y1hist.computeSimilarity(&b_hist);

        //MeanShift Vector
        double m0 = 0, m1x = 0, m1y = 0;
        double Swigdist_1 = 0, Swigdist_2 = 0;
        double wk_sum = 0, Sbg = 0, Sfg = 0;

        for(int i = rowMin; i < rowMax; i++){
            const uchar* Mi1 = im1.ptr<uchar>(i);
            const uchar* Mi2 = im2.ptr<uchar>(i);
            const uchar* Mi3 = im3.ptr<uchar>(i);
            double tmp_y = std::pow((cy-(float)i)/hh,2);

            for(int j = colMin; j < colMax; j++){
                double arg = std::pow((cx-(float)j)/wh,2) + tmp_y;
                if (arg>1)
                    continue;

                //likelihood weights
                double wqi = sqrt(q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
                double wbi = sqrt(b_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
                double w = std::max(wqi/batta_q - wbi/batta_b, 0.0);

                double wg = w*(-kernelProfile_EpanechnikovDeriv(arg));

                wk_sum += (w * kernelProfile_Epanechnikov(arg));
                Swigdist_1 += wg*std::pow((cx-(float)j),2);
                Swigdist_2 += wg*std::pow((cy-(float)i),2);

                //likelihood
                Sbg += (wqi < wbi) ? y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]) : 0;
                Sfg += q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);

                m0 += wg;
                m1x += (j-cx)*wg;
                m1y += (i-cy)*wg;
            }
        }

        double a2 = std::pow(w2,2);
        double b2 = std::pow(h2,2);

        float mx = (h0_2/h0_1)*(m1x/m0);
        float my = (h0_1/h0_2)*(m1y/m0);

        double reg1 = (wAvgBg - Sbg/Sfg);
        if (std::abs(reg1) > bound1)
            reg1 = reg1 > 0 ? bound1 : -bound1;

        double reg2_1 = -(log(h0_1));
        if (std::abs(reg2_1) > bound2)
            reg2_1 = reg2_1 > 0 ? bound2 : -bound2;
        double reg2_2 = -(log(h0_2));
        if (std::abs(reg2_2) > bound2)
            reg2_2 = reg2_2 > 0 ? bound2 : -bound2;

        double h_1 = h0_1 - (h0_2/2)*(wk_sum / m0) + (h0_2 / (h0_1 * h0_1 * a2)) * (Swigdist_1 / m0) + reg1 + reg2_1;
        double h_2 = h0_2 - (h0_1/2)*(wk_sum / m0) + (h0_1 / (h0_2 * h0_2 * b2)) * (Swigdist_2 / m0) + reg1 + reg2_2;

        if (std::pow(mx,2) + std::pow(my,2) < 1)
            break;

        if (m0==m0 && !std::isinf(m0) && m0 > 0){
            cx = cx + mx;
            cy = cy + my;
            h0_1 = 0.7*h0_1 + 0.3*h_1;
            h0_2 = 0.7*h0_2 + 0.3*h_2;
            if (borderX > 5){
                borderX /= 3;
                borderY /= 3;
            }
        }else if (ii == 0){
            //if in first iteration is m0 not valid => fail (maybe too fast movement)
            //  try to enlarge the search region
            borderX = 3*borderX;
            borderY = 3*borderY;
        }
    }

    *width = 2*w2*h0_1;
    *height = 2*h2*h0_2;

    return  cv::Point(cx, cy);
}



BBox * ColorTracker::track(cv::Mat & img,cv::Mat & img_old, double x1, double y1, double x2, double y2, int * retiter)
{
    double width = x2-x1;
    double height = y2-y1;

    //MS with scale estimation
    double scale = 1;
    int iter = 0;

    BBox *retBB = new BBox();
    //loss checking
    int checkstep = (_lose_target)?20:10;
    if(frame%checkstep==0){
        if(!_lose_target){
            double simidist = calcProbDist(_cur_batta_q,_cur_batta_b);
            if(simidist<0)
                 _lose_count++;
            else if(_lose_count>0)
                _lose_count--;
        }
        if(_lose_count>=5){
            //std::cout<<"global search"<<std::endl;
            _lose_target = true;
            if(!this->reSearchTarget(img,retBB))
                *retBB = lastPosition;
            else{
                lastPosition = *retBB;
                _lose_count = 0;
                _lose_target = false;
            }
            frame++;
            if(retiter!=NULL){ *retiter = 0; }
            return retBB;
        }
    }
    if(_lose_target){
        *retBB = lastPosition;
        frame++;
        if(retiter!=NULL){ *retiter = 0; }
        return retBB;
    }
    _record_batta = true;
    cv::Point modeCenter = histMeanShiftIsotropicScale_c(img,x1, y1, x2, y2, &scale, _maxiter, &iter);
    width = 0.6*width + 0.4*width*scale;
    height = 0.6*height + 0.4*height*scale;
    
    //Forward-Backward validation
    if (std::abs(std::log(scale)) > _backvalid_thresh){
        double scaleB = scale;
        histMeanShiftIsotropicScale_c(img_old,modeCenter.x - width/2, modeCenter.y - height/2, modeCenter.x + width/2, modeCenter.y + height/2, &scaleB,_maxiter);
        if (std::abs(std::log(scale*scaleB)) > 0.1){
            double alfa = 0.1*(defaultWidth/(float)(x2-x1));
            width = (0.6 - alfa)*(x2-x1) + 0.4*(x2-x1)*scale + alfa*defaultWidth;
            height = (0.6 - alfa)*(y2-y1) + 0.4*(y2-y1)*scale + alfa*defaultHeight;
        }
    }
    retBB->setBBox(modeCenter.x - width/2, modeCenter.y - height/2, width, height, 1, 1);
    lastPosition.setBBox(modeCenter.x - width/2, modeCenter.y - height/2, width, height, 1, 1);
    frame++;
    if(retiter!=NULL)
        *retiter = iter;

    return retBB;
}

void ColorTracker::preprocessImage(cv::Mat &img)
{
    cv::Mat ra[3] = {im1, im2, im3};
    cv::split(img, ra);
}

void ColorTracker::extractBackgroundHistogram(int x1, int y1, int x2, int y2, Histogram & hist)
{
    int offsetX = (x2-x1)/2;
    int offsetY = (y2-y1)/2;

    int rowMin = std::max(0, (int)(y1-offsetY));
    int rowMax = std::min(im1.rows, (int)(y2+offsetY+1));
    int colMin = std::max(0, (int)(x1-offsetX));
    int colMax = std::min(im1.cols, (int)(x2+offsetX+1));

    int numData = (rowMax-rowMin)*(colMax-colMin) - (y2-y1)*(x2-x1);

    if (numData < 1)
        numData = (rowMax-rowMin)*(colMax-colMin)/2 + 1;

    std::vector<int> d1, d2, d3;
    std::vector<double> weights;
    d1.reserve(numData);
    d2.reserve(numData);
    d3.reserve(numData);

    for (int y = rowMin; y < rowMax; ++y){
        const uchar * M1 = im1.ptr<uchar>(y);
        const uchar * M2 = im2.ptr<uchar>(y);
        const uchar * M3 = im3.ptr<uchar>(y);
        for (int x = colMin; x < colMax; ++x){
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2)
                continue;
            d1.push_back(M1[x]);
            d2.push_back(M2[x]);
            d3.push_back(M3[x]);
        }
    }
    hist.clear();    
    hist.insertValues(d1, d2, d3, weights);
}


void ColorTracker::extractForegroundHistogram(int x1, int y1, int x2, int y2, Histogram & hist)
{
    hist.clear();
    std::vector<int> data1;
    std::vector<int> data2;
    std::vector<int> data3;
    std::vector<double> weights;



    double w2 = (double)(x2-x1)/2;
    double h2 = (double)(y2-y1)/2;

    double cx = x1 + w2;
    double cy = y1 + h2;

    double rescale = 1;

    if(w2*h2*4 > _maxdatanum)
        rescale = sqrt(w2*h2*4/_maxdatanum);
    int sw = ((int)(x2-x1))/rescale;
    int sh = ((int)(y2-y1))/rescale;
    int numData = sw*sh;
    double xstep = 2*w2/sw;
    double ystep = 2*h2/sh;
    if (numData <= 0){
        return;
    }

    data1.reserve(numData);
    data2.reserve(numData);
    data3.reserve(numData);
    weights.reserve(numData);

    double wh_i = 1.0/(w2*1.4142+1);  //sqrt(2)
    double hh_i = 1.0/(h2*1.4142+1);

    for (int y = 0; y < sh; ++y){
        int yidx = floor(y*ystep)+y1;
        const uchar * M1 = im1.ptr<uchar>(yidx);
        const uchar * M2 = im2.ptr<uchar>(yidx);
        const uchar * M3 = im3.ptr<uchar>(yidx);
        double tmp_y = std::pow((cy-yidx)*hh_i,2);
        for (int x = 0; x < sw; ++x){
            int xidx = floor(x*xstep)+x1;
            data1.push_back(M1[xidx]);
            data2.push_back(M2[xidx]);
            data3.push_back(M3[xidx]);
            weights.push_back(kernelProfile_Epanechnikov(std::pow((cx-xidx)*wh_i,2) + tmp_y));
        }
    }
    hist.clear();
    hist.insertValues(data1, data2, data3, weights);
}

void ColorTracker::tgColorDistribution(int x1, int y1, int x2, int y2, Histogram hist)
{
    double wh = x2-x1;
    double hh = y2-y1;
    double cx = x1 + wh/2;
    double cy = y1 + hh/2;
    double rescale = 1;
    int rowMin = std::max(0, (int)(cy-hh));
    rowMin = std::min(rowMin,im1.rows-1);
    int rowMax = std::min(im1.rows-1, (int)(cy+hh));
    rowMax = std::max(0,rowMax);

    int colMin = std::max(0, (int)(cx-wh));
    colMin = std::min(colMin,im1.cols-1);
    int colMax = std::min(im1.cols-1, (int)(cx+wh));
    colMax = std::max(0,colMax);

    if(wh*hh > _maxdatanum)
        rescale = sqrt(wh*hh*4/_maxdatanum);
    int sw = (int)(wh/rescale);
    int sh = (int)(hh/rescale);
    double xstep = wh/sw;
    double ystep = hh/sh;
    double w1=0,w2=0,w3=0;
    double segline1 = 2*(double)sh*segment_u;
    double segline2 = 2*(double)sh*segment_d;


    for(int y = 0; y < 2*sh; y++){
        int i = floor(ystep*y)+rowMin;
        if(i < rowMin){i = rowMin;}
        else if(i > rowMax){i = rowMax;}
        const uchar* Mi1 = im1.ptr<uchar>(i);
        const uchar* Mi2 = im2.ptr<uchar>(i);
        const uchar* Mi3 = im3.ptr<uchar>(i);
        double tmp_y = std::pow((cy-i)/hh,2);
        for(int x = 0; x < 2*sw; x++){
            int j = floor(xstep*x)+colMin;
            if(j < colMin){j = colMin;}
            else if(j > colMax){j = colMax;}
            double arg = std::pow((cx-j)/wh,2) + tmp_y;
            if (arg>1)
                continue;
            double y1hist_elem =q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);
            if(y1hist_elem<=0)
                y1hist_elem = 1;
            double wqi = sqrt(q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist_elem);
            double wbi = sqrt(b_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist_elem);
            double w = std::max(wqi - wbi/_orig_qb_simi, 0.0);
            double wg = w*(-kernelProfile_EpanechnikovDeriv(arg));

            if(y<=segline1)
                w1+=wg;
            else if(y>segline1 && y<=segline2 )
                w2+=wg;
            else
                w3+=wg;
        }
    }
    w1/=segline1;
    w2/=(segline2-segline1);
    w3/=(2*sh-segline2);
    _bodyratio = std::floor(w2/(w1+w3)+0.5);
    _bodyratio = std::min(2,_bodyratio);
    _bodyratio = std::max(1,_bodyratio);
    std::cout<<"Body Weight "<<_bodyratio<<std::endl;
    return;
}
void ColorTracker::extractForegroundHistogram_c(const cv::Mat &img,int x1, int y1, int x2, int y2, Histogram & hist)
{
    hist.clear();
    std::vector<int> data1;
    std::vector<int> data2;
    std::vector<int> data3;
    std::vector<double> weights;

    double w2 = (double)(x2-x1)/2;
    double h2 = (double)(y2-y1)/2;

    double cx = x1 + w2;
    double cy = y1 + h2;

    double rescale = 1;

    if(w2*h2*4 > _maxdatanum)
        rescale = sqrt(w2*h2*4/_maxdatanum);
    int sw = ((int)(x2-x1))/rescale;
    int sh = ((int)(y2-y1))/rescale;
    int numData = sw*sh;
    double xstep = 2*w2/sw;
    double ystep = 2*h2/sh;
    if (numData <= 0){
        return;
    }

    data1.reserve(numData);
    data2.reserve(numData);
    data3.reserve(numData);
    weights.reserve(numData);

    double wh_i = 1.0/(w2*1.4142+1);  //sqrt(2)
    double hh_i = 1.0/(h2*1.4142+1);

    for (int y = 0; y < sh; ++y){
        int yidx = floor(y*ystep)+y1;
        const cv::Vec3b* Mi = img.ptr<cv::Vec3b>(yidx);
        double tmp_y = std::pow((cy-yidx)*hh_i,2);
        for (int x = 0; x < sw; ++x){
            int xidx = floor(x*xstep)+x1;
            data1.push_back(Mi[xidx][0]);
            data2.push_back(Mi[xidx][1]);
            data3.push_back(Mi[xidx][2]);
            weights.push_back(kernelProfile_Epanechnikov(std::pow((cx-xidx)*wh_i,2) + tmp_y));
        }
    }
    hist.clear();
    hist.insertValues(data1, data2, data3, weights);
}
void ColorTracker::extractForegroundHistogram(const cv::Mat &target, Histogram & hist)
{
    cv::Mat imt1,imt2,imt3;
    imt1 = cv::Mat( defaultHeight,defaultWidth, CV_8UC1 );
    imt2 = cv::Mat( defaultHeight,defaultWidth, CV_8UC1 );
    imt3 = cv::Mat( defaultHeight,defaultWidth, CV_8UC1 );
    cv::Mat ra[3] = {imt1, imt2, imt3};
    cv::split(target, ra);
    hist.clear();
    std::vector<int> data1;
    std::vector<int> data2;
    std::vector<int> data3;
    std::vector<double> weights;

    int numData = defaultHeight*defaultWidth;

    if (numData <= 0){
        return;
    }

    data1.reserve(numData);
    data2.reserve(numData);
    data3.reserve(numData);
    weights.reserve(numData);

    double w2 = defaultWidth/2;
    double h2 = defaultHeight/2;


    double cx = defaultWidth/2;
    double cy = defaultHeight/2;

    double wh_i = 1.0/(w2*1.4142+1);  //sqrt(2)
    double hh_i = 1.0/(h2*1.4142+1);

    for (int y = 0; y < defaultHeight; ++y){
        const uchar * M1 = imt1.ptr<uchar>(y);
        const uchar * M2 = imt2.ptr<uchar>(y);
        const uchar * M3 = imt3.ptr<uchar>(y);
        double tmp_y = std::pow((cy-y)*hh_i,2);
        for (int x = 0; x < defaultWidth; ++x){
            data1.push_back(M1[x]);
            data2.push_back(M2[x]);
            data3.push_back(M3[x]);
            weights.push_back(kernelProfile_Epanechnikov(std::pow((cx-x)*wh_i,2) + tmp_y));
        }
    }


    hist.clear();
    hist.insertValues(data1, data2, data3, weights);
}
void ColorTracker::extractBackgroundHistogram(const cv::Mat &target,int x1, int y1, int x2, int y2, Histogram &hist)
{
    cv::Mat imt1,imt2,imt3;
    double theight = target.rows;
    double twidth = target.cols;
    imt1 = cv::Mat( theight,twidth, CV_8UC1 );
    imt2 = cv::Mat( theight,twidth, CV_8UC1 );
    imt3 = cv::Mat( theight,twidth, CV_8UC1 );
    cv::Mat ra[3] = {imt1, imt2, imt3};
    cv::split(target, ra);
    int offsetX = (x2-x1)/2;
    int offsetY = (y2-y1)/2;

    int rowMin = std::max(0, (int)(y1-offsetY));
    int rowMax = std::min(imt1.rows, (int)(y2+offsetY+1));
    int colMin = std::max(0, (int)(x1-offsetX));
    int colMax = std::min(imt1.cols, (int)(x2+offsetX+1));

    int numData = (rowMax-rowMin)*(colMax-colMin) - (y2-y1)*(x2-x1);

    if (numData < 1)
        numData = (rowMax-rowMin)*(colMax-colMin)/2 + 1;

    std::vector<int> d1, d2, d3;
    std::vector<double> weights;
    d1.reserve(numData);
    d2.reserve(numData);
    d3.reserve(numData);

    for (int y = rowMin; y < rowMax; ++y){
        const uchar * M1 = imt1.ptr<uchar>(y);
        const uchar * M2 = imt2.ptr<uchar>(y);
        const uchar * M3 = imt3.ptr<uchar>(y);
        for (int x = colMin; x < colMax; ++x){
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2)
                continue;
            d1.push_back(M1[x]);
            d2.push_back(M2[x]);
            d3.push_back(M3[x]);
        }
    }
    hist.clear();
    hist.insertValues(d1, d2, d3, weights);
}

bool ColorTracker::reSearchTarget(cv::Mat& img,BBox* box)
{
    int cnum = std::ceil(img.cols/(1.5*defaultWidth));
    int rnum = std::ceil(img.rows/(2*defaultHeight));
    int datanum = rnum*cnum;
    int sx,sy;
    int dist[datanum];
    double h[datanum];
    cv::Point cvg_point[datanum];
    int stepx = std::floor(img.cols/cnum);
    int stepy = std::floor(img.rows/rnum);
    double scale=1;
    for(int j=0;j<rnum;j++){
        sy =0.5*stepy+j*stepy;
        for(int i=0;i<cnum;i++){
            sx=0.5*stepx+i*stepx;
            _record_batta = true;
            cvg_point[j*cnum+i] = histMeanShiftIsotropicScale_c(img,sx-0.5*defaultWidth,
                                                      sy-0.5*defaultHeight,
                                                      sx+0.5*defaultWidth,
                                                      sy+0.5*defaultHeight, &scale,2);
            dist[j*cnum+i]=calcProbDist(_cur_batta_q,_cur_batta_b);
            h[j*cnum+i] = scale;
        }
    }
    bool foundtarget = false;
    double min_dist = 1;
    int min_idx = -1;
    for(int i=0;i<datanum;i++){
        if(dist[i]<0)
            continue;
        if(dist[i]<min_dist){
            min_idx = i;
            min_dist = dist[i];
            foundtarget = true;
        }
    }
    if(foundtarget){
       box->setBBox(cvg_point[min_idx].x - defaultWidth*h[min_idx]/2,
                          cvg_point[min_idx].y - defaultHeight*h[min_idx]/2,
                          defaultWidth*h[min_idx], defaultHeight*h[min_idx], 1, 1);
       return true;
    }
    return false;
}

double ColorTracker::calcProbDist(double q, double b)
{
    if(!_lose_target && q<0.3)
        return -1;
    else if(_lose_target && q<0.6)
        return -1;
    if((b/q)>1.05)
        return -1;
    double d1 = ((q-0.9)+(b-_orig_qb_simi))/1.414;
    double d2 = ((-(q-0.9)+(b-_orig_qb_simi)))/1.414;
    double dist = std::pow(d1*0.1/1.414,2)+std::pow(0.5*0.7*d2/(1-_orig_qb_simi),2);
//    if(dist>4)
//        return -1;
    return sqrt(dist);
}
