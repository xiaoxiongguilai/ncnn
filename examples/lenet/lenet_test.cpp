/* lenet 测试demo
   ncnn版本：20171225(Release) [1f5c646ee0023f6ab96a908d9e02c08e94f47885]
   高版本ncnn会导致innerproduct输出连续性不一致
*/
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "net.h"
int main()
{
    cv::Mat img = cv::imread("../test2.jpg");
    printf("img w=%d h=%d c=%d\n", img.cols, img.rows,img.channels());

    cv::resize(img, img, cv::Size(28,28));
    cv::cvtColor(img, img, CV_RGB2GRAY);
    cv::threshold(img, img, 180, 255, cv::THRESH_BINARY_INV);

    int w = img.cols;
    int h = img.rows;

    cv::imwrite("./aa.jpg",img);
    system("~/imgcat.sh ./aa.jpg");
    // init net
    ncnn::Net net;
    {
        int r0 = net.load_param("../lenet_ncnn.param");
        int r1 = net.load_model("../lenet_ncnn.bin");
        printf("r0=%d r1=%d \n",r0,r1);
    }

#if 0
    for(int i=0;i<10;i++){
        printf("i=%d data=%d \n",i,img.data[i]);
    }
#endif

    printf("22 w=%d h=%d \n", img.cols,img.rows);
    printf("##### Forward start\n");
    ncnn::Mat tmp;
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(1);

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_GRAY, w, h, 28, 28);
        //ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2GRAY, w, h, 28, 28);

        //const float mean_vals[3] = {104.f, 117.f, 123.f};
        const float norm_vals[1] = {0.00390625};
        in.substract_mean_normalize(0, norm_vals);

        ex.input("data", in);
        //ex.extract("ip2", tmp);
        ex.extract("loss", out);

        printf("##### out %d %d %d  \n",out.w,out.h,out.c);
    }
    printf("##### Forward End\n");

    std::vector<float> cls_scores;
    {
        cls_scores.resize(out.c);
        for (int j=0; j<out.c; j++){
            const float* prob = (const float*)out.data + out.cstep * j;
            cls_scores[j] = prob[0];
        }
    }

    int max_index=0;
    float max = 0;
    for (int j=0; j<out.c; j++){
        printf("##### AA %lf  \n", cls_scores[j]);

        if(cls_scores[j]>max){
            max = cls_scores[j];
            max_index = j;
        }
    }
    //net.clear();
    printf("Rec Num is %d \n",max_index);

    return 0;
}
