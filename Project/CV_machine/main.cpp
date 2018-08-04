


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include"zbar.h"
#include<chrono>
//#include"zbar/Symbol.h"
//#include"zbar/Exception.h"
//#include"zbar/Image.h"
//#include"zbar/ImageScanner.h"
//#include"zbar/Decoder.h"
//#include"zbar/zbar_scan_image.h"
//#include"ImageScanner.h"

//#include <QDebug>

using namespace cv;
using namespace std;
using namespace zbar;


Mat src; Mat src_gray;


RNG rng(12345);
//Scalar colorful = CV_RGB(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));

//获取轮廓的中心点
Point Center_cal(vector<vector<Point> > contours,int i)
{
      int centerx=0,centery=0,n=contours[i].size();
      //在提取的小正方形的边界上每隔周长个像素提取一个点的坐标，
      //求所提取四个点的平均坐标（即为小正方形的大致中心）
      centerx = (contours[i][n/4].x + contours[i][n*2/4].x + contours[i][3*n/4].x + contours[i][n-1].x)/4;
      centery = (contours[i][n/4].y + contours[i][n*2/4].y + contours[i][3*n/4].y + contours[i][n-1].y)/4;
      Point point1=Point(centerx,centery);
      return point1;
}


int main( int argc, char** argv[] )
{
     chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    src = imread( "core.png", 1 );
    Mat src_all=src.clone();


    //彩色图转灰度图
    cvtColor( src, src_gray, CV_BGR2GRAY );
    //对图像进行平滑处理
    blur( src_gray, src_gray, Size(3,3) );
    //使灰度图象直方图均衡化
    equalizeHist( src_gray, src_gray );
//    namedWindow("src_gray");
//    imshow("src_gray",src_gray);


    Scalar color = Scalar(1,1,255 );
    Mat threshold_output;
    vector<vector<Point> > contours,contours2;
    vector<Vec4i> hierarchy;
    Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
    Mat drawing2 = Mat::zeros( src.size(), CV_8UC3 );
    Mat drawingAllContours = Mat::zeros( src.size(), CV_8UC3 );

    //指定112阀值进行二值化
    threshold( src_gray, threshold_output, 112, 255, THRESH_BINARY );

 //   namedWindow("Threshold_output");
  //  imshow("Threshold_output",threshold_output);


    /*查找轮廓
     *  参数说明
        输入图像image必须为一个2值单通道图像
        contours参数为检测的轮廓数组，每一个轮廓用一个point类型的vector表示
        hiararchy参数和轮廓个数相同，每个轮廓contours[ i ]对应4个hierarchy元素hierarchy[ i ][ 0 ] ~hierarchy[ i ][ 3 ]，
            分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，该值设置为负数。
        mode表示轮廓的检索模式
            CV_RETR_EXTERNAL 表示只检测外轮廓
            CV_RETR_LIST 检测的轮廓不建立等级关系
            CV_RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
            CV_RETR_TREE 建立一个等级树结构的轮廓。具体参考contours.c这个demo
        method为轮廓的近似办法
            CV_CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
            CV_CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
            CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS 使用teh-Chinl chain 近似算法
        offset表示代表轮廓点的偏移量，可以设置为任意值。对ROI图像中找出的轮廓，并要在整个图像中进行分析时，这个参数还是很有用的。
     */
    findContours( threshold_output, contours, hierarchy,  CV_RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0) );

    int c=0,ic=0,k=0,area=0;

    //通过黑色定位角作为父轮廓，有两个子轮廓的特点，筛选出三个定位角
    int parentIdx=-1;
    for( int i = 0; i< contours.size(); i++ )
    {
        //画出所以轮廓图
        drawContours( drawingAllContours, contours, parentIdx,  CV_RGB(255,255,255) , 1, 8);
        if (hierarchy[i][2] != -1 && ic==0)
        {
            parentIdx = i;
            ic++;
        }
        else if (hierarchy[i][2] != -1)
        {
            ic++;
        }
        else if(hierarchy[i][2] == -1)
        {
            ic = 0;
            parentIdx = -1;
        }

        //有两个子轮廓
        if ( ic >= 2)
        {
            //保存找到的三个黑色定位角
            contours2.push_back(contours[parentIdx]);
            //画出三个黑色定位角的轮廓
            drawContours( drawing, contours, parentIdx,  CV_RGB(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)) , 1, 8);
            ic = 0;
            parentIdx = -1;
        }
    }

    //填充的方式画出三个黑色定位角的轮廓
    for(int i=0; i<contours2.size(); i++)
        drawContours( drawing2, contours2, i,  CV_RGB(rng.uniform(100,255),rng.uniform(100,255),rng.uniform(100,255)) , -1, 4, hierarchy[k][2], 0, Point() );

    //获取三个定位角的中心坐标
    Point point[3];
    for(int i=0; i<contours2.size(); i++)
    {
        point[i] = Center_cal( contours2, i );
    }

    //计算轮廓的面积，计算定位角的面积，从而计算出边长
    area = contourArea(contours2[1]);
    int area_side = cvRound (sqrt (double(area)));
    for(int i=0; i<contours2.size(); i++)
    {
        //画出三个定位角的中心连线
        line(drawing2,point[i%contours2.size()],point[(i+1)%contours2.size()],color,area_side/2,8);
    }

//    namedWindow("DrawingAllContours");
//    imshow( "DrawingAllContours", drawingAllContours );

//    namedWindow("Drawing2");
//    imshow( "Drawing2", drawing2 );

//    namedWindow("Drawing");
//    imshow( "Drawing", drawing );


    //接下来要框出这整个二维码
    Mat gray_all,threshold_output_all;
    vector<vector<Point> > contours_all;
    vector<Vec4i> hierarchy_all;
    cvtColor( drawing2, gray_all, CV_BGR2GRAY );


    threshold( gray_all, threshold_output_all, 45, 255, THRESH_BINARY );
    findContours( threshold_output_all, contours_all, hierarchy_all,  RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0) );//RETR_EXTERNAL表示只寻找最外层轮廓


    Point2f fourPoint2f[4];
    //求最小包围矩形
    RotatedRect rectPoint = minAreaRect(contours_all[0]);

    //将rectPoint变量中存储的坐标值放到 fourPoint的数组中
    rectPoint.points(fourPoint2f);


    for (int i = 0; i < 4; i++)
    {
        line(src_all, fourPoint2f[i%4], fourPoint2f[(i + 1)%4]
            , Scalar(20,21,237), 3);
    }

//    namedWindow("Src_all");
//    imshow( "Src_all", src_all );

    //框出二维码后，就可以提取出二维码，然后使用解码库zxing，解出码的信息。
    //或者研究二维码的排布规则，自己写解码部分
     //对截取后的区域进行解码
        //Mat imageSource = cv::Mat(*src);
        Mat imageSource(src);
       // cvResetImageROI(src);//源图像用完后，清空ROI
        cvtColor( imageSource, imageSource, CV_BGR2GRAY );  //zbar需要输入灰度图像才能很好的识别

        //Zbar二维码识别
        ImageScanner scanner;
        scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
        int width1 = imageSource.cols;
        int height1 = imageSource.rows;
        uchar *raw = (uchar *)imageSource.data;

        Image imageZbar(width1, height1, "Y800", raw, width1 * height1);
        scanner.scan(imageZbar); //扫描条码
        Image::SymbolIterator symbol = imageZbar.symbol_begin();
        if(imageZbar.symbol_begin()==imageZbar.symbol_end())
        {
            cout<<"查询条码失败，请检查图片！"<<endl;
        }

        for(;symbol != imageZbar.symbol_end();++symbol)
        {
            cout<<"类型："<<endl<<symbol->get_type_name()<<endl;
            cout<<"条码："<<endl<<symbol->get_data()<<endl;
        }

        imageZbar.set_data(NULL,0);

         chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"遍历图像用时："<<time_used.count()<<" 秒。"<<endl;
    waitKey(0);
    return(0);
}


/*-----------------------------------------
#include "zbar.h"
//#include "cv.h"
//#include "highgui.h"


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace zbar;  //添加zbar名称空间
using namespace cv;

int main(int argc,char*argv[])
{
	Mat imageSource=imread(argv[1],0);
	Mat image;
	imageSource.copyTo(image);
	GaussianBlur(image,image,Size(3,3),0);  //滤波
	threshold(image,image,100,255,CV_THRESH_BINARY);  //二值化
	imshow("二值化",image);
	Mat element=getStructuringElement(2,Size(7,7));	 //膨胀腐蚀核
	//morphologyEx(image,image,MORPH_OPEN,element);
	for(int i=0;i<10;i++)
	{
		erode(image,image,element);
		i++;
	}
	imshow("腐蚀s",image);
	Mat image1;
	erode(image,image1,element);
	image1=image-image1;
	imshow("边界",image1);
	//寻找直线 边界定位也可以用findContours实现
	vector<Vec2f>lines;
	HoughLines(image1,lines,1,CV_PI/150,250,0,0);
	Mat DrawLine=Mat::zeros(image1.size(),CV_8UC1);
	for(int i=0;i<lines.size();i++)
	{
		float rho=lines[i][0];
		float theta=lines[i][1];
		Point pt1,pt2;
		double a=cos(theta),b=sin(theta);
		double x0=a*rho,y0=b*rho;
		pt1.x=cvRound(x0+1000*(-b));
		pt1.y=cvRound(y0+1000*a);
		pt2.x=cvRound(x0-1000*(-b));
		pt2.y=cvRound(y0-1000*a);
		line(DrawLine,pt1,pt2,Scalar(255),1,CV_AA);
	}
	imshow("直线",DrawLine);
	Point2f P1[4];
	Point2f P2[4];
	vector<Point2f>corners;
	goodFeaturesToTrack(DrawLine,corners,4,0.1,10,Mat()); //角点检测
	for(int i=0;i<corners.size();i++)
	{
		circle(DrawLine,corners[i],3,Scalar(255),3);
		P1[i]=corners[i];
	}
	imshow("交点",DrawLine);
	int width=P1[1].x-P1[0].x;
	int hight=P1[2].y-P1[0].y;
	P2[0]=P1[0];
	P2[1]=Point2f(P2[0].x+width,P2[0].y);
	P2[2]=Point2f(P2[0].x,P2[1].y+hight);
	P2[3]=Point2f(P2[1].x,P2[2].y);
	Mat elementTransf;
	elementTransf=	getAffineTransform(P1,P2);
	warpAffine(imageSource,imageSource,elementTransf,imageSource.size(),1,0,Scalar(255));
	imshow("校正",imageSource);
	//Zbar二维码识别
	ImageScanner scanner;
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
	int width1 = imageSource.cols;
	int height1 = imageSource.rows;
	uchar *raw = (uchar *)imageSource.data;
	Image imageZbar(width1, height1, "Y800", raw, width * height1);
	scanner.scan(imageZbar); //扫描条码
	Image::SymbolIterator symbol = imageZbar.symbol_begin();
	if(imageZbar.symbol_begin()==imageZbar.symbol_end())
	{
		cout<<"查询条码失败，请检查图片！"<<endl;
	}
	for(;symbol != imageZbar.symbol_end();++symbol)
	{
		cout<<"类型："<<endl<<symbol->get_type_name()<<endl<<endl;
		cout<<"条码："<<endl<<symbol->get_data()<<endl<<endl;
	}
	namedWindow("Source Window",0);
	imshow("Source Window",imageSource);
	waitKey();
	imageZbar.set_data(NULL,0);
	return 0
	}
	------------------*/
