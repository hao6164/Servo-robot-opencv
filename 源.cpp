#include "cv.h" 
#include "highgui.h"
#include "cxcore.h"
#include "camerads.h"

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <assert.h> 
#include <math.h> 
#include <float.h> 
#include <limits.h> 
#include <time.h> 
#include <ctype.h>
#include <fstream>
#include <iostream> 
#include <list>
#ifdef _EiC 
#define WIN32 
#endif
using namespace std;

static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;
bool writevideo_switch = false;
bool outfile_switch = true;
#define CAM 0
#define VIDEO 1
int camera_or_video = VIDEO;
int tempdata = 0;

//原始数据环形缓冲区
#define cyclebufsize 30
int xcyclebuf[cyclebufsize];
int ycyclebuf[cyclebufsize];
int cyclepointer = 0;
bool cyclebufinit = false;
int process_x_predict = 0;
int process_y_predict = 0;

typedef struct haolocationdata
{
	int x = 0;
	int y = 0;
	int size = 0;
	int height = 0;
	bool can_use = FALSE;
	haolocationdata(int a, int b, int c)
	{
		x = a;
		y = b;
		size = c;
	};
	void set(int a, int b, int c, int d, bool e)
	{
		x = a;
		y = b;
		size = c;
		height = d;
		can_use = e;
	};
}
haolocationdata;
typedef struct haoboxdata
{
	int x = 0;
	int y = 0;
	int size = 0;
	haoboxdata(int a, int b, int c)
	{
		x = a;
		y = b;
		size = c;
	};
	void set(int a, int b, int c)
	{
		x = a;
		y = b;
		size = c;
	};
	bool can_use = FALSE;
}
haoboxdata;
typedef list<haoboxdata> Listhaoboxdata;

Listhaoboxdata listhaoboxdata;

//声明i为迭代器

Listhaoboxdata::iterator list_Listhaoboxdata_iterator;
typedef list<CvPoint> Listcvpoint;

Listcvpoint listcvpoint;

//声明i为迭代器

Listcvpoint::iterator listiterator;
haolocationdata location_storage(0, 0, 0);
//using namespace cv;

bool directshow_camera_initial(CCameraDS* camera);
void detect_and_draw(IplImage* image);
void detect_feet(IplImage* erode_image, haolocationdata Olocation, int haoboxrate);
void feet_point_plot(IplImage* img);
void putintocyclebuf();
int average(int* databuf, int acceptdatanum);

const char* cascade_name = "haarcascade_lowerbody.xml";
ofstream outfile("F:\\360data\\重要数据\\桌面\\毕业论文\\MATLABdata.txt");


int main(int argc, char** argv)
{
	//载入harr模板
	cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);

	if (!cascade)
	{
		fprintf(stderr, "ERROR: Could not load classifier cascade\n");
		return -1;
	}
	storage = cvCreateMemStorage(0);
     CCameraDS camera;
	 CvCapture * pCap;
	//初始化DirectShow摄像头
	 if (camera_or_video == CAM)
	 {
		 if (!directshow_camera_initial(&camera))
			 return -1;
	}
	 else
	 {
		 pCap = cvCreateFileCapture("test11.avi");//视频文件
	 }

	

	cvClearMemStorage(storage);

	///////////////////保存视频部分
	int fps = 20; //捕捉帧率 
	CvVideoWriter* writer = 0; //保存就加上这句
	int isColol = 1;
	int frameW = 640;
	int frameH = 480;
	writer = cvCreateVideoWriter("out.avi", CV_FOURCC('F', 'L', 'V', '1'), fps, cvSize(frameW, frameH), isColol);



	double t = (double)cvGetTickCount();
	while (1)
	{
		IplImage *pFrame;
		//获取一帧
		if (camera_or_video == CAM)
		{

			pFrame = camera.QueryFrame();
		}
		else
		{
			pFrame = cvQueryFrame(pCap);
			if (pFrame == 0)
			{
				break;
			}
		}

		//显示
		//cvShowImage("camera", pFrame)
		t = (double)cvGetTickCount();;
		detect_and_draw(pFrame);
		t = (double)cvGetTickCount() - t;
		printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));

		if (writevideo_switch)
		{
			cvWriteFrame(writer, pFrame);
		}


		if (cvWaitKey(1) == 'q')
			break;
		if ((location_storage.x != 0 || location_storage.y != 0) )
		{
			putintocyclebuf();
			process_x_predict=average(xcyclebuf, 6);
			process_y_predict = average(ycyclebuf, 6);
		}
		if ((location_storage.x != 0 || location_storage.y != 0) && outfile_switch)
		{
			outfile << process_x_predict << " " << process_y_predict << " " << location_storage.x << " " << location_storage.y << endl;
			feet_point_plot(pFrame);
		}
			

	}
	if (camera_or_video==CAM)
	{
		camera.CloseCamera(); //可不调用此函数，CCameraDS析构时会自动关闭摄像头
	}
	
	

	cvDestroyWindow("camera");
	return 0;
	/*	VideoCapture cap(1); // open the default camera
		if (!cap.isOpened())  // check if we succeeded
		return -1;

		Mat edges;
		namedWindow("edges", 1);
		for (;;)
		{
		double t = (double)cvGetTickCount();
		Mat frame;
		cap >> frame; // get a new frame from camera
		//cvtColor(frame, edges, CV_BGR2GRAY);
		//	GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		//	Canny(edges, edges, 0, 30, 3);
		imshow("edges", frame);
		if (waitKey(30) >= 0) break;
		t = (double)cvGetTickCount() - t;
		printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));
		t = (double)cvGetTickCount();
		}
		// the camera will be deinitialized automatically in VideoCapture destructor
		return 0;

		//cascade_name = "haarcascade_mcs_upperbody.xml";
		cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);

		if (!cascade)
		{
		fprintf(stderr, "ERROR: Could not load classifier cascade\n");
		return -1;
		}
		storage = cvCreateMemStorage(0);
		//cvNamedWindow("result", 1);

		//const char* filename = "2.jpg";
		//IplImage* image = cvLoadImage(filename, 1);
		//CvCapture * pCap = cvCreateFileCapture("test11.avi");//视频文件
		CvCapture* pCap = cvCreateCameraCapture(1);//摄像头这里-1也可以，不过我的电脑装的有CyberLink YouCam软件，
		//OpenCV会默认调用该摄像头，而不调用系统的驱动
		IplImage *frame = NULL;

		if (cvCreateCameraCapture == NULL)
		{
		return(0);
		}

		//cvNamedWindow("Camera", CV_WINDOW_FULLSCREEN);
		cvClearMemStorage(storage);
		double t = (double)cvGetTickCount();
		while ((frame = cvQueryFrame(pCap)) != 0 && cvWaitKey(1) != 27)
		{



		double t = (double)cvGetTickCount();
		frame = cvQueryFrame(pCap);

		//	detect_and_draw(frame);
		t = (double)cvGetTickCount() - t;
		printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));
		t = (double)cvGetTickCount();
		//cvShowImage("result", frame);
		//cvWaitKey(2000);
		}

		cvReleaseCapture(&pCap);
		cvDestroyWindow("Camera");
		return 0;*/


}
bool directshow_camera_initial(CCameraDS* camerapointer)
{
	int cam_count;

	//仅仅获取摄像头数目
	cam_count = CCameraDS::CameraCount();
	printf("There are %d cameras.\n", cam_count);


	//获取所有摄像头的名称
	for (int i = 0; i < cam_count; i++)
	{
		char camera_name[1024];
		int retval = CCameraDS::CameraName(i, camera_name, sizeof(camera_name));

		if (retval >0)
			printf("Camera %d Name is %s.\n", i, camera_name);
		else
			printf("Can not get Camera #%d's name.\n", i);
	}
	int camera_use = 0;
	if (cam_count>1)
	{
		printf("Select the camera you use\n");
		cin >> camera_use;
	}
	

	if (cam_count == 0)
		return false;




	//打开摄像头
	if (!camerapointer->OpenCamera(camera_use, false, 640, 480)) //不弹出属性选择窗口，用代码制定图像宽和高
	{
		fprintf(stderr, "Can not open camera.\n");
		return false;
	}
	return true;
}
void detect_and_draw(IplImage* img)
{

	double scale = 1.3;
	static CvScalar colors[] = {
		{ { 0, 0, 255 } }, { { 0, 128, 255 } }, { { 0, 255, 255 } }, { { 0, 255, 0 } },
		{ { 255, 128, 0 } }, { { 255, 255, 0 } }, { { 255, 0, 0 } }, { { 65, 106, 225 } }
	};//Just some pretty colors to draw with

	//Image Preparation 
	// 
	IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);
	IplImage* small_img = cvCreateImage(cvSize(cvRound(img->width / scale), cvRound(img->height / scale)), 8, 1);
	//3月31日精简	IplImage* pCannyImg = NULL;
	IplImage* haopicsrc = NULL;
	IplImage* CannyBinarypic = NULL;
	IplImage* img_dilate = NULL;
	IplImage* img_erode = NULL;


	cvCvtColor(img, gray, CV_BGR2GRAY);
	cvResize(gray, small_img, CV_INTER_LINEAR);

	cvEqualizeHist(small_img, small_img); //直方图均衡


	//Detect objects if any 
	// 
	//cvClearMemStorage(storage);



	CvSeq* objects = cvHaarDetectObjects(small_img,
		cascade,
		storage,
		1.1,
		2,
		/*0*/CV_HAAR_DO_CANNY_PRUNING,
		cvSize(140, 140));



	if (location_storage.can_use)//&& (objects ? objects->total : 0)==0
	{
		//cvRectangle(img, cvPoint(location_storage.x - 30, location_storage.y - 30), cvPoint(location_storage.x+ 30, location_storage.y+ 30), colors[3 % 8], 3);
		int predictROIx = location_storage.x;
		int predictROIy = location_storage.y;
		int predictROIwidth = 130;
		int predictROIheight = 130;
		if (predictROIx - predictROIwidth / 2 < 0)
			predictROIx = predictROIwidth / 2;
		else if (predictROIx + predictROIwidth / 2 > img->width)
			predictROIx = img->width - predictROIwidth / 2;
		if (predictROIy - predictROIheight*0.8<0)
			predictROIy = predictROIheight*0.8;
		else if (predictROIy + predictROIheight*0.2>img->height)
			predictROIy = img->height - predictROIheight*0.2;/**/
		haopicsrc = cvCreateImage(cvSize(predictROIwidth, predictROIheight), IPL_DEPTH_8U, 1);//识别人形大小
		CannyBinarypic = cvCreateImage(cvGetSize(haopicsrc), IPL_DEPTH_8U, 1);
		img_dilate = cvCreateImage(cvGetSize(CannyBinarypic), IPL_DEPTH_8U, 1);
		img_erode = cvCreateImage(cvGetSize(CannyBinarypic), IPL_DEPTH_8U, 1);
		cvRectangle(img, cvPoint(predictROIx - predictROIwidth / 2, predictROIy - predictROIheight*0.8), cvPoint(predictROIx + predictROIwidth / 2, predictROIy + predictROIheight*0.2), colors[7 % 8], 3);

		cvSetImageROI(gray, cvRect(predictROIx - predictROIwidth / 2, predictROIy - predictROIheight*0.8, predictROIwidth, predictROIheight));
		//cvRectangle(img, cvPoint(predictROIx - location_storage.size / 2, predictROIy - location_storage.height * 3 / 4), cvPoint(predictROIx + location_storage.size / 2, predictROIy + location_storage.height * 1 / 4), colors[3 % 8], 3);
		//cvSetImageROI(gray, cvRect(0,0, location_storage.size, location_storage.height));
		cvResize(gray, haopicsrc);


		cvThreshold(haopicsrc, CannyBinarypic, 59, 255, CV_THRESH_BINARY);


		cvDilate(CannyBinarypic, img_dilate, NULL, 4);

		cvErode(img_dilate, img_erode, NULL, 4);

		detect_feet(img_erode, haolocationdata(predictROIx - predictROIwidth / 2, predictROIy - predictROIheight*0.8, 0), 8);

		cvResetImageROI(img);
	}


	//Loop through found objects and draw boxes around them 
	//if (location_storage.can_use == false)
	//{
	for (int i = 0; i < (objects ? objects->total : 0); ++i)
	{
		CvRect* r = (CvRect*)cvGetSeqElem(objects, i);
		cvRectangle(img, cvPoint(r->x*scale, r->y*scale), cvPoint((r->x + r->width)*scale, (r->y + r->height)*scale), colors[i % 8], 3);
		//outfile << r->x + r->width / 2 << " " << r->y + r->height / 2 << " " << endl;
		//cout << r->x + r->width / 2 << " " << r->y + r->height / 2 << " " << endl;
		haopicsrc = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1);//同摄像头大小
		haopicsrc = cvCreateImage(cvSize(r->width*scale, r->height*scale + 30), IPL_DEPTH_8U, 1);//识别人形大小
		CannyBinarypic = cvCreateImage(cvGetSize(haopicsrc), IPL_DEPTH_8U, 1);
		img_dilate = cvCreateImage(cvGetSize(CannyBinarypic), IPL_DEPTH_8U, 1);
		img_erode = cvCreateImage(cvGetSize(CannyBinarypic), IPL_DEPTH_8U, 1);
		cvSetImageROI(gray, cvRect(r->x*scale, r->y*scale, r->width*scale, r->height*scale + 30));
		//	cvSetImageROI(haopicsrc, cvRect(r->x*scale, r->y*scale, r->width*scale, r->height*scale + 30));


		cvResize(gray, haopicsrc);

		//cvResetImageROI(haopicsrc);
		cvThreshold(haopicsrc, CannyBinarypic, 59, 255, CV_THRESH_BINARY);

		for (size_t i = 0; i < 1; i++)
		{
			cvDilate(CannyBinarypic, img_dilate, NULL, 3);

			cvErode(img_dilate, img_erode, NULL, 3);

			detect_feet(img_erode, haolocationdata(r->x*scale, r->y*scale, 0), 12);
		}

		// 显示二值图  


		//3月31日精简pCannyImg = cvCreateImage(cvGetSize(haopicsrc), IPL_DEPTH_8U, 1);//3月31日精简
		//3月31日精简cvCanny(haopicsrc, pCannyImg, 400, 500, 3);



		cvResetImageROI(img);



		//3月31日精简listcvpoint.push_back(cvPoint((r->x + r->width / 2)*scale, (r->y + r->height / 2)*scale));//中心动态点放入链表中


	}
	//}
	CvPoint cvpointlast;
	bool CvPointcount = FALSE;
	//3月31日精简
	/*for (listiterator = listcvpoint.begin(); listiterator != listcvpoint.end(); ++listiterator)

	{
	if (CvPointcount == TRUE)
	cvLine(img, *listiterator, cvpointlast, colors[3 % 8], 1, 8, 0);

	CvPointcount = TRUE;
	cvpointlast = *listiterator;
	}*/
	cvRectangle(img, cvPoint(location_storage.x - 10, location_storage.y - 10), cvPoint(location_storage.x + 10, location_storage.y + 10), colors[3 % 8], 3);
	cvShowImage("haopicsrc", haopicsrc);
	//3月31日精简cvShowImage("canny", pCannyImg);
	cvShowImage("binarypic", CannyBinarypic);
	cvShowImage("erode", img_erode);
	cvShowImage("dilate", img_dilate);
	cvShowImage("binarypic", CannyBinarypic);
	cvShowImage("result", img);
	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);
	//3月31日精简cvReleaseImage(&pCannyImg);
	cvReleaseImage(&haopicsrc);

	//cvWaitKey(0);

}
void detect_feet(IplImage* erode_image, haolocationdata Olocation, int haoboxrate)
{
	//static int haoboxrate = 12;//检测框与原图的比例
	static float haoboxpixelThreshold = 0.8;//阈值
	unsigned char* base_point = NULL;
	unsigned char* base_point_box = NULL;
	IplImage* feet_temp_show = cvCreateImage(cvGetSize(erode_image), IPL_DEPTH_8U, 1);
	cvCopy(erode_image, feet_temp_show);
	int Pixelcount = 0;
	static CvScalar colors[] = {
		{ { 0, 0, 0 } }, { { 0, 128, 255 } }, { { 0, 255, 255 } }, { { 0, 255, 0 } },
		{ { 255, 128, 0 } }, { { 255, 255, 0 } }, { { 255, 0, 0 } }, { { 255, 0, 255 } } };

	CvSize haobox = cvSize((int)(erode_image->width / haoboxrate), (int)(erode_image->width / haoboxrate));//正方形框
	base_point = (unsigned char*)erode_image->imageData + erode_image->width / 2 + erode_image->widthStep*(erode_image->height - haobox.height / 2 - 1);

	for (size_t y = 0; y < erode_image->height - haobox.height; y = y + 3)
	{
		for (size_t x = 1; x < erode_image->width / 2 - haobox.width / 2; x = x + 3)
		{
			base_point_box = base_point + x - y*erode_image->widthStep;//右片
			Pixelcount = 0;//初始像素统计

			//	cvRectangle(erode_image, cvPoint(erode_image->width / 2 - x - haobox.width / 2, erode_image->height - y - haobox.width ), cvPoint(erode_image->width / 2 - x + haobox.width / 2, erode_image->height - y + haobox.width ), colors[0 % 8], 1);

			//	cvShowImage("erode", erode_image);
			//	cvWaitKey(1000);

			for (size_t county = 0; county < haobox.height; county += 3)
				//框内像素统计
			{

				for (size_t countx = 0; countx < haobox.width; countx += 3)

				{

					if (*(base_point_box - haobox.width / 2 - haobox.height / 2 * erode_image->widthStep + countx + county*erode_image->widthStep) == 0)

					{
						Pixelcount += 1;
					}

				}/**/
			}

			if (Pixelcount > haobox.width*haobox.width / 9 * haoboxpixelThreshold)
			{
				listhaoboxdata.push_back(haoboxdata(erode_image->width / 2 + x, erode_image->height - haobox.width / 2 - 1 - y, haobox.width));

			}

			base_point_box = base_point - x - y*erode_image->widthStep;//左片
			Pixelcount = 0;//初始像素统计

			for (size_t county = 0; county < haobox.height; county += 3)
				//框内像素统计
			{

				for (size_t countx = 0; countx < haobox.width; countx += 3)

				{

					if (*(base_point_box - haobox.width / 2 - haobox.height / 2 * erode_image->widthStep + countx + county*erode_image->widthStep) == 0)

					{
						Pixelcount += 1;
					}

				}/**/
			}

			if (Pixelcount > haobox.width*haobox.width / 9 * haoboxpixelThreshold)
			{
				listhaoboxdata.push_back(haoboxdata(erode_image->width / 2 - x, erode_image->height - haobox.width / 2 - 1 - y, haobox.width));
			}
			if (listhaoboxdata.size() > 20)
				break;


		}
		cvRectangle(feet_temp_show, cvPoint(0, 0), cvPoint(haobox.width, haobox.width), colors[0 % 8], 1);
		if (listhaoboxdata.size() > 20)
		{
			//	list_Listhaoboxdata_iterator = listhaoboxdata.begin();
			//	cvRectangle(feet_temp_show, cvPoint(list_Listhaoboxdata_iterator->x - list_Listhaoboxdata_iterator->size / 2, list_Listhaoboxdata_iterator->y - list_Listhaoboxdata_iterator->size / 2), cvPoint(list_Listhaoboxdata_iterator->x + list_Listhaoboxdata_iterator->size / 2, list_Listhaoboxdata_iterator->y + list_Listhaoboxdata_iterator->size / 2), colors[0 % 8], 1);

			//	location_storage.set(list_Listhaoboxdata_iterator->x + Olocation.x, list_Listhaoboxdata_iterator->y + Olocation.y, erode_image->width, erode_image->height, TRUE);
			Listhaoboxdata::iterator listiteratormin_x_plus_y;
			int min_x_plus_y = erode_image->height + erode_image->width;
			/**/for (list_Listhaoboxdata_iterator = listhaoboxdata.begin(); list_Listhaoboxdata_iterator != listhaoboxdata.end(); ++list_Listhaoboxdata_iterator)

			{

				cvRectangle(feet_temp_show, cvPoint(list_Listhaoboxdata_iterator->x - list_Listhaoboxdata_iterator->size / 2, list_Listhaoboxdata_iterator->y - list_Listhaoboxdata_iterator->size / 2), cvPoint(list_Listhaoboxdata_iterator->x + list_Listhaoboxdata_iterator->size / 2, list_Listhaoboxdata_iterator->y + list_Listhaoboxdata_iterator->size / 2), colors[0 % 8], 1);
				if (min_x_plus_y > erode_image->height - list_Listhaoboxdata_iterator->y + list_Listhaoboxdata_iterator->x)
				{
					min_x_plus_y = erode_image->height - list_Listhaoboxdata_iterator->y + list_Listhaoboxdata_iterator->x;
					listiteratormin_x_plus_y = list_Listhaoboxdata_iterator;
				}
				//	location_storage.set(list_Listhaoboxdata_iterator->x + Olocation.x, list_Listhaoboxdata_iterator->y + Olocation.y, erode_image->width, erode_image->height, TRUE);
			}
			//location_storage.set(listiteratormin_x_plus_y->x + Olocation.x, listiteratormin_x_plus_y->y + Olocation.y, erode_image->width, erode_image->height, TRUE);
			if (((listiteratormin_x_plus_y->x + Olocation.x - location_storage.x<300) && (listiteratormin_x_plus_y->x + Olocation.x - location_storage.x>-300)) || location_storage.can_use == false)
			{
				location_storage.set(listiteratormin_x_plus_y->x + Olocation.x, listiteratormin_x_plus_y->y + Olocation.y, erode_image->width, erode_image->height, TRUE);
			}
			listhaoboxdata.clear();
			break;
		}
	}
	cvShowImage("feet_temp_show", feet_temp_show);
}
void feet_point_plot(IplImage* img)
{
	listcvpoint.push_back(cvPoint(process_x_predict, tempdata));//中心动态点放入链表中
	tempdata++;


CvPoint cvpointlast;
bool CvPointcount = FALSE;
//3月31日精简
for (listiterator = listcvpoint.begin(); listiterator != listcvpoint.end(); ++listiterator)

{
if (CvPointcount == TRUE)
cvLine(img, *listiterator, cvpointlast, { { 0, 255, 0 } }, 1, 8, 0);

CvPointcount = TRUE;
cvpointlast = *listiterator;
}/**/
cvShowImage("plot", img);
}
void putintocyclebuf(void)
{
	if (cyclebufinit == false)//初始化环形缓冲区 填充为第一个获取的位置值
	{
		cyclebufinit = true;
		for (size_t i = 0; i < cyclebufsize; i++)
		{
        xcyclebuf[i] = location_storage.x;
		ycyclebuf[i] = location_storage.y;
		}
		
	}
	xcyclebuf[cyclepointer] = location_storage.x;
	ycyclebuf[cyclepointer] = location_storage.y;
	if (cyclepointer < cyclebufsize-1)
		cyclepointer++;
	else
		cyclepointer = 0;
}
int average(int* databuf, int acceptdatanum)
{
	int temp;
	int arr[cyclebufsize];
	for (int i = 0; i<cyclebufsize ; i++)
	{
		arr[i] = databuf[i];
	}
	for (int i = 0; i<cyclebufsize - 1; i++)
	{
		for (int j = 0; j<cyclebufsize - 1 - i; j++)
		{
			if (arr[j]>arr[j + 1])
			{
				temp = arr[j + 1];
				arr[j + 1] = arr[j];
				arr[j] = temp;
			}
		}
	}
	int tempsum=0;
	for (size_t i = 0; i <acceptdatanum; i++)
	{
		tempsum+=arr[i + (cyclebufsize - acceptdatanum) / 2];
	}
	return tempsum / acceptdatanum;
}