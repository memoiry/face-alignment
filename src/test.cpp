#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>

#include "lbf.hpp"

using namespace cv;
using namespace std;
using namespace lbf;

// dirty but works
void parseTxt(const string &txt, vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<BBox> &bboxes);

int test(int argc, char* argv[]) {
    if(argc != 2) {
      printf("test model_file test.txt\n");
      return -1;
    }

    Config &config = Config::GetInstance();
    LbfCascador lbf_cascador;
    FILE *fd = fopen(argv[0], "rb");
    lbf_cascador.Read(fd);
    fclose(fd);

    LOG("Load test data from %s", argv[1]);
    vector<Mat> imgs, gt_shapes;
    vector<BBox> bboxes;
    parseTxt(argv[1], imgs, gt_shapes, bboxes);

    int N = imgs.size();
    lbf_cascador.Test(imgs, gt_shapes, bboxes);

    return 0;
}

int run(int argc, char* argv[]) {
    if(argc != 2) {
      printf("run model_file test.txt\n");
      return -1;
    }

    Config &config = Config::GetInstance();
    FILE *fd = fopen(argv[1], "r");
    assert(fd);
    int N;
    int landmark_n;
    fscanf(fd, "%d%d", &N, &landmark_n);
    char img_path[256];
    double bbox[4];
    vector<double> x(landmark_n), y(landmark_n);

    LbfCascador lbf_cascador;
    FILE *model = fopen(argv[0], "rb");
    lbf_cascador.Read(model);
    fclose(model);

    for (int i = 0; i < N; i++) {
        fscanf(fd, "%s", img_path);
        for (int j = 0; j < 4; j++) {
            fscanf(fd, "%lf", &bbox[j]);
        }
        for (int j = 0; j < landmark_n; j++) {
            fscanf(fd, "%lf%lf", &x[j], &y[j]);
        }
        Mat img = imread(img_path);
        // crop img
        double x_min, y_min, x_max, y_max;
        x_min = *min_element(x.begin(), x.end());
        x_max = *max_element(x.begin(), x.end());
        y_min = *min_element(y.begin(), y.end());
        y_max = *max_element(y.begin(), y.end());
        x_min = max(0., x_min - bbox[2] / 2);
        x_max = min(img.cols - 1., x_max + bbox[2] / 2);
        y_min = max(0., y_min - bbox[3] / 2);
        y_max = min(img.rows - 1., y_max + bbox[3] / 2);
        double x_, y_, w_, h_;
        x_ = x_min; y_ = y_min;
        w_ = x_max - x_min; h_ = y_max - y_min;
        BBox bbox_(bbox[0] - x_, bbox[1] - y_, bbox[2], bbox[3]);
        Rect roi(x_, y_, w_, h_);
        img = img(roi).clone();

        Mat gray, shape;
        cvtColor(img, gray, CV_BGR2GRAY);
        LOG("Run %s", img_path);
	TIMER_BEGIN
        shape = lbf_cascador.Predict(gray, bbox_);
	LOG("alignment time costs %.4lf s", TIMER_NOW);
	TIMER_END
        img = drawShapeInImage(img, shape, bbox_);
        imshow("landmark", img);
        //imwrite("result.jpg",img);       
        waitKey(0);
    }
    fclose(fd);
    return 0;
}

int live(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "model_name cascade_name image_name\n");
        return -1;
    }

    const char *model_name = argv[0];
    const char *cascade_name = argv[1];
    const char *image_name = argv[2];

    Config &config = Config::GetInstance();
    int N;
    int landmark_n = config.landmark_n;
    double bbox[4];
    double scale = 1.3;
    vector<double> x(landmark_n), y(landmark_n);
    vector<Rect> faces;
    vector<Mat> shapes;
    vector<BBox> bboxes;

    CascadeClassifier cascade;
    if(!cascade.load(cascade_name)){
        fprintf(stderr, "Could not load classifier cascade %s\n", cascade_name);
        return -1;
    }

    LbfCascador lbf_cascador;
    FILE *model = fopen(model_name, "rb");
    if(!model) {
        fprintf(stderr, "Could not load lbf model %s\n", model_name);
        return -1;
    }
    lbf_cascador.Read(model);
    fclose(model);

    Mat img = imread(image_name);
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    double t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0|CV_HAAR_SCALE_IMAGE,
        Size(30, 30));

    t = (double)cvGetTickCount() - t;
    printf("detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

    for(vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++){
        Mat shape;
        bbox[0] = r->x*scale;
        bbox[1] = r->y*scale;
        bbox[2] = r->width*scale;
        bbox[3] = r->height*scale;
        BBox bbox_(bbox[0], bbox[1], bbox[2], bbox[3]);

        t =(double)cvGetTickCount();
        shape = lbf_cascador.Predict(gray, bbox_);
        t = (double)cvGetTickCount() - t;
        printf( "alignment time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
        shapes.push_back(shape);
	bboxes.push_back(bbox_);
    }

    for(int i = 0; i < shapes.size(); i++) {
        img = drawShapeInImage(img, shapes[i], bboxes[i]);
    }

    imshow("landmark", img);
    imwrite("result.jpg", img);
    waitKey(0);
    return 0;
}

int camera(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "model_name cascade_name camera_index\n");
        return -1;
    }

    const char *model_name = argv[0];
    const char *cascade_name = argv[1];
    int camera_index = atoi(argv[2]);

    Config &config = Config::GetInstance();
    int landmark_n = config.landmark_n;
    double bbox[4];
    double scale = 1.3;
    vector<double> x(landmark_n), y(landmark_n);
    cv::VideoCapture vc(camera_index);
    Mat frame, frameCopy, image, img;
    vc >> frame;

    CascadeClassifier cascade;
    if(!cascade.load(cascade_name)){
        fprintf(stderr, "Could not load classifier cascade %s\n", cascade_name);
        return -1;
    }

    LbfCascador lbf_cascador;
    FILE *model = fopen(model_name, "rb");
    if(!model) {
        fprintf(stderr, "Could not load lbf model %s\n", model_name);
        return -1;
    }
    lbf_cascador.Read(model);
    fclose(model);

    for(;;){
    	vector<BBox> bboxes;
    	vector<Mat> shapes;
    	vector<Rect> faces;
    	vc >> frame;
        Mat gray, smallImg;
        double t;
   	cvtColor(frame, gray, CV_BGR2GRAY);

    	smallImg = Mat(cvRound (frame.rows/scale), cvRound(frame.cols/scale), CV_8UC1 );
    	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
    	equalizeHist(smallImg, smallImg);
    	t = (double)cvGetTickCount();
    	cascade.detectMultiScale( smallImg, faces,
    		1.1, 2, 0|CV_HAAR_SCALE_IMAGE,
    		Size(30, 30) );
    	t = (double)cvGetTickCount() - t;
    	printf("detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
        if(faces.size() == 0)
            continue;

    	for(vector<Rect>::iterator r = faces.begin(); r != faces.end(); r++){
    	    Mat shape;
    	    bbox[0] = r->x*scale;
    	    bbox[1] = r->y*scale;
            bbox[2] = r->width*scale;
    	    bbox[3] = r->height*scale;
    	    BBox bbox_(bbox[0], bbox[1], bbox[2], bbox[3]);
    	    t =(double)cvGetTickCount();
    	    shape = lbf_cascador.Predict(gray, bbox_);
    	    t = (double)cvGetTickCount() - t;
    	    printf("alignment time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    	    shapes.push_back(shape);
    	    bboxes.push_back(bbox_);
    	}

    	for(int i = 0; i < shapes.size(); i++) {
    	    frame = drawShapeInImage(frame, shapes[i], bboxes[i]);
    	}
    	imshow("camera", frame);

        if(waitKey(1) == 'q')
            break;
    }

    return 0;
}
