#include <iostream>
#include <fstream>
#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

CascadeClassifier cc;

Rect getBBox(Mat &img, Mat_<double> &shape) {
    vector<Rect> rects;
    cc.detectMultiScale(img, rects, 1.05, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
    if (rects.size() == 0) return Rect(-1, -1, -1, -1);
    double center_x, center_y, x_min, x_max, y_min, y_max;
    center_x = center_y = 0;
    x_min = x_max = shape(0, 0);
    y_min = y_max = shape(0, 1);
    for (int i = 0; i < shape.rows; i++) {
        center_x += shape(i, 0);
        center_y += shape(i, 1);
        x_min = min(x_min, shape(i, 0));
        x_max = max(x_max, shape(i, 0));
        y_min = min(y_min, shape(i, 1));
        y_max = max(y_max, shape(i, 1));
    }
    center_x /= shape.rows;
    center_y /= shape.rows;

    for (int i = 0; i < rects.size(); i++) {
        Rect r = rects[i];
        if (x_max - x_min > r.width*1.5) continue;
        if (y_max - y_min > r.height*1.5) continue;
        if (abs(center_x - (r.x + r.width / 2)) > r.width / 2) continue;
        if (abs(center_y - (r.y + r.height / 2)) > r.height / 2) continue;
        return r;
    }
    return Rect(-1, -1, -1, -1);
}

int NUM_LAND_MARK = 0;

void genTxt(const string &inTxt, const string &outTxt) {
    FILE *inFile = fopen(inTxt.c_str(), "r");
    FILE *outFile = fopen(outTxt.c_str(), "w");
    assert(inFile && outFile);

    char line[256];
    char buff[1000];
    string out_string("");
    int N = 0;
    while (fgets(line, sizeof(line), inFile)) {
        string img_path(line, strlen(line) - 1);
        printf("Handle image %s\n", img_path.c_str());
        string pts = img_path.substr(0, img_path.find_last_of(".")) + ".pts";
        printf("Handle point %s\n", pts.c_str());
        fstream fp;
        fp.open(pts.c_str(), ios::in);
        if(!fp.is_open()) continue;
	      string temp;
        getline(fp, temp);
      	getline(fp, temp, ' ');
      	getline(fp, temp);
      	int NbOfPoints = atoi(temp.c_str());
      	printf("Handle point %d\n", NbOfPoints);
      	getline(fp, temp);
      	if(NUM_LAND_MARK == 0) {
          NUM_LAND_MARK = NbOfPoints;
      	} else if(NUM_LAND_MARK != NbOfPoints) {
      		printf("ERROR: Points data does NOT has the same size!\n%s has %d points, but other has %d points\n",
      			pts.c_str(), NbOfPoints, NUM_LAND_MARK);
      		exit(0);
      	}

        Mat_<double> gt_shape(NUM_LAND_MARK, 2);
        for (int i = 0; i < NUM_LAND_MARK; i++) {
            fp >> gt_shape(i, 0) >> gt_shape(i, 1);
        }
        fp.close();

        Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
        Rect bbox = getBBox(img, gt_shape);

        if (bbox.x != -1) {
            N++;
            sprintf(buff, "%s %d %d %d %d", img_path.c_str(), bbox.x, bbox.y, bbox.width, bbox.height);
            out_string += buff;
            printf("%d %d %d %d ", bbox.x, bbox.y, bbox.width, bbox.height);
            for (int i = 0; i < NUM_LAND_MARK; i++) {
                sprintf(buff, " %lf %lf", gt_shape(i, 0), gt_shape(i, 1));
                out_string += buff;
                printf("%lf %lf ", gt_shape(i, 0), gt_shape(i, 1));
            }
            out_string += "\n";
        }
    }
    fprintf(outFile, "%d %d\n%s", N, NUM_LAND_MARK, out_string.c_str());

    fclose(inFile);
    fclose(outFile);
}

int prepare(int argc, char* argv[]) {
    if(argc != 3) {
        printf("prepare path_images_train.txt cascade_file output.txt\n");
	return -1;
    }
    if(!cc.load(argv[1])) {
	printf("failed to load cascade file %s\n", argv[1]);
	return -1;
    }

    genTxt(argv[0], argv[2]);
    return 0;
}
