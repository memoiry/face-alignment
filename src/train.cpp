#include <ctime>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lbf.hpp"

using namespace cv;
using namespace std;
using namespace lbf;

void parseTxt(const string &txt, vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<BBox> &bboxes) {
    Config &config = Config::GetInstance();
    FILE *fd = fopen(txt.c_str(), "r");
    assert(fd);
    int N;
    int landmark_n;
    fscanf(fd, "%d", &N);
    fscanf(fd, "%d", &landmark_n);
    imgs.resize(N);
    gt_shapes.resize(N);
    bboxes.resize(N);
    char img_path[256];
    double bbox[4];
    vector<double> x(landmark_n), y(landmark_n);
    for (int i = 0; i < N; i++) {
        fscanf(fd, "%s", img_path);
        for (int j = 0; j < 4; j++) {
            fscanf(fd, "%lf", &bbox[j]);
        }
        for (int j = 0; j < landmark_n; j++) {
            fscanf(fd, "%lf%lf", &x[j], &y[j]);
        }
        Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
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
        bboxes[i] = BBox(bbox[0] - x_, bbox[1] - y_, bbox[2], bbox[3]);
        gt_shapes[i] = Mat::zeros(landmark_n, 2, CV_64FC1);
        for (int j = 0; j < landmark_n; j++) {
            gt_shapes[i].at<double>(j, 0) = x[j] - x_;
            gt_shapes[i].at<double>(j, 1) = y[j] - y_;
        }
        Rect roi(x_, y_, w_, h_);
        imgs[i] = img(roi).clone();
    }
    fclose(fd);
}

int train(int argc, char* argv[], int start_from) {
    if(argc != 2) {
        printf("train train.txt model_file\n");
        return -1;
    }
    Config &config = Config::GetInstance();
    LOG("Load train data from %s", argv[0]);
    vector<Mat> imgs_, gt_shapes_;
    vector<BBox> bboxes_;
    parseTxt(argv[0], imgs_, gt_shapes_, bboxes_);

    Mat mean_shape = getMeanShape(gt_shapes_, bboxes_);

    int N = imgs_.size();
    int L = N*config.initShape_n;
    vector<Mat> imgs(L), gt_shapes(L), current_shapes(L);
    vector<BBox> bboxes(L);
    RNG rng(getTickCount());
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < config.initShape_n; j++) {
            int idx = i*config.initShape_n + j;
            int k = 0;
            do {
                //if (i < (N / 2)) k = rng.uniform(0, N / 2);
                //else k = rng.uniform(N / 2, N);
                k = rng.uniform(0, N);
            } while (k == i);
            imgs[idx] = imgs_[i];
            gt_shapes[idx] = gt_shapes_[i];
            bboxes[idx] = bboxes_[i];
            current_shapes[idx] = bboxes_[i].ReProject(bboxes_[k].Project(gt_shapes_[k]));
        }
    }
    // random shuffle
    std::srand(std::time(0));
    std::random_shuffle(imgs.begin(), imgs.end());
    std::srand(std::time(0));
    std::random_shuffle(gt_shapes.begin(), gt_shapes.end());
    std::srand(std::time(0));
    std::random_shuffle(bboxes.begin(), bboxes.end());
    std::srand(std::time(0));
    std::random_shuffle(current_shapes.begin(), current_shapes.end());

    LbfCascador lbf_cascador;
    lbf_cascador.Init(config.stages_n);
    if (start_from > 0) {
        lbf_cascador.ResumeTrainModel(start_from);
    }
    // Train
    TIMER_BEGIN
        lbf_cascador.Train(imgs, gt_shapes, current_shapes, bboxes, mean_shape, start_from);
        LOG("Train Model Down, cost %.4lf s", TIMER_NOW);
    TIMER_END

    // Save
    FILE *fd = fopen(argv[1], "wb");
    assert(fd);
    lbf_cascador.Write(fd);
    fclose(fd);

    return 0;
}
