#include <cstdio>
#include "common.hpp"

using namespace std;
using namespace lbf;

int prepare(int argc, char* argv[]);
int train(int argc, char* argv[], int start_from);
int test(int argc, char* argv[]);
int run(int argc, char* argv[]);
int live(int argc, char *argv[]);
int camera(int argc, char *argv[]);

static void usage(const char* cmd) {
    printf("Usage: %s prepare/train/test/run/live/camera\n", cmd);
    exit(0);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage(argv[0]);
    } else if (strcmp(argv[1], "prepare") == 0) {
        return prepare(argc-2, argv+2);
    } else if (strcmp(argv[1], "train") == 0) {
        return train(argc-2, argv+2, 0);
    } else if (strcmp(argv[1], "test") == 0) {
        return test(argc-2, argv+2);
    } else if (strcmp(argv[1], "run") == 0) {
        return run(argc-2, argv+2);
    } else if (strcmp(argv[1], "live") == 0) {
        return live(argc-2, argv+2);
    } else if (strcmp(argv[1], "camera") == 0) {
        return camera(argc-2, argv+2);
    } else {
        printf("Unsupport command %s\n", argv[1]);
        usage(argv[0]);
    }
    return 0;
}
