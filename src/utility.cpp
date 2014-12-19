#include "utility.h"

#include "opencv2/imgproc/imgproc.hpp"

#include <stdexcept>

using namespace cv;
using namespace std;

void show_image(const Mat& img) {
  namedWindow("MRF Window", CV_WINDOW_AUTOSIZE);
  imshow("MRF Window", img);

  waitKey(0);

  destroyWindow("MyWindow");
}

Mat load_binary_image(const string& filename) {
  Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  if (img.empty()) {
    throw runtime_error("Error: Image cannot be loaded!");
  }
  threshold(img, img, 128, 255, CV_THRESH_BINARY);
  return img;
}

Mat add_salt_and_pepper_noise(const Mat& img, 
                              int black_proba,
                              int white_proba) {

  assert(black_proba >= 0);
  assert(black_proba <= 100);
  assert(white_proba >= 0);
  assert(white_proba <= 100);

  Mat noise_distribution = Mat::zeros(img.rows, img.cols, CV_8U);
  randu(noise_distribution, 0, 100);

  Mat black_noise = noise_distribution < black_proba;
  Mat white_noise = noise_distribution >= (100 - white_proba);

  Mat noisy_img = img.clone();
  noisy_img.setTo(0, black_noise);
  noisy_img.setTo(255, white_noise);

  return noisy_img;
}

Mat make_comparison(const Mat& img1, const Mat& img2, const Mat& img3) {
  int rows = img1.rows;
  int cols = img1.cols;

  Mat triple = Mat::zeros(rows, cols * 3 + 2, CV_8U);
  img1.copyTo(triple(Rect(0, 0, cols, rows)));
  img2.copyTo(triple(Rect(cols + 1, 0, cols, rows)));
  img3.copyTo(triple(Rect(cols * 2 + 2, 0, cols, rows)));

  return triple;
}
