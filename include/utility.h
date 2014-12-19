#ifndef UTILITY_H
#define UTILITY_H

#include "opencv2/highgui/highgui.hpp"

#include <string>

// Shows the image and waits for input before destroying the window.
void show_image(const cv::Mat& img);

// Loads the image as grayscale and thresholds it such that all pixels less
// than or equal to 128 are set to 0 and pixels greater than 128 are set to
// white.
cv::Mat load_binary_image(const std::string& filename);

// Adds salt-and-pepper noise to the specified image with the specified black
// probability and white probability.
//
// Both white and black probabilities are integers between 0 and 100
// (inclusive) which represent percetanges.
cv::Mat add_salt_and_pepper_noise(const cv::Mat& img, 
                                  int black_proba,
                                  int white_proba);

// Makes a comparison image as a composite of the three input images. The first
// image is placed to the left, the second in the middle, and the third, to the
// right.
cv::Mat make_comparison(const cv::Mat& img1, 
                        const cv::Mat& img2, 
                        const cv::Mat& img3);


#endif
