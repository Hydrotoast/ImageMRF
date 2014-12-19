#include "utility.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <sstream>

using namespace cv;
using namespace std;

typedef double Prob;
typedef uchar Label;
typedef map<Label, Prob> LabelDist;

typedef pair<size_t, size_t> Coord;
typedef pair<Coord, Coord> MessageParams;

Label LABELS[] = {0, 255};
map<MessageParams, LabelDist> MESSAGES;

// Computes the sum of squared differences of two matrices.
Prob sum_square_diff(const Mat& A, const Mat& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);

  Prob ssd = 0;

  int rows = A.rows;
  int cols = A.cols;

  const Label *a, *b;
  for (int i = 0; i < rows; ++i) {
    a = A.ptr<Label>(i);
    b = B.ptr<Label>(i);
    for (int j = 0; j < cols; ++j) {
      Prob diff = *a - *b;
      ssd += diff * diff;
    }
  }

  return ssd;
}

// Returns the neighboring coordinates of xi except the exlusion coordinate in
// the image with respect to the boundaries of the image.
vector<Coord> neighborhood(const Mat& img, 
                           const Coord& xi, 
                           const Coord& exclusion) {

  static vector<int> drow{-1, 0, 1, 0};
  static vector<int> dcol{0, 1, 0, -1};

  int rows = img.rows;
  int cols = img.cols;
  vector<Coord> neighbors;

  int row;
  int col;
  for (int i = 0; i < 4; i++) {
    row = xi.first + drow[i];
    col = xi.second + dcol[i];
    Coord xk = make_pair(row, col);

    if (row < 0 || col < 0) {
      continue;
    } else if (row >= rows || col >= cols) {
      continue;
    } else if (xk != exclusion) {
      continue;
    }

    neighbors.push_back(xk);
  }
  
  return neighbors;
}

// Returns the neighboring coordinates of xi in the image with respect to the
// boundaries of the image.
vector<Coord> neighborhood(const Mat& img, 
                           const Coord& xi) {

  assert(xi.first >= 0);
  assert(xi.first < img.rows);
  assert(xi.second >= 0);
  assert(xi.second < img.cols);

  static vector<int> drow{-1, 0, 1, 0};
  static vector<int> dcol{0, 1, 0, -1};

  int rows = img.rows;
  int cols = img.cols;
  vector<Coord> neighbors;

  int row;
  int col;
  for (int i = 0; i < 4; i++) {
    row = xi.first + drow[i];
    col = xi.second + dcol[i];
    Coord xk = make_pair(row, col);

    if (row < 0 || col < 0) {
      continue;
    } else if (row >= rows || col >= cols) {
      continue;
    }
    neighbors.push_back(xk);
  }
  
  return neighbors;
}

Prob unary_energy(const Label& xi_label, const Label& zi_label) {
  return xi_label == zi_label ? 3 : 3.5;
}

Prob binary_energy(const Label& xi_label, const Label& xj_label) {
  return xi_label == xj_label ? 0 : 1;
}

Prob h(const Mat& img, const Coord& xi, const Coord& xj, const Label& xi_label) {
  Prob term = unary_energy(xi_label, img.at<Label>(xi.first, xi.second));
  assert(term > 0);
  for (Coord xk : neighborhood(img, xi, xj)) {
    term += MESSAGES[make_pair(xk, xi)][xi_label];
  }
  return term;
}

Prob min_interaction_energy(Mat& img, 
                            const Coord& xi, 
                            const Coord& xj) {

  Prob min_proba = std::numeric_limits<Prob>::max();
  for (Label xi_label : LABELS) {
    Prob proba = h(img, xi, xj, xi_label) + 1;

    if (proba < min_proba) {
      min_proba = proba;
    }
  }

  return min_proba;
}

void send_message(Mat& img, const Coord& xi, const Coord& xj) {
  Prob min_energy = min_interaction_energy(img, xi, xj);
  assert(min_energy > 0);
  for (Label label : LABELS) {
    Prob equal_energy = h(img, xi, xj, label);
    assert(equal_energy > 0);
    MESSAGES[make_pair(xi, xj)][label] = std::min(min_energy, equal_energy);
  }
}

// Perform the believe step in belief propagation.
void believe(Mat& img, const Coord& xi) {
  Label best_label;
  Prob best_proba = numeric_limits<Prob>::max();
  for (Label xi_label : LABELS) {
    Prob proba = unary_energy(xi_label, img.at<Label>(xi.first, xi.second));

    for (Coord xk : neighborhood(img, xi)) {
      Prob message = MESSAGES[make_pair(xk, xi)][xi_label];
      assert(message > 0);
      proba += message;
    }
    if (proba < best_proba) {
      best_proba = proba;
      best_label = xi_label;
    }
  }
  img.at<Label>(xi.first, xi.second) = best_label;
}

// Denoises the image with the specified number of iterations using loopy
// belief propagation.
Mat denoise(const Mat& img, int iterations) {
  Mat X = img.clone();
  int rows = X.rows;
  int cols = X.cols;

  Mat X_prev;
  for (int i = 0; i < iterations; ++i) {
    X_prev = X.clone();
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols - 1; ++col) {
        Coord xi = make_pair(row, col);
        Coord xj = make_pair(row, col + 1);
        send_message(X, xi, xj);
      }
    }

    for (int row = 0; row < rows; ++row) {
      for (int col = cols - 1; col > 0; --col) {
        Coord xi = make_pair(row, col);
        Coord xj = make_pair(row, col - 1);
        send_message(X, xi, xj);
      }
    }

    for (int col = 0; col < cols; ++col) {
      for (int row = rows - 1; row > 0; --row) {
        Coord xi = make_pair(row, col);
        Coord xj = make_pair(row - 1, col);
        send_message(X, xi, xj);
      }
    }

    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < rows - 1; ++row) {
        Coord xi = make_pair(row, col);
        Coord xj = make_pair(row + 1, col);
        send_message(X, xi, xj);
      }
    }

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        Coord xi = make_pair(row, col);
        believe(X, xi);
      }
    }

    cout << "Energy: " << sum_square_diff(X, X_prev) << endl;
  }

  return X;
}

int main(int argc, char *argv[]) {
  // Image parameter
  std::string filename = "lena.png";
  if (argc > 1) {
    filename = argv[1];
  }

  // Salt-and-pepper noise parameters
  int black_proba = 2;
  int white_proba = 2;
  if (argc > 3) {
    stringstream ss;
    ss << argv[2] << " " << argv[3];
    ss >> black_proba >> white_proba;
  }

  Mat img = load_binary_image(filename);
  Mat noisy_img = add_salt_and_pepper_noise(img, black_proba, white_proba);

  auto start = std::chrono::steady_clock::now();
  Mat denoised_img = denoise(noisy_img, 1);
  auto end = std::chrono::steady_clock::now();
  std::cout << "Time: "
            << chrono::duration_cast<chrono::milliseconds>(end - start).count() 
            << " ms"
            << std::endl;

  Mat comparison = make_comparison(img, noisy_img, denoised_img);

  stringstream ss;
  ss << "denoised_" << black_proba << "_" << white_proba << "_" << filename;
  imwrite(ss.str(), comparison);
  show_image(comparison);
}
