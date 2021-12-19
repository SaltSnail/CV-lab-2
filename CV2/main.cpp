#include <iostream>
#include <opencv2/opencv.hpp>
#include "Header.h"

using namespace std;

void main()
{
    string image_path = "balls.jpg";
    cv::Mat source_image = cv::imread(image_path);
    cout << "Initial params: " << source_image.cols << ", " << source_image.rows << ", " << source_image.channels() << endl;

    auto filters = getDistribution(0, 1);
    cv::Mat l1 = Convolution(source_image, 1, filters);
    cout << "After convolution: " << l1.cols << ", " << l1.rows << ", " << l1.channels() << endl;
    cv::Mat l2 = Normalize(l1, 1, 1);
    cout << "After norm: " << l2.cols << ", " << l2.rows << ", " << l2.channels() << endl;
    cv::Mat l3 = Relu(l2);
    cout << "After ReLU: " << l3.cols << ", " << l3.rows << ", " << l3.channels() << endl;
    cv::Mat l4 = MaxPooling(l3, 2, 2);
    cout << "After max pooling: " << l4.cols << ", " << l4.rows << ", " << l4.channels() << endl;
    cv::Mat l5 = Softmax(l4);

    cv::imshow("Image", source_image);
    cv::waitKey(0);

    system("pause");
}

