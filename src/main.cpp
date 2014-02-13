#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "SvmLightLib.h"

#define TRAINING_SET_PATH "/Users/alberto/tmp/samples/"

using namespace cv;
using namespace std;

void train()
{
    clock_t start_clock = clock();
    time_t start_time = time(0);

    SVMLight::SVMTrainer svm("features.dat");

    HOGDescriptor hog;
    hog.winSize = Size(50,50);

    size_t posCount = 0, negCount = 0;
    for (size_t i = 1; i <= 1800; ++i)
    {
        // in this concrete case I had files 0001.JPG to 0800.JPG in both "positive" and "negative" subfolders:
        std::ostringstream os;
        os << TRAINING_SET_PATH << "positive/" << std::setw(4) << std::setfill('0') << i << ".png";

        Mat img = imread(os.str(),CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data) {
            std::cout << "problems.. " << os.str() << std::endl;
            continue;
        } else {
            std::cout << "read: " << os.str() << std::endl;
        }

        // obtain feature vector:
        vector<float> featureVector;
        hog.compute(img, featureVector, Size(8, 8), Size(0, 0));

        // write feature vector to file that will be used for training:
        svm.writeFeatureVectorToFile(featureVector, true);                  // true = positive sample
        posCount++;

        // clean up:
        featureVector.clear();
        img.release();              // we don't need the original image anymore
        os.clear(); os.seekp(0);    // reset string stream

        // do the same for negative sample:
        os << TRAINING_SET_PATH << "negative/" << std::setw(4) << std::setfill('0') << i << ".png";
        img = imread(os.str(),CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data) {
            std::cout << "problems.. " << os.str() << std::endl;
            continue;
        } else {
            std::cout << "read: " << os.str() << std::endl;
        }

        hog.compute(img, featureVector, Size(8, 8), Size(0, 0));
        svm.writeFeatureVectorToFile(featureVector, false);
        negCount++;
        img.release();
    }

    std::cout   << "finished writing features: "
                << posCount << " positive and "
                << negCount << " negative samples used";
    std::string modelName("classifier.dat");
    svm.trainAndSaveModel(modelName);
    std::cout   << "SVM saved to " << modelName << std::endl;

    clock_t end_clock = clock();
    double time_clock = (double) (end_clock-start_clock) / CLOCKS_PER_SEC * 1000.0;
    time_t end_time = time(0);
    double time_time = difftime(end_time, start_time) * 1000.0;
    std::cout << "everything has been done in: " << std::endl
              << "    " << time_clock << " CPU clock" << std::endl
              << "    " << time_time << " milliseconds" << std::endl
              << std::endl;
}

int main()
{

    train();

    HOGDescriptor hog;
    hog.winSize = Size(50,50);
    SVMLight::SVMClassifier c("classifier.dat");
    vector<float> descriptorVector = c.getDescriptorVector();
    std::cout << descriptorVector.size() << std::endl;
    hog.setSVMDetector(descriptorVector);

//    Mat m = imread("/Users/alberto/tmp/samples/fullframe.png");

//    vector<Rect> found;
//    Size padding(Size(0, 0));
//    Size winStride(Size(8, 8));
//    hog.detectMultiScale(m, found, 0.0, winStride, padding, 1.01, 0.1);

//    std::cout << "found: " << found.size() << std::endl;


    return 0;
}

