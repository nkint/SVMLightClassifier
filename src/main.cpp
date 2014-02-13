#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SvmLightLib.h"

#define TRAINING_SET_PATH "/Users/alberto/tmp/samples/"
#define FILE_VERBOSE false

using namespace cv;

const Size sampleSize(48,48);
const Size winStride(12,12);

SVMLight::SVMTrainer svm("features.dat");
HOGDescriptor hog;
std::ostringstream os;
vector<float> featureVector;
size_t posCount = 0, negCount = 0;

void cleanup() {
    featureVector.clear();
    os.clear(); os.seekp(0);    // reset string stream
    os.str("");
}

void computeHog(Mat img, bool flag)
{
    hog.compute(img, featureVector, winStride, Size(0, 0));
    svm.writeFeatureVectorToFile(featureVector, (flag?true:false) );
    (flag ? posCount++ : negCount++);
}

void handleSample(size_t i, bool flag)
{
    Mat img, img_original;
    // do the same for negative sample:
    os << TRAINING_SET_PATH << (flag?"positive/":"negative/") << std::setw(4) << std::setfill('0') << i << ".png";
    img_original = imread(os.str(),CV_LOAD_IMAGE_GRAYSCALE);
    if (!img_original.data) {
        if(FILE_VERBOSE)std::cout << "problems.. " << os.str() << std::endl;
        cleanup();
        return;
    } else {
        if(FILE_VERBOSE)std::cout << "read: " << os.str() << std::endl;
    }

    img = img_original(Rect(0,0,48,48));
    computeHog(img, flag);

    flip(img, img, 0);
    computeHog(img, flag);

    img.release();
    img_original.release();
    cleanup();
}

void train()
{
    hog.winSize = Size(sampleSize);

    for (size_t i = 1; i <= 8000; ++i)
    {
        handleSample(i, true);
        handleSample(i, false);
    }

    std::cout   << "finished writing features: "
                << posCount << " positive and "
                << negCount << " negative samples used"
                << std::endl;

    std::string modelName("classifier.dat");
    svm.trainAndSaveModel(modelName);
    std::cout   << "SVM saved to " << modelName << std::endl;
}

void classify()
{
    HOGDescriptor hog;
    hog.winSize = Size(sampleSize);
    SVMLight::SVMClassifier c("classifier.dat");
    vector<float> descriptorVector = c.getDescriptorVector();
    std::cout << descriptorVector.size() << std::endl;
    hog.setSVMDetector(descriptorVector);

    Mat m = imread("/Users/alberto/tmp/samples/fullframe.png");
    Mat m1 = m.clone();

    vector<Rect> found;
    vector<Point> foundPoint;
    Size padding(Size(0, 0));

    std::cout << "try to detect.." << std::endl;
    //hog.detectMultiScale(m, found, 0.0, winStride, padding, 1.01, 0.1);
    hog.detect(m, foundPoint, 0.0, winStride, padding);
    std::cout << "found: " << foundPoint.size() << std::endl;


    for(int i=0; i<foundPoint.size(); ++i) {
        Rect r;
        r.x = foundPoint[i].x;
        r.y = foundPoint[i].y;
        r.width = 48;
        r.height = 48;
        rectangle(m, r, Scalar(255,255,255));




        Mat imageroi = m1(r);
        std::stringstream ss;
        ss << "/Users/alberto/tmp/samples/tmp/test";
        ss << i;
        ss << ".png";
        cv::imwrite(ss.str(), imageroi);

    }

    imshow("result", m);
}

int main()
{
    clock_t start_clock, end_clock;
    time_t start_time, end_time;
    double time_clock, time_time;

    std::cout << std::endl
              << "-------------------- train"
              << std::endl;
    start_clock = clock();
    start_time = time(0);
    train();
    end_clock = clock();
    time_clock = (double) (end_clock-start_clock) / CLOCKS_PER_SEC * 1000.0;
    end_time = time(0);
    time_time = difftime(end_time, start_time) * 1000.0;
    std::cout << "done in: " << std::endl
              << "    " << time_clock << " CPU clock" << std::endl
              << "    " << time_time << " in time" << std::endl
              << std::endl;


    std::cout << std::endl
              << "-------------------- classify"
              << std::endl;
    start_clock = clock();
    start_time = time(0);
    classify();
    end_clock = clock();
    time_clock = (double) (end_clock-start_clock) / CLOCKS_PER_SEC * 1000.0;
    end_time = time(0);
    time_time = difftime(end_time, start_time) * 1000.0;
    std::cout << "done in: " << std::endl
              << "    " << time_clock << " CPU clock" << std::endl
              << "    " << time_time << " in time" << std::endl
              << std::endl;

    waitKey();

    return 0;
}

