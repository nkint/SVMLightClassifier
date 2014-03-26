#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <time.h>


#define _DEBUG true

const cv::Size sampleSize(48,48);
const cv::Size winStride(8,8);


using namespace cv;
using namespace std;

void get_svm_detector(const SVM& svm, vector< float > & hog_detector );
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData );
void load_images( const string & prefix, const string & filename, vector< Mat > & img_lst );
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size );
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size );
void compute_hog( const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size );
void train_svm( const vector< Mat > & gradient_lst, const vector< int > & labels );
void draw_locations( Mat & img, const vector< Rect > & locations, const Scalar & color );
void test_it( const Size & size );

void get_svm_detector(const SVM& svm, vector< float > & hog_detector )
{
    // get the number of variables
    const int var_all = svm.get_var_count();
    // get the number of support vectors
    const int sv_total = svm.get_support_vector_count();
    // get the decision function
    const CvSVMDecisionFunc* decision_func = svm.get_decision_function();
    // get the support vectors
    const float** sv = new const float*[ sv_total ];
    for( int i = 0 ; i < sv_total ; ++i )
        sv[ i ] = svm.get_support_vector(i);

    CV_Assert( var_all > 0 &&
        sv_total > 0 &&
        decision_func != 0 &&
        decision_func->alpha != 0 &&
        decision_func->sv_count == sv_total );

    float svi = 0.f;

    hog_detector.clear(); //clear stuff in vector.
    hog_detector.reserve( var_all + 1 ); //reserve place for memory efficiency.

     /**
    * hog_detector^i = \sum_j support_vector_j^i * \alpha_j
    * hog_detector^dim = -\rho
    */
   for( int i = 0 ; i < var_all ; ++i )
    {
        svi = 0.f;
        for( int j = 0 ; j < sv_total ; ++j )
        {
            if( decision_func->sv_index != NULL ) // sometime the sv_index isn't store on YML/XML.
                svi += (float)( sv[decision_func->sv_index[j]][i] * decision_func->alpha[ j ] );
            else
                svi += (float)( sv[j][i] * decision_func->alpha[ j ] );
        }
        hog_detector.push_back( svi );
    }
    hog_detector.push_back( (float)-decision_func->rho );

    delete[] sv;
}


/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1 );
    vector< Mat >::const_iterator itr = train_samples.begin();
    vector< Mat >::const_iterator end = train_samples.end();
    for( int i = 0 ; itr != end ; ++itr, ++i )
    {
        CV_Assert( itr->cols == 1 ||
            itr->rows == 1 );
        if( itr->cols == 1 )
        {
            transpose( *(itr), tmp );
            tmp.copyTo( trainData.row( i ) );
        }
        else if( itr->rows == 1 )
        {
            itr->copyTo( trainData.row( i ) );
        }
    }
}

void load_images( const string & prefix, const string & filename, vector< Mat > & img_lst )
{
    string line;
    ifstream file;

    file.open( (prefix+filename).c_str() );
    if( !file.is_open() )
    {
        cerr << "Unable to open the list of images from " << (prefix+filename) << " filename." << endl;
        exit( -1 );
    }

    bool end_of_parsing = false;
    while( !end_of_parsing )
    {
        getline( file, line );
        if( line == "" ) // no more file to read
        {
            end_of_parsing = true;
            break;
        }
        std::string s = (prefix+line);
        //std::cout << "try to open: " << s << std::endl;
        Mat img = imread( s.c_str() ); // load the image
        if( !img.data ) // invalid image, just skip it.
            continue;
#ifdef _DEBUG
        imshow( "image", img );
        waitKey( 10 );
#endif

        if(img.size().width==50) {
            Rect r = Rect(0,0,48,48);
            img = img(r);
        } else {

        }
        img_lst.push_back( img.clone() );

//        Mat flipped;
//        flip(img, flipped, 1);
//        img_lst.push_back( flipped.clone() );
//        flip(img, flipped, -1);
//        img_lst.push_back( flipped.clone() );

    }
}

void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size )
{
    Rect box;
    box.width = size.width;
    box.height = size.height;

    const int size_x = box.width;
    const int size_y = box.height;

    srand( (unsigned int)time( NULL ) );

    vector< Mat >::const_iterator img = full_neg_lst.begin();
    vector< Mat >::const_iterator end = full_neg_lst.end();
    int i=0;
    for( ; img != end ; ++img )
    {
        box.x = 0; //rand() % (img->cols - size_x);
        box.y = 0; //rand() % (img->rows - size_y);
        Mat roi = (*img)(box);
        neg_lst.push_back( roi.clone() );
#ifdef _DEBUG
        imshow( "img", roi.clone() );
        waitKey( 10 );
#endif
    }
}

// From http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size )
{
    const int DIMX = size.width;
    const int DIMY = size.height;
    float zoomFac = 3;
    Mat visu;
    resize(color_origImg, visu, Size( (int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac) ) );

    int cellSize        = winStride.width;
    int gradientBinSize = 9;
    float radRangeForOneBin = (float)(CV_PI/(float)gradientBinSize); // dividing 180° into 9 bins, how large (in rad) is one bin?

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = DIMX / cellSize;
    int cells_in_y_dir = DIMY / cellSize;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;

            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }

    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;

    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;

    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                cellx = blockx;
                celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }

                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;

                    gradientStrengths[celly][cellx][bin] += gradientStrength;

                } // for (all bins)


                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;

            } // for (all cells)


        } // for (all block x pos)
    } // for (all block y pos)


    // compute average gradient strengths
    for (celly=0; celly<cells_in_y_dir; celly++)
    {
        for (cellx=0; cellx<cells_in_x_dir; cellx++)
        {

            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }

    // draw cells
    for (celly=0; celly<cells_in_y_dir; celly++)
    {
        for (cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize;
            int drawY = celly * cellSize;

            int mx = drawX + cellSize/2;
            int my = drawY + cellSize/2;

            rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX+cellSize)*zoomFac), (int)((drawY+cellSize)*zoomFac)), CV_RGB(100,100,100), 1);

            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];

                // no line to draw?
                if (currentGradStrength==0)
                    continue;

                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;

                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = (float)(cellSize/2.f);
                float scale = 2.5; // just a visualization scale, to see the lines better

                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                // draw gradient visualization
                line(visu, Point((int)(x1*zoomFac),(int)(y1*zoomFac)), Point((int)(x2*zoomFac),(int)(y2*zoomFac)), CV_RGB(0,255,0), 1);

            } // for (all bins)

        } // for (cellx)
    } // for (celly)


    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
        for (int x=0; x<cells_in_x_dir; x++)
        {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;

    return visu;

} // get_hogdescriptor_visu

void compute_hog( const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size )
{
    HOGDescriptor hog;
    hog.winSize = size;
    Mat gray;
    vector< Point > location;
    vector< float > descriptors;

    vector< Mat >::const_iterator img = img_lst.begin();
    vector< Mat >::const_iterator end = img_lst.end();
    for( ; img != end ; ++img )
    {
        cvtColor( *img, gray, COLOR_BGR2GRAY );
        hog.compute( gray, descriptors, winStride, Size( 0, 0 ), location );
        gradient_lst.push_back( Mat( descriptors ).clone() );
#ifdef _DEBUG
        imshow( "gradient", get_hogdescriptor_visu( img->clone(), descriptors, size ) );
        waitKey( 10 );
#endif
    }
}

void train_svm( const vector< Mat > & gradient_lst, const vector< int > & labels )
{
    SVM svm;

    /* Default values to train SVM */
    SVMParams params;
    params.coef0 = 0.0;
    params.degree = 3;
    params.term_crit.epsilon = 1e-3;
    params.gamma = 0;
    params.kernel_type = SVM::LINEAR;
    params.nu = 0.5;
    params.p = 0.1; // for EPSILON_SVR, epsilon in loss function?
    params.C = 0.01; // From paper, soft classifier
    params.svm_type = SVM::EPS_SVR; // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task

    Mat train_data;
    convert_to_ml( gradient_lst, train_data );

    clog << "Start training...";
    svm.train( train_data, Mat( labels ), Mat(), Mat(), params );
    clog << "...[done]" << endl;

    svm.save( "my_people_detector.yml" );
}

void draw_locations( Mat & img, const vector< Rect > & locations, const Scalar & color )
{
    if( !locations.empty() )
    {
        vector< Rect >::const_iterator loc = locations.begin();
        vector< Rect >::const_iterator end = locations.end();
        for( ; loc != end ; ++loc )
        {
            rectangle( img, *loc, color, 2 );
        }
    }
}

void filter_rects(const std::vector<cv::Rect>& candidates, std::vector<cv::Rect>& objects)
{
    size_t i, j;
    for (i = 0; i < candidates.size(); ++i)
    {
        cv::Rect r = candidates[i];

        for (j = 0; j < candidates.size(); ++j)
            if (j != i && (r & candidates[j]) == r)
                break;

        if (j == candidates.size())
            objects.push_back(r);
    }
}

void test_it( const Size & size )
{
    char key = 27;
    Scalar reference( 0, 255, 0 );
    Scalar trained( 0, 0, 255 );
    Mat img, draw;
    SVM svm;
    HOGDescriptor hog;
    hog.winSize = size;
    vector< Rect > locations;

    // Load the trained SVM.
    svm.load( "my_people_detector.yml" );
    // Set the trained svm to my_hog
    vector< float > hog_detector;
    get_svm_detector( svm, hog_detector );
    hog.setSVMDetector( hog_detector );
    // Set the people detector.


    Mat m = imread("/Users/alberto/tmp/samples/fullframe10.png");
    Mat m1 = m.clone();

    std::vector<Rect> found, found_filtered;
    std::vector<Point> foundPoint;
    Size padding(Size(0, 0));

    std::cout << "try to detect.." << std::endl;
    //hog.detectMultiScale(m, found, 0.0, winStride, padding, 1.01, 0.1);
    hog.detect(m, foundPoint, 0.0, winStride, padding);
    std::cout << "found: " << foundPoint.size() << std::endl;

    hog.detectMultiScale( m, locations );
    std::cout << "found multi scale: " << locations.size() << std::endl;

    for(int i=0; i<locations.size(); ++i) {
        Rect &r = locations[i];
        rectangle(m, r, Scalar(0,0,255));
        found.push_back(r);
    }

    for(int i=0; i<foundPoint.size(); ++i) {
        Rect r;
        r.x = foundPoint[i].x;
        r.y = foundPoint[i].y;
        r.width = sampleSize.width;
        r.height = sampleSize.height;
        rectangle(m, r, Scalar(255,255,255));

        // save for hard learning
        Mat imageroi = m1(r);
        std::stringstream ss;
        ss << "/Users/alberto/tmp/samples/tmp/test";
        ss << i;
        ss << ".png";
        cv::imwrite(ss.str(), imageroi);

        found.push_back(r);
    }

    filter_rects(found, found_filtered);
    for(int i=0; i<found_filtered.size(); ++i) {
        Rect &r = found_filtered[i];
        rectangle(m, r, Scalar(255,0,0));
    }

    cv::pyrDown(m, m1);

    imshow("result", m1);


    waitKey();
}

int main( int argc, char** argv )
{

    bool only_test = true;
    if(!only_test) {
        vector< Mat > pos_lst;
        vector< Mat > full_neg_lst;
        vector< Mat > neg_lst;
        vector< Mat > gradient_lst;
        vector< int > labels;

        const std::string path = "/Users/alberto/tmp/samples/";
        const std::string poslist = "positive.lst";
        const std::string neglist = "negative.lst";

        std::cout << "load positives" << std::endl;
        load_images( path, poslist, pos_lst );
        labels.assign( pos_lst.size(), +1 );
        const unsigned int old = (unsigned int)labels.size();

        std::cout << "load negatives" << std::endl;
        load_images( path, neglist, full_neg_lst );
    std::cout << "dudee" << std::endl;
        sample_neg( full_neg_lst, neg_lst, sampleSize );

        labels.insert( labels.end(), neg_lst.size(), -1 );
        CV_Assert( old < labels.size() );

        std::cout << "compute hogs" << std::endl;
        compute_hog( pos_lst, gradient_lst, sampleSize );
        compute_hog( neg_lst, gradient_lst, sampleSize );

        std::cout << "train" << std::endl;
        train_svm( gradient_lst, labels );
    }

    std::cout << "test" << std::endl;
    test_it( sampleSize ); // change with your parameters

    return 0;
}
