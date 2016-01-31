#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <FaceTracker/Tracker.h>
#include <iostream>
#include <ExpressionClassifier.h>
#include <math.h>

using namespace cv;
using namespace std;
#define GRAY CV_RGB(128,128,128)
#define WHITE CV_RGB(255,255,255)
#define BLACK CV_RGB(0,0,0)
#define RED CV_RGB(255,0,0)

void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
    int i,n = shape.rows/2; cv::Point p1,p2;
    //draw triangulation
    for(i = 0; i < tri.rows; i++){
        if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
            visi.at<int>(tri.at<int>(i,1),0) == 0 ||
            visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
            p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
                shape.at<double>(tri.at<int>(i,0)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
            shape.at<double>(tri.at<int>(i,1)+n,0));
        cv::line(image,p1,p2,BLACK);
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
            shape.at<double>(tri.at<int>(i,0)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
            shape.at<double>(tri.at<int>(i,2)+n,0));
        cv::line(image,p1,p2,BLACK);
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
            shape.at<double>(tri.at<int>(i,2)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
            shape.at<double>(tri.at<int>(i,1)+n,0));
        cv::line(image,p1,p2,BLACK);
    }
    //draw connections
    for(i = 0; i < con.cols; i++){
        if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
            visi.at<int>(con.at<int>(1,i),0) == 0)continue;
        p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
            shape.at<double>(con.at<int>(0,i)+n,0));
        p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
            shape.at<double>(con.at<int>(1,i)+n,0));
        cv::line(image,p1,p2,BLACK,1);
    }
    //draw points
    for(i = 0; i < n; i++){    
        if(visi.at<int>(i,0) == 0)continue;
        p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
        cv::circle(image,p1,2,RED);
    }return;
}
int main(){
    //settings
    bool fcheck = true; double scale = 1; int fpd = -1; bool show = true;

    //set other tracking parameters
    std::vector<int> wSize1(1); wSize1[0] = 7;
    std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
    int nIter = 5; double clamp=3,fTol=0.01; 

    FACETRACKER::Tracker model("face2.tracker");
    cv::Mat tri=FACETRACKER::IO::LoadTri("face.tri");
    cv::Mat con=FACETRACKER::IO::LoadCon("face.con");

    //initialize camera and display window
    cv::Mat frame,gray,im; double fps=0; char sss[256]; std::string text; 
    VideoCapture camera(0);
    camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    int width = camera.get(CV_CAP_PROP_FRAME_WIDTH), height = camera.get(CV_CAP_PROP_FRAME_HEIGHT);
    printf("width: %d, height: %d\n", width, height);

    int64 t1,t0 = cvGetTickCount(); int fnum=0;
    if(show)
        cvNamedWindow("Smale Detect",1);

    //load expressions
    ExpressionClassifier classifier;
    classifier.load();

    bool failed = true;
    cout<<endl;

    int facenum = 0;

    while(1){
        camera.read(frame);

        //grab image, resize and flip
        if(scale == 1)im = frame;
        else cv::resize(frame,im,cv::Size(scale*frame.cols,scale*frame.rows));
        cv::flip(im,im,1);
        cv::cvtColor(im,gray,CV_BGR2GRAY);

        //track this image
        std::vector<int> wSize; if(failed)wSize = wSize2; else wSize = wSize1; 
        if(model.Track(gray,wSize,fpd,nIter,clamp,fTol,fcheck) == 0){
            int idx = model._clm.GetViewIdx();
            failed = false;
            Draw(im,model._shape,con,tri,model._clm._visi[idx]);
        }else{
            if(show){
                cv::Mat R(im,cvRect(0,0,150,50));
                R = cv::Scalar(0,0,255);
            }
            model.FrameReset();
            failed = true;
        }

        //draw framerate on display image 
        if(fnum >= 9){      
            t1 = cvGetTickCount();
            fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6); 
            t0 = t1; fnum = 0;
        }else fnum += 1;
        
        if(show){
            sprintf(sss,"%d frames/sec",(int)round(fps)); text = sss;
            cv::putText(im,text,cv::Point(10,20),
            CV_FONT_HERSHEY_SIMPLEX,0.5,RED);
        }
        
        //classify the points
        const Mat& mean = model._clm._pdm._M;
        const Mat& variation = model._clm._pdm._V;
        const Mat& weights = model._clm._plocal;
        Mat objectPoints = mean + variation * weights;

        classifier.classify(objectPoints);

        int n = classifier.size();
        for(int i = 0; i < n; i++){
            sprintf(sss,"%s: %f", classifier.getDescription(i).c_str(), classifier.getProbability(i));
            cv::putText(im,sss,cv::Point(10, 40 + i*20),
            CV_FONT_HERSHEY_SIMPLEX,0.5,RED);
        }
        // printf("smile: %.4f\n", classifier.getProbability(1) );

        if(show){
            //show image and check for user input
            imshow("Smale Detect",im); 
            int c = waitKey(10)&0xFF;
            if(c == 27){
                cout<<"Esc"<<endl;
                break;
            }
            else if( char(c) == ' '){
                model.FrameReset();
                cout<<"Redetect"<<endl;
            }else if( char(c) == 'c'){
                //save face model
                facenum ++;
                char facefilename[50];
                sprintf(facefilename, "face/face%d.yml", facenum);
                FileStorage fs(facefilename, FileStorage::WRITE);
                fs << "shape" << model._shape;
                fs << "tri" << tri;
                fs.release();

                FileStorage fs2("face/objectPoints.yml", FileStorage::WRITE);
                fs2 <<   "description" << "emotion" <<
                "samples" << "[";
                fs2 << objectPoints;
                fs2 << "]";
                fs2.release();
            }else if( char(c) == 'v'){
                Mat shape = model._shape;
                double scale = 5;
                Mat test(480*scale, 640*scale, CV_8UC3, Scalar(255,255,255));
                int i, n = 66; cv::Point p1,p2;
                //draw triangulation
                for(i = 0; i < tri.rows; i++){
                    p1 = cv::Point(scale*shape.at<double>(tri.at<int>(i,0),0),
                        scale*shape.at<double>(tri.at<int>(i,0)+n,0));
                    p2 = cv::Point(scale*shape.at<double>(tri.at<int>(i,1),0),
                        scale*shape.at<double>(tri.at<int>(i,1)+n,0));
                    cv::line(test,p1,p2,BLACK);
                    p1 = cv::Point(scale*shape.at<double>(tri.at<int>(i,0),0),
                        scale*shape.at<double>(tri.at<int>(i,0)+n,0));
                    p2 = cv::Point(scale*shape.at<double>(tri.at<int>(i,2),0),
                        scale*shape.at<double>(tri.at<int>(i,2)+n,0));
                    cv::line(test,p1,p2,BLACK);
                    p1 = cv::Point(scale*shape.at<double>(tri.at<int>(i,2),0),
                        scale*shape.at<double>(tri.at<int>(i,2)+n,0));
                    p2 = cv::Point(scale*shape.at<double>(tri.at<int>(i,1),0),
                        scale*shape.at<double>(tri.at<int>(i,1)+n,0));
                    cv::line(test,p1,p2,BLACK);
                }
                //draw points
                for(i = 0; i < n; i++){    
                    p1 = cv::Point(scale*shape.at<double>(i,0),scale*shape.at<double>(i+n,0));
                    char buf[50];
                    sprintf(buf, "%d", i);
                    cv::circle(test,p1,scale*2, RED);
                    p1.x += 10;
                    cv::putText(test, buf, p1, CV_FONT_HERSHEY_SIMPLEX, 0.7, RED);
                }
                imwrite("face/test.png", test);
            }
        }
        // Point p1,p2;
        // Mat shape = model._shape;
        // p1 = cv::Point(shape.at<double>(23,0),
        //     shape.at<double>(23+66,0));
        // p2 = cv::Point(shape.at<double>(43,0),
        //     shape.at<double>(43+66,0));
        // double distance = sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
        // cout<<distance-34<<endl;

    }
    return 0;
}
