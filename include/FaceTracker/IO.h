#ifndef IO_h_
#define IO_h_
#include <opencv/cv.h>
#include <fstream>
namespace FACETRACKER
{
  //===========================================================================
  /** 
      Input-output Operations
  */
  class IO{
  public:
    enum{PDM = 0,PAW,PATCH,MPATCH,CLM,FDET,FCHECK,MFCHECK,TRACKER};
    static void ReadMat(std::ifstream& s,cv::Mat &M);
    static void WriteMat(std::ofstream& s,cv::Mat &M);
    static cv::Mat LoadCon(const char* fname);
    static cv::Mat LoadTri(const char* fname);
  };
  //===========================================================================
}
#endif
