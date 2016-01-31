#ifndef FCheck_h_
#define FCheck_h_
#include <FaceTracker/PAW.h>
#include <vector>
namespace FACETRACKER
{
  //===========================================================================
  /** 
      Checks for Tracking Failure
  */
  class FCheck{
  public:    
    PAW     _paw; /**< Piecewise affine warp */
    double  _b;   /**< SVM bias              */
    cv::Mat _w;   /**< SVM gain              */

    FCheck(){;}
    FCheck(const char* fname){this->Load(fname);}
    FCheck(double b, cv::Mat &w, PAW &paw){this->Init(b,w,paw);}
    FCheck& operator=(FCheck const&rhs);
    void Init(double b, cv::Mat &w, PAW &paw);
    void Load(const char* fname);
    void Save(const char* fname);
    void Write(std::ofstream &s);
    void Read(std::ifstream &s,bool readType = true);
    bool Check(cv::Mat &im,cv::Mat &s);
    
  private:
    cv::Mat crop_,vec_;
  };
  //===========================================================================
  /** 
      Checks for Multiview Tracking Failure
  */
  class MFCheck{
  public:    
    std::vector<FCheck> _fcheck; /**< FCheck for each view */
    
    MFCheck(){;}
    MFCheck(const char* fname){this->Load(fname);}
    MFCheck(std::vector<FCheck> &fcheck){this->Init(fcheck);}
    MFCheck& operator=(MFCheck const&rhs){      
      this->_fcheck = rhs._fcheck; return *this;
    }
    void Init(std::vector<FCheck> &fcheck){_fcheck = fcheck;}
    void Load(const char* fname);
    void Save(const char* fname);
    void Write(std::ofstream &s);
    void Read(std::ifstream &s,bool readType = true);
    bool Check(int idx,cv::Mat &im,cv::Mat &s);
  };
  //===========================================================================
}
#endif
