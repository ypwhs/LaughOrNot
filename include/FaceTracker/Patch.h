#ifndef Patch_h_
#define Patch_h_
#include <FaceTracker/IO.h>
namespace FACETRACKER
{
  //===========================================================================
  /** 
      A Patch Expert
  */
  class Patch{
  public:
    int     _t; /**< Type of patch (0=raw,1=grad,2=lbp) */
    double  _a; /**< scaling                            */
    double  _b; /**< bias                               */
    cv::Mat _W; /**< Gain                               */
    
    Patch(){;}
    Patch(const char* fname){this->Load(fname);}
    Patch(int t,double a,double b,cv::Mat &W){this->Init(t,a,b,W);}
    Patch& operator=(Patch const&rhs);
    inline int w(){return _W.cols;}
    inline int h(){return _W.rows;}
    void Load(const char* fname);
    void Save(const char* fname);
    void Write(std::ofstream &s);
    void Read(std::ifstream &s,bool readType = true);
    void Init(int t, double a, double b, cv::Mat &W);
    void Response(cv::Mat &im,cv::Mat &resp);    

  private:
    cv::Mat im_,res_;
  };
  //===========================================================================
  /**
     A Multi-patch Expert
  */
  class MPatch{
  public:
    int _w,_h;             /**< Width and height of patch */
    std::vector<Patch> _p; /**< List of patches           */
    
    MPatch(){;}
    MPatch(const char* fname){this->Load(fname);}
    MPatch(std::vector<Patch> &p){this->Init(p);}
    MPatch& operator=(MPatch const&rhs);
    inline int nPatch(){return _p.size();}
    void Load(const char* fname);
    void Save(const char* fname);
    void Write(std::ofstream &s);
    void Read(std::ifstream &s,bool readType = true);
    void Init(std::vector<Patch> &p);
    void Response(cv::Mat &im,cv::Mat &resp);    

  private:
    cv::Mat res_;
  };
  //===========================================================================
}
#endif
