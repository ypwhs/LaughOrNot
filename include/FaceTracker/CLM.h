#ifndef CLM_h
#define CLM_h
#include <FaceTracker/PDM.h>
#include <FaceTracker/Patch.h>
#include <vector>
namespace FACETRACKER
{
  //===========================================================================
  /** 
      A Constrained Local Model
  */
  class CLM{
  public:
    PDM                               _pdm;   /**< 3D Shape model           */
    cv::Mat                           _plocal;/**< local parameters         */
    cv::Mat                           _pglobl;/**< global parameters        */
    cv::Mat                           _refs;  /**< Reference shape          */
    std::vector<cv::Mat>              _cent;  /**< Centers/view (Euler)     */
    std::vector<cv::Mat>              _visi;  /**< Visibility for each view */
    std::vector<std::vector<MPatch> > _patch; /**< Patches/point/view       */
    
    CLM(){;}
    CLM(const char* fname){this->Load(fname);}
    CLM(PDM &s,cv::Mat &r, std::vector<cv::Mat> &c,
	std::vector<cv::Mat> &v,std::vector<std::vector<MPatch> > &p){
      this->Init(s,r,c,v,p);
    }
    CLM& operator=(CLM const&rhs);
    inline int nViews(){return _patch.size();}
    int GetViewIdx();
    void Load(const char* fname);
    void Save(const char* fname);
    void Write(std::ofstream &s);
    void Read(std::ifstream &s,bool readType = true);
    void Init(PDM &s,cv::Mat &r, std::vector<cv::Mat> &c,
	      std::vector<cv::Mat> &v,std::vector<std::vector<MPatch> > &p);
    void Fit(cv::Mat im, std::vector<int> &wSize,
	     int nIter = 10,double clamp = 3.0,double fTol = 0.0);
  private:
    cv::Mat cshape_,bshape_,oshape_,ms_,u_,g_,J_,H_; 
    std::vector<cv::Mat> prob_,pmem_,wmem_;
    void Optimize(int idx,int wSize,int nIter,
		  double fTol,double clamp,bool rigid);
  };
  //===========================================================================
}
#endif
