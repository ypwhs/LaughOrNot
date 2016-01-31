#ifndef PDM_h_
#define PDM_h_
#include <FaceTracker/IO.h>
namespace FACETRACKER
{
  //===========================================================================
  /** 
      A 3D Point Distribution Model
  */
  class PDM{
  public:    
    cv::Mat _V; /**< basis of variation                            */
    cv::Mat _E; /**< vector of eigenvalues (row vector)            */
    cv::Mat _M; /**< mean 3D shape vector [x1,..,xn,y1,...yn]      */

    PDM(){;}
    PDM(const char* fname){this->Load(fname);}
    PDM(cv::Mat &M,cv::Mat &V,cv::Mat &E){this->Init(M,V,E);}
    PDM& operator=(PDM const&rhs);
    inline int nPoints(){return _M.rows/3;}
    inline int nModes(){return _V.cols;}
    inline double Var(int i){assert(i<_E.cols); return _E.at<double>(0,i);}
    void Load(const char* fname);
    void Save(const char* fname);
    void Write(std::ofstream &s);
    void Read(std::ifstream &s,bool readType = true);
    void Clamp(cv::Mat &p,double c);
    void Init(cv::Mat &M,cv::Mat &V,cv::Mat &E);
    void Identity(cv::Mat &plocal,cv::Mat &pglobl);
    void CalcShape3D(cv::Mat &s,cv::Mat &plocal);
    void CalcShape2D(cv::Mat &s,cv::Mat &plocal,cv::Mat &pglobl);
    void CalcParams(cv::Mat &s,cv::Mat &plocal,cv::Mat &pglobl);
    void CalcRigidJacob(cv::Mat &plocal,cv::Mat &pglobl,cv::Mat &Jacob);
    void CalcJacob(cv::Mat &plocal,cv::Mat &pglobl,cv::Mat &Jacob);
    void CalcReferenceUpdate(cv::Mat &dp,cv::Mat &plocal,cv::Mat &pglobl);
    void ApplySimT(double a,double b,double tx,double ty,cv::Mat &pglobl);
    
  private:
    cv::Mat S_,R_,s_,P_,Px_,Py_,Pz_,R1_,R2_,R3_;
  };
  //===========================================================================
}
#endif
