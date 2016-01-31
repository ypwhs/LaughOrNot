#ifndef Tracker_h_
#define Tracker_h_
#include <FaceTracker/CLM.h>
#include <FaceTracker/FDet.h>
#include <FaceTracker/FCheck.h>
namespace FACETRACKER
{
  //===========================================================================
  /** 
      Face Tracker
  */
  class Tracker{
  public:    
    CLM        _clm;    /**< Constrained Local Model           */
    FDet       _fdet;   /**< Face Detector                     */
    int64      _frame;  /**< Frame number since last detection */    
    MFCheck    _fcheck; /**< Failure checker                   */
    cv::Mat    _shape;  /**< Current shape                     */
    cv::Mat    _rshape; /**< Reference shape                   */
    cv::Rect   _rect;   /**< Detected rectangle                */
    cv::Scalar _simil;  /**< Initialization similarity         */
    
    /** NULL constructor */
    Tracker(){;}
    
    /** Constructor from model file */
    Tracker(const char* fname){this->Load(fname);}

    /** Constructor from components */
    Tracker(CLM &clm,FDet &fdet,MFCheck &fcheck,
	    cv::Mat &rshape,cv::Scalar &simil){
      this->Init(clm,fdet,fcheck,rshape,simil);
    }
    /**
       Track model in current frame
       @param im     Image containing face
       @param wSize  List of search window sizes (set from large to small)
       @param fpd    Number of frames between detections (-1: never)
       @param nIter  Maximum number of optimization steps to perform.
       @param clamp  Shape model parameter clamping factor (in standard dev's)
       @param fTol   Convergence tolerance of optimization
       @param fcheck Check if tracking succeeded?
       @return       -1 on failure, 0 otherwise.
    */
    int Track(cv::Mat im,std::vector<int> &wSize,
	      const int    fpd    =-1,
	      const int    nIter  = 10,
	      const double clamp  = 3.0,
	      const double fTol   = 0.01,
	      const bool   fcheck = true);

    /** Reset frame number (will perform detection in next image) */
    inline void FrameReset(){_frame = -1;}

    /** Load tracker from model file */
    void Load(const char* fname);

    /** Save tracker to model file */
    void Save(const char* fname);
    
    /** Write tracker to file stream */
    void Write(std::ofstream &s);

    /** Read tracking from file stream */
    void Read(std::ifstream &s,bool readType = true);

  private:
    cv::Mat gray_,temp_,ncc_,small_;
    void Init(CLM &clm,FDet &fdet,MFCheck &fcheck,
	      cv::Mat &rshape,cv::Scalar &simil);    
    void InitShape(cv::Rect &r,cv::Mat &shape);
    cv::Rect ReDetect(cv::Mat &im);
    cv::Rect UpdateTemplate(cv::Mat &im,cv::Mat &s,bool rsize);
  };
  //===========================================================================
}
#endif
