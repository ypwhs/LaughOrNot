#ifndef FDet_h_
#define FDet_h_
#include <FaceTracker/IO.h>
namespace FACETRACKER
{
  class FDet{
  public:    
    int _min_neighbours;
    int _min_size;
    double _img_scale;
    double _scale_factor;
    CvHaarClassifierCascade* _cascade;

    FDet(){storage_=NULL;_cascade=NULL;}
    FDet(const char* fname){this->Load(fname);}
    FDet(const char*  cascFile,
   const double img_scale = 1.3,
   const double scale_factor = 1.1,
   const int    min_neighbours = 2,
   const int    min_size = 30){
      this->Init(cascFile,img_scale,scale_factor,min_neighbours,min_size);
    }
    ~FDet();
    FDet& operator=(FDet const&rhs);
    void Init(const char* fname,
        const double img_scale = 1.3,
        const double scale_factor = 1.1,
        const int    min_neighbours = 2,
        const int    min_size = 30);
    cv::Rect Detect(cv::Mat im);
    void Load(const char* fname);
    void Save(const char* fname);
    void Write(std::ofstream &s);
    void Read(std::ifstream &s,bool readType = true);
    
  private:
    cv::Mat small_img_; CvMemStorage* storage_;
  };
}
#endif
