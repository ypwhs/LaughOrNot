#ifndef PAW_h_
#define PAW_h_
#include <FaceTracker/IO.h>
namespace FACETRACKER
{
  //===========================================================================
  /** 
      A Piecewise Affine Warp
  */
  class PAW{
  public:    
    int     _nPix;   /**< Number of pixels                   */
    double  _xmin;   /**< Minimum x-coord for src            */
    double  _ymin;   /**< Minimum y-coord for src            */
    cv::Mat _src;    /**< Source points                      */
    cv::Mat _dst;    /**< destination points                 */
    cv::Mat _tri;    /**< Triangulation                      */
    cv::Mat _tridx;  /**< Triangle for each valid pixel      */
    cv::Mat _mask;   /**< Valid region mask                  */
    cv::Mat _coeff;  /**< affine coeffs for all triangles    */
    cv::Mat _alpha;  /**< matrix of (c,x,y) coeffs for alpha */
    cv::Mat _beta;   /**< matrix of (c,x,y) coeffs for alpha */
    cv::Mat _mapx;   /**< x-destination of warped points     */
    cv::Mat _mapy;   /**< y-destination of warped points     */

    PAW(){;}
    PAW(const char* fname){this->Load(fname);}
    PAW(cv::Mat &src,cv::Mat &tri){this->Init(src,tri);}
    PAW& operator=(PAW const&rhs);
    inline int nPoints(){return _src.rows/2;}
    inline int nTri(){return _tri.rows;}
    inline int Width(){return _mask.cols;}
    inline int Height(){return _mask.rows;}
    void Load(const char* fname);
    void Save(const char* fname);
    void Write(std::ofstream &s);
    void Read(std::ifstream &s,bool readType = true);
    void Init(cv::Mat &src,cv::Mat &tri);
    void Crop(cv::Mat &src, cv::Mat &dst,cv::Mat &s);

  private:    
    void CalcCoeff();
    void WarpRegion(cv::Mat &mapx,cv::Mat &mapy);
  };
  //===========================================================================
}
#endif
