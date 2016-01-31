#include <FaceTracker/IO.h>
#include <stdio.h>
using namespace FACETRACKER;
using namespace std;
//===========================================================================
void IO::ReadMat(ifstream& s,cv::Mat &M)
{
  int r,c,t; s >> r >> c >> t;
  M = cv::Mat(r,c,t);
  switch(M.type()){
  case CV_64FC1: 
    {
      cv::MatIterator_<double> i1 = M.begin<double>(),i2 = M.end<double>();
      while(i1 != i2)s >> *i1++;
    }break;
  case CV_32FC1:
    {
      cv::MatIterator_<float> i1 = M.begin<float>(),i2 = M.end<float>();
      while(i1 != i2)s >> *i1++;
    }break;
  case CV_32SC1:
    {
      cv::MatIterator_<int> i1 = M.begin<int>(),i2 = M.end<int>();
      while(i1 != i2)s >> *i1++;
    }break;
  case CV_8UC1:
    {
      cv::MatIterator_<uchar> i1 = M.begin<uchar>(),i2 = M.end<uchar>();
      while(i1 != i2)s >> *i1++;
    }break;
  default:
    printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", 
	   __FILE__,__LINE__,M.type()); abort();
  }return;
}
//===========================================================================
void IO::WriteMat(ofstream& s,cv::Mat &M)
{
  s << M.rows << " " << M.cols << " " << M.type() << " ";
  switch(M.type()){
  case CV_64FC1: 
    {
      cv::MatIterator_<double> i1 = M.begin<double>(),i2 = M.end<double>();
      while(i1 != i2)s << *i1++ << " ";
    }break;
  case CV_32FC1:
    {
      cv::MatIterator_<float> i1 = M.begin<float>(),i2 = M.end<float>();
      while(i1 != i2)s << *i1++ << " ";
    }break;
  case CV_32SC1:
    {
      cv::MatIterator_<int> i1 = M.begin<int>(),i2 = M.end<int>();
      while(i1 != i2)s << *i1++ << " ";
    }break;
  case CV_8UC1:
    {
      cv::MatIterator_<uchar> i1 = M.begin<uchar>(),i2 = M.end<uchar>();
      while(i1 != i2)s << *i1++ << " ";
    }break;
  default:
    printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", 
	   __FILE__,__LINE__,M.type()); abort();
  }return;
}
//===========================================================================
cv::Mat IO::LoadCon(const char* fname)
{
  int i,n; char str[256]; char c; fstream file(fname,fstream::in);
  if(!file.is_open()){
    printf("ERROR(%s,%d) : Failed opening file %s for reading\n", 
	   __FILE__,__LINE__,fname); abort();
  }
  while(1){file >> str; if(strncmp(str,"n_connections:",14) == 0)break;}
  file >> n; cv::Mat con(2,n,CV_32S);
  while(1){file >> c; if(c == '{')break;}
  for(i = 0; i < n; i++)file >> con.at<int>(0,i) >> con.at<int>(1,i);
  file.close(); return con;
}
//=============================================================================
cv::Mat IO::LoadTri(const char* fname)
{
  int i,n; char str[256]; char c; fstream file(fname,fstream::in);
  if(!file.is_open()){
    printf("ERROR(%s,%d) : Failed opening file %s for reading\n", 
	   __FILE__,__LINE__,fname); abort();
  }
  while(1){file >> str; if(strncmp(str,"n_tri:",6) == 0)break;}
  file >> n; cv::Mat tri(n,3,CV_32S);
  while(1){file >> c; if(c == '{')break;}
  for(i = 0; i < n; i++)
    file >> tri.at<int>(i,0) >> tri.at<int>(i,1) >> tri.at<int>(i,2);
  file.close(); return tri;
}
//===========================================================================
