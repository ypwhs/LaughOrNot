#pragma once

#include <FaceTracker/Tracker.h>

class Expression {
public:
	Expression(cv::string description = "");
	void setDescription(cv::string description);
	cv::string getDescription() const;
	void addSample(const cv::Mat& sample);
	cv::Mat& getExample(unsigned int i);
	unsigned int size() const;
	void reset();
	void save(cv::string filename) const;
	void load(cv::string filename);
protected:
	cv::string description;
	std::vector<cv::Mat> samples;
};
