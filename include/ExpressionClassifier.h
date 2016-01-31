#pragma once

#include "Expression.h"
#include <FaceTracker/Tracker.h>

class ExpressionClassifier {
public:
	ExpressionClassifier();
	void load();
    void save() const;
    unsigned int classify(const cv::Mat& objectPoints);
	unsigned int getPrimaryExpression() const;
	double getProbability(unsigned int i) const;
	cv::string getDescription(unsigned int i) const;
	Expression& getExpression(unsigned int i);
	void setSigma(double sigma);
	double getSigma() const;
	unsigned int size() const;
    void addExpression(cv::string description = "");
    void addExpression(Expression& expression);
    void addSample(const cv::Mat& objectPoints);
	void reset();
protected:
	std::vector<Expression> expressions;
	std::vector<double> probability;
	float sigma;
};
