#include "ExpressionClassifier.h"

using namespace cv;
using namespace std;

/*
	sigma describes the classification sharpness. A larger sigma means the
	boundary between different expressions is more blurry. It won't change
	the classification, but will give you probabilities that are smoother.
*/

ExpressionClassifier::ExpressionClassifier()
:sigma(10.0) {
}

void ExpressionClassifier::save() const {
	for(unsigned int i = 0; i < size(); i++) {
		string filename = expressions[i].getDescription() + ".yml";
		cout << "saving to " << filename << endl;
		expressions[i].save(filename);
	}
}

void ExpressionClassifier::load() {
	string filenames[] = {"neutral.yml", "smile.yml", "surprise.yml"};
	unsigned int n = 3;
	expressions.resize(n);
	for(unsigned int i = 0; i < n; i++) {
		expressions[i].load(filenames[i]);
	}
}

unsigned int ExpressionClassifier::classify(const Mat& objectPoints) {
	Mat cur;
	objectPoints.copyTo(cur);
	norm(cur);
	int n = size();
	probability.resize(n);
	if(n == 0) {
		return 0;
	}
	vector<vector<double> > val(n);
	double sum = 0;
	for(int i = 0; i < n; i++){
		int m = expressions[i].size();
		for(int j = 0; j < m; j++){
			double v = norm(cur, expressions[i].getExample(j));
			double p = exp(-v * v / sigma);
			val[i].push_back(p);
			sum += p;
		}
	}
	for(int i = 0; i < n; i++){
		probability[i] = 0;
		int m = expressions[i].size();
		for(int j = 0; j < m; j++) {
			probability[i] += val[i][j];
		}
		probability[i] /= sum;
	}
	return getPrimaryExpression();
}

unsigned int ExpressionClassifier::getPrimaryExpression() const {
	int maxExpression = 0;
	double maxProbability = 0;
	for(int i = 0; i < (int)probability.size(); i++) {
		double cur = getProbability(i);
		if(cur > maxProbability) {
			maxExpression = i;
			maxProbability = cur;
		}
	}
	return maxExpression;
}


double ExpressionClassifier::getProbability(unsigned int i) const {
	if(i < probability.size()) {
		return probability[i];
	} else {
		return 0;
	}
}

string ExpressionClassifier::getDescription(unsigned int i) const {
	return expressions[i].getDescription();
}

Expression& ExpressionClassifier::getExpression(unsigned int i) {
	return expressions[i];
}

void ExpressionClassifier::setSigma(double sigma) {
	this->sigma = sigma;
}

double ExpressionClassifier::getSigma() const {
	return sigma;
}

unsigned int ExpressionClassifier::size() const {
	return expressions.size();
}

void ExpressionClassifier::reset() {
	expressions.clear();
}
