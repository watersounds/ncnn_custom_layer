#include "layer.h"
using namespace ncnn;

class MyLayer : public Layer
{
public:
	MyLayer();
	virtual int load_param(const ParamDict& pd);
	virtual int load_model(const ModelBin& mb);

	virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
	virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;



private:
	int channels;
	float eps;
	Mat gamma_data;

};

