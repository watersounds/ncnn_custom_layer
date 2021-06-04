#pragma once
#ifndef LAYER_RELU6_H
#define LAYER_RELU6_H

#include "layer.h"

namespace ncnn {

	class Relu6 : public Layer
	{
	public:
		Relu6();

		virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
	};

} // namespace ncnn

#endif // LAYER_RELU6_H