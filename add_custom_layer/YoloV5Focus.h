#pragma once
#ifndef LAYER_YOLOV5FOCUS_H
#define LAYER_YOLOV5FOCUS_H

#include "layer.h"

namespace ncnn {

	class YoloV5Focus : public Layer
	{
	public:
		YoloV5Focus();

		virtual int forward(const Mat&, Mat&, const Option&) const;
	};
} // namespace ncnn


#endif // LAYER_YOLOV5FOCUS_H

//DEFINE_LAYER_CREATOR(YoloV5Focus)