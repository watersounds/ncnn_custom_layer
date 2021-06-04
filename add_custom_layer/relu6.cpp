#include "relu6.h"
#include <math.h>

namespace ncnn {

	Relu6::Relu6()
	{
		one_blob_only = true;
		support_inplace = true;
	}

	int Relu6::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
	{
		int w = bottom_top_blob.w;
		int h = bottom_top_blob.h;
		int channels = bottom_top_blob.c;
		int size = w * h;

		#pragma omp parallel for num_threads(opt.num_threads)
		for (int q = 0; q < channels; q++)
		{
			float* ptr = bottom_top_blob.channel(q);

			for (int i = 0; i < size; i++)
			{
				//ptr[i] = std::min(6, (0 > ptr[i] ? 0 : ptr[i]));
				ptr[i] = min(6, max(0, ptr[i]));
				//ptr[i] = 6 < (0 > ptr[i] ? 0 : ptr[i]) ? 6 : (0 > ptr[i] ? 0 : ptr[i]);
			}
		}

		return 0;
	}

} // namespace ncnn