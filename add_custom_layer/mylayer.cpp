#include "mylayer.h"

DEFINE_LAYER_CREATOR(MyLayer)

MyLayer::MyLayer()
{
	// one input and one output
	// typical one_blob_only type: Convolution, Pooling, ReLU, Softmax ...
	// typical non-non_blob_only type: Eltwise, Split, Concat, Slice ...
	one_blob_only = true;

	// do not change the blob size, modify data in-place
	// typical support_inplace type: ReLU, Sigmoid ...
	// typical non-support_inplace type: Convolution, Pooling ...
	support_inplace = true;

}

int MyLayer::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
	//check inputs dims, return non-zero on error
	if (bottom_blob.c != channels)
	{
		return -1;
	}

	// x = (x + eps) * gamma_per_channel

	int w = bottom_blob.w;
	int h = bottom_blob.h;
	size_t elemsize = bottom_blob.elemsize;
	int size = w * h;

	top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
	if (top_blob.empty())
		return -100; // return non-zero on error, -100 indicates out-of-memory

	#pragma omp parallel for num_threads(opt.num_threads)
	for (int  q = 0; q < channels; q++)
	{
		const float* ptr = bottom_blob.channel(q);
		float* outptr = top_blob.channel(q);
		const float gamma = gamma_data[q];

		for (int i = 0; i < size; i++)
		{
			outptr[i] = (ptr[i] + eps) * gamma;
		}
	}
	return 0;
}

int MyLayer::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
	// check input dims, return non-zero on error
	if (bottom_top_blob.c != channels)
		return -1;
	// x = (x + eps) * gamma_per_channel

	int w = bottom_top_blob.w;
	int h = bottom_top_blob.h;
	int size = w * h;


	#pragma omp parallel for num_threads(opt.num_threads)
	for (int q = 0; q < channels; q++)
	{
		float* ptr = bottom_top_blob.channel(q);
		const float gamma = gamma_data[q];

		for (int i = 0; i < size; i++)
		{
			ptr[i] = (ptr[i] + eps) * gamma;
		}
	}
	return 0;
}


int MyLayer::load_param(const ParamDict& pd)
{
	channels = pd.get(0, 0);
	eps = pd.get(1, 0.001f);

	// you could alter the behavior based on loaded parameter
	// if (eps == 0.001f)
	// {
	//     one_blob_only = false;
	//     support_inplace = false;
	// }

	return 0;
}

int MyLayer::load_model(const ModelBin& mb)
{
	// read weights with length of channels * sizeof(float)
	// the second argument explains as follows
	// 0 judge the value type automatically, you may get float or float16 or uint8 etc
	//   depends on the model storage and the supporting target hardware
	// 1 read float values anyway
	// 2 read float16 values anyway
	// 3 read uint8 values anyway
	gamma_data = mb.load(channels, 1);
	if (gamma_data.empty())
		return -100;// return non-zero on error, -100 indicates out-of-memory

	// you could alter the behavior based on loaded weight
	// if (gamma_data[0] == 0.f)
	// {
	//     one_blob_only = false;
	//     support_inplace = false;
	// }

	
	return 0;// return zero if success
}