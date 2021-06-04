#include "layer/YoloV5Focus.h"
#include "testutil.h"

static int test_YoloV5Focus(const ncnn::Mat& a)
{
	ncnn::ParamDict pd;

	std::vector<ncnn::Mat> weights(0);

	int ret = test_layer<ncnn::YoloV5Focus>("YoloV5Focus", pd, weights, a);
	if (ret != 0)
	{
		fprintf(stderr, "test_YoloV5Focus failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
	}
	return ret;
}

static int test_YoloV5Focus_0()
{
	return 0
		||test_YoloV5Focus(RandomMat(12, 12, 3))
		||test_YoloV5Focus(RandomMat(12, 6, 3))
		||test_YoloV5Focus(RandomMat(6, 12, 3));
}

static int test_YoloV5Focus_1()
{
	return 0
		||test_YoloV5Focus(RandomMat(12, 12))
		||test_YoloV5Focus(RandomMat(12, 6))
		||test_YoloV5Focus(RandomMat(6, 12));
}

static int test_YoloV5Focus_2()
{
	return 0
		||test_YoloV5Focus(RandomMat(12))
		||test_YoloV5Focus(RandomMat(8))
		||test_YoloV5Focus(RandomMat(6));
}

int main()
{
	SRAND(7767517);

	return 0
		|| test_YoloV5Focus_0()
		|| test_YoloV5Focus_1()
		|| test_YoloV5Focus_2();
}