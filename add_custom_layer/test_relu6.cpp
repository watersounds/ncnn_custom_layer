#include "layer/relu6.h"
#include "testutil.h"

static int test_relu6(const ncnn::Mat& a)
{
	ncnn::ParamDict pd;

	std::vector<ncnn::Mat> weights(0);

	int ret = test_layer<ncnn::Relu6>("Relu6", pd, weights, a);
	if (ret != 0)
	{
		fprintf(stderr, "test_relu6 failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
	}

	return ret;
}

static int test_relu6_0()
{
	return 0
		|| test_relu6(RandomMat(5, 7, 24))
		|| test_relu6(RandomMat(7, 9, 12))
		|| test_relu6(RandomMat(3, 5, 13));
}

static int test_relu6_1()
{
	return 0
		|| test_relu6(RandomMat(15, 24))
		|| test_relu6(RandomMat(17, 12))
		|| test_relu6(RandomMat(19, 15));
}

static int test_relu6_2()
{
	return 0
		|| test_relu6(RandomMat(128))
		|| test_relu6(RandomMat(124))
		|| test_relu6(RandomMat(127));
}

int main()
{
	SRAND(7767517);

	return 0
		|| test_relu6_0()
		|| test_relu6_1()
		|| test_relu6_2();
}
