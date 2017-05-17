#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

//1M
#include "lenet.h"
#include <memory.h>

#include <math.h>

#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000
// #define BATCHSIZE		300

#define BLOCK_NUM 40 //40
#define THREAD_NUM 256// 256
#define trainBLOCK_NUM  2
#define trainTHREAD_NUM 160 // 1024  // 320 // 160
#define BATCHSIZE		trainBLOCK_NUM * trainTHREAD_NUM


int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image || !fp_label) return 1;
    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(*data)*count, 1, fp_image);
    fread(label, count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
    return 0;
}

int save(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) return 1;
    fwrite(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

int load(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

double wtime(void)
{
    double          now_time;
    struct timeval  etstart;

    if (gettimeofday(&etstart, NULL) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((etstart.tv_sec) * 1000 + etstart.tv_usec / 1000.0);
    return now_time;
}

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
    FOREACH(o0,GETLENGTH(output))														\
    FOREACH(o1,GETLENGTH(*(output)))												    \
    FOREACH(w0,GETLENGTH(weight))												        \
    FOREACH(w1,GETLENGTH(*(weight)))										            \
        (output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	            \
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
    FOREACH(i0,GETLENGTH(input))														\
    FOREACH(i1,GETLENGTH(*(input)))													    \
    FOREACH(w0,GETLENGTH(weight))												        \
    FOREACH(w1,GETLENGTH(*(weight)))										            \
        (output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	            \
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					    \
{																				    \
    for (int x = 0; x < GETLENGTH(weight); ++x)									    \
    for (int y = 0; y < GETLENGTH(*weight); ++y)							        \
        CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					        \
    FOREACH(j, GETLENGTH(output))												    \
        FOREACH(i, GETCOUNT(output[j]))											    \
            ((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
    for (int x = 0; x < GETLENGTH(weight); ++x)								\
        for (int y = 0; y < GETLENGTH(*weight); ++y)						\
            CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
    FOREACH(i, GETCOUNT(inerror))											\
        ((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
    FOREACH(j, GETLENGTH(outerror))											\
        FOREACH(i, GETCOUNT(outerror[j]))									\
            bd[j] += ((double *)outerror[j])[i];							\
    for (int x = 0; x < GETLENGTH(weight); ++x)								\
        for (int y = 0; y < GETLENGTH(*weight); ++y)						\
            CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
    const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
    const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
    FOREACH(i, GETLENGTH(output))																\
        FOREACH(o0, GETLENGTH(*(output)))															\
            FOREACH(o1, GETLENGTH(**(output)))															\
            {																							\
                int x0 = 0, x1 = 0, ismax;																\
                FOREACH(l0, len0)																		\
                    FOREACH(l1, len1)																	\
                    {																						\
                        ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
                        x0 += ismax * (l0 - x0);															\
                        x1 += ismax * (l1 - x1);															\
                    }																						\
                    output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
            }																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
    const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
    const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
    FOREACH(i, GETLENGTH(outerror))																\
        FOREACH(o0, GETLENGTH(*(outerror)))															\
            FOREACH(o1, GETLENGTH(**(outerror)))														\
            {																							\
                int x0 = 0, x1 = 0, ismax;																\
                FOREACH(l0, len0)																		\
                    FOREACH(l1, len1)																	\
                    {																						\
                        ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
                        x0 += ismax * (l0 - x0);															\
                        x1 += ismax * (l1 - x1);															\
                    }																						\
                inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
            }																							\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
    for (int x = 0; x < GETLENGTH(weight); ++x)								\
        for (int y = 0; y < GETLENGTH(*weight); ++y)						\
            ((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
    FOREACH(j, GETLENGTH(bias))												\
        ((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
    for (int x = 0; x < GETLENGTH(weight); ++x)									\
        for (int y = 0; y < GETLENGTH(*weight); ++y)							\
            ((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
    FOREACH(i, GETCOUNT(inerror))												\
        ((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
    FOREACH(j, GETLENGTH(outerror))												\
        bd[j] += ((double *)outerror)[j];										\
    for (int x = 0; x < GETLENGTH(weight); ++x)									\
        for (int y = 0; y < GETLENGTH(*weight); ++y)							\
            wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}

__host__ __device__ double relu(double x)
{
    return x*(x > 0);
}

__host__ __device__ double relugrad(double y)
{
    return y > 0;
}


__host__ __device__ static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
    CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
    SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
    CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
    SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
    CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
    DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

__host__ __device__ static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
    DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
    CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
    SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
    CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
    SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
    CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}


__host__ __device__ static inline void load_input(Feature *features, image input)
{
    double(*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
    const long sz = sizeof(image) / sizeof(**input);
    double mean = 0, std = 0;
    FOREACH(j, sizeof(image) / sizeof(*input))
        FOREACH(k, sizeof(*input) / sizeof(**input))
        {
            mean += input[j][k];
            std += input[j][k] * input[j][k];
        }
    mean /= sz;
    std = sqrt(std / sz - mean*mean);
    FOREACH(j, sizeof(image) / sizeof(*input))
        FOREACH(k, sizeof(*input) / sizeof(**input))
        {
            layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
        }
}

__host__ __device__ static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
    double inner = 0;
    for (int i = 0; i < count; ++i)
    {
        double res = 0;
        for (int j = 0; j < count; ++j)
        {
            res += exp(input[j] - input[i]);
        }
        loss[i] = 1. / res;
        inner -= loss[i] * loss[i];
    }
    inner += loss[label];
    for (int i = 0; i < count; ++i)
    {
        loss[i] *= (i == label) - loss[i] - inner;
    }
}

__host__ __device__ static void load_target(Feature *features, Feature *errors, int label)
{
    double *output = (double *)features->output;
    double *error = (double *)errors->output;
    softmax(output, error, label, GETCOUNT(features->output));
}

__host__ __device__ static uint8 get_result(Feature *features, uint8 count)
{
    double *output = (double *)features->output;
    const int outlen = GETCOUNT(features->output);
    uint8 result = 0;
    double maxvalue = *output;
    for (uint8 i = 1; i < count; ++i)
    {
        if (output[i] > maxvalue)
        {
            maxvalue = output[i];
            result = i;
        }
    }
    return result;
}

static double f64rand()
{
    static int randbit = 0;
    if (!randbit)
    {
        srand((unsigned)time(0));
        for (int i = RAND_MAX; i; i >>= 1, ++randbit);
    }

    unsigned long long lvalue = 0x4000000000000000L;
    int i = 52 - randbit;
    for (; i > 0; i -= randbit)
        lvalue |= (unsigned long long)rand() << i;
    lvalue |= (unsigned long long)rand() >> -i;

    return *(double *)&lvalue - 3;
}

__global__ void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize, LeNet5 *gputrain )
{	
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num = bid*blockDim.x + tid;
    Feature gpufeatures = { 0 };
    Feature gpuerrors = { 0 };

    // gputrain[num] = { 0 };
    // cudaMemset(gputrain[num], 0, sizeof(LeNet5));  
    load_input(&gpufeatures, inputs[num]);
    forward(lenet, &gpufeatures, relu);
    load_target(&gpufeatures, &gpuerrors, labels[num]);
    backward(lenet, gputrain+num, &gpuerrors, &gpufeatures, relugrad);
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
    Feature features = { 0 };
    Feature errors = { 0 };
    LeNet5 deltas = { 0 };
    load_input(&features, input);
    forward(lenet, &features, relu);
    load_target(&features, &errors, label);
    backward(lenet, &deltas, &errors, &features, relugrad);
    FOREACH(i, GETCOUNT(LeNet5))
        ((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

__host__ __device__ uint8 Predict(LeNet5 *lenet, image input, uint8 count)
{
    Feature features = { 0 };
    load_input(&features, input);
    forward(lenet, &features, relu);
    return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
    for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
    for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
    for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
    for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
    for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
    for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}

//打印设备信息
void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//CUDA 初始化
bool InitCUDA()
{
    int count;

    //取得支持Cuda的装置的数目
    cudaGetDeviceCount(&count);

    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;

    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
		//打印设备信息
        printDeviceProp(prop);

        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }

    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
    LeNet5 *tgpulenet;
    cudaMalloc((void**)&tgpulenet, sizeof(LeNet5));
    image *gputrain_data;
    cudaMalloc((void**)&gputrain_data, COUNT_TRAIN * sizeof(image));
    uint8 *gputrain_label;
    cudaMalloc((void**)&gputrain_label, COUNT_TRAIN * sizeof(uint8));
    LeNet5 *gputrain;
    cudaMalloc((void**)&gputrain, sizeof(LeNet5)*batch_size);
    // Feature *gpufeatures;
    // cudaMalloc((void**)&gpufeatures, sizeof(Feature)*batch_size);
    // Feature *gpuerrors;
    // cudaMalloc((void**)&gpuerrors, sizeof(Feature)*batch_size);
    LeNet5 *deltas= (LeNet5 *) malloc(sizeof(LeNet5)*batch_size);
	
    LeNet5	ttt = { 0 };
    cudaMemcpy(gputrain_data, train_data, COUNT_TRAIN * sizeof(image), cudaMemcpyHostToDevice);

    cudaMemcpy(gputrain_label, train_label, COUNT_TRAIN * sizeof(uint8), cudaMemcpyHostToDevice);

    for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
    {
        cudaMemset(gputrain, 0, sizeof(LeNet5)*batch_size);
        // cudaMemset(gpufeatures, 0, sizeof(Feature)*batch_size);	
        // cudaMemset(gpuerrors, 0, sizeof(Feature)*batch_size);	
        cudaMemcpy(tgpulenet, lenet, sizeof(LeNet5), cudaMemcpyHostToDevice);
        TrainBatch << < trainBLOCK_NUM, trainTHREAD_NUM >> > (tgpulenet, gputrain_data + i, gputrain_label + i, batch_size, gputrain);
        cudaMemcpy(deltas, gputrain, sizeof(LeNet5) * batch_size, cudaMemcpyDeviceToHost);
	    
        double buffer[GETCOUNT(LeNet5)] = { 0 };
        for (int mm = 0; mm < batch_size; ++mm)
        {
            ttt = deltas[mm];
            FOREACH(j, GETCOUNT(LeNet5))
                buffer[j] += ((double *)&ttt)[j];
        }
        FOREACH(nn, GETCOUNT(LeNet5))
            ((double *)lenet)[nn] +=  buffer[nn] * ALPHA / batch_size;

        if (i * 100 / total_size > percent)
            printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
    }
    cudaFree(tgpulenet);
    cudaFree(gputrain_data);
    cudaFree(gputrain_label);
    cudaFree(gputrain); 
    free(deltas);
}


__global__ static void testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size, int *gpuresult)
{
	//	int right = 0;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ int shared[];
    shared[tid] = 0;
    for (int i = bid * THREAD_NUM + tid; i < total_size; i += BLOCK_NUM * THREAD_NUM) {
        uint8 l = test_label[i];
        int p = Predict(lenet, test_data[i], 10);
        shared[tid] += l == p;
    }

    //同步 保证每个 thread 都已经把结果写到 shared[tid] 里面
    __syncthreads();

    //使用线程0完成加和
    if (tid == 0)
    {
        for (int i = 1; i < THREAD_NUM; i++)
        {
            shared[0] += shared[i];
        }
        gpuresult[bid] = shared[0];

    }
}



int main()
{
	//CUDA 初始化
    if (!InitCUDA()) {
        return 0;
    }


    image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
    uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
    if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
    {
        printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
        free(train_data);
        free(train_label);
    }
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
    {
        printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
        free(test_data);
        free(test_label);
    }

    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    if (load(lenet, LENET_FILE))
        Initial(lenet);

    clock_t start = clock();
    double bef_train = wtime();     

    training(lenet, train_data, train_label, BATCHSIZE, COUNT_TRAIN);

    clock_t end_train = clock();
    double aft_train = wtime(); 

    LeNet5 *gpulenet;
    cudaMalloc((void**)&gpulenet, sizeof(LeNet5));
    cudaMemcpy(gpulenet, lenet, sizeof(LeNet5), cudaMemcpyHostToDevice);
    image *gputest_data;
    cudaMalloc((void**)&gputest_data, COUNT_TEST * sizeof(image));
    cudaMemcpy(gputest_data, test_data, COUNT_TEST * sizeof(image), cudaMemcpyHostToDevice);
    uint8 *gputest_label;
    cudaMalloc((void**)&gputest_label, COUNT_TEST * sizeof(uint8));
    cudaMemcpy(gputest_label, test_label, COUNT_TEST * sizeof(uint8), cudaMemcpyHostToDevice);

    int *gpuresult;
    cudaMalloc((void**)&gpuresult, sizeof(int)*BLOCK_NUM);
	
    testing <<< BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >>> (gpulenet, gputest_data, gputest_label, COUNT_TEST, gpuresult);

    int sum[BLOCK_NUM];
    cudaMemcpy(&sum, gpuresult, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);
    cudaFree(gpulenet);
    cudaFree(gputest_data);
    cudaFree(gputest_label);
    cudaFree(gpuresult);
    int right = 0;

    for (int i = 0; i < BLOCK_NUM; i++) {
        right += sum[i];
    }

    clock_t end_test = clock();
    double aft_test = wtime();

    printf("Precision is %f\n", right*1.0/(COUNT_TEST*1.0));  
    // printf("Traing CPU time:%u\n", (unsigned)(end_train - start));
    // printf("Traing CPU time with second:%fs\n", (float)(end_train - start)*1.0/(CLOCKS_PER_SEC*1.0));
    // printf("Test CPU time:%u\n", (unsigned)(end_test - end_train));
    // printf("Test CPU time with second:%fs\n", (float)(end_test - end_train)*1.0/(CLOCKS_PER_SEC*1.0));
    printf("REAL Training time is %lfs\n", (aft_train - bef_train)/1000.0);
    printf("REAL Testing time is %lfs\n", (aft_test - aft_train)/1000.0);

    free(lenet);
    free(train_data);
    free(train_label);
    free(test_data);
    free(test_label);

    return 0;	
}






