#include "lenet.h"

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

/*
void CONVOLUTION_FORWARD(double* input[][], double*** output, double**** weight, double* bias, double(*action)(double))					
{																				
	for (int x = 0; x < GETLENGTH(weight); ++x)									
		for (int y = 0; y < GETLENGTH(*weight); ++y)							
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					
		                                                                        
	FOREACH(j, GETLENGTH(output))												
		FOREACH(i, GETCOUNT(output[j]))											
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	
}
*/


#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
		                                                                        \
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
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
		bd[j] += ((double *)outerror[j])[i];								\
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

double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
    // CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
    // #define CONVOLUTION_FORWARD(input,output,weight,bias,action)		
    #pragma omp parallel for																							
    for (int x = 0; x < 1; ++x)									
        for (int y = 0; y < 6; ++y)							
            CONVOLUTE_VALID(features->input[x], features->layer1[y], lenet->weight0_1[x][y]);					
    // printf("GETLENGTH(lenet->weight0_1) %d  %d\n", GETLENGTH(lenet->weight0_1), GETLENGTH(*(lenet->weight0_1)));

    #pragma omp parallel for		                                                                        
    FOREACH(j, 6)												
        FOREACH(i, 784)											
            ((double *)features->layer1[j])[i] = action(((double *)features->layer1[j])[i] + lenet->bias0_1[j]);	
    // printf("GETLENGTH(features->layer1)  %d  %d\n", GETLENGTH(features->layer1), GETCOUNT(features->layer1[0]));

    SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);

    // CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
    #pragma omp parallel for
    for (int x = 0; x < 6; ++x)									
        for (int y = 0; y < 16; ++y)							
            CONVOLUTE_VALID(features->layer2[x], features->layer3[y], lenet->weight2_3[x][y]);					
    // printf("GETLENGTH(lenet->weight2_3)  %d  %d\n", GETLENGTH(lenet->weight2_3), GETLENGTH(*(lenet->weight2_3)));

    #pragma omp parallel for		                                                                        
    FOREACH(j, 16)												
        FOREACH(i, 100)											
            ((double *)features->layer3[j])[i] = action(((double *)features->layer3[j])[i] + lenet->bias2_3[j]);
    // printf("GETLENGTH(features->layer3)  %d  %d\n", GETLENGTH(features->layer3), GETCOUNT(features->layer3[0]));

	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);


	// CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	#pragma omp parallel for
    for (int x = 0; x < 16; ++x)									
        for (int y = 0; y < 120; ++y)							
            CONVOLUTE_VALID(features->layer4[x], features->layer5[y], lenet->weight4_5[x][y]);					
    // printf("GETLENGTH(lenet->weight4_5)  %d  %d\n", GETLENGTH(lenet->weight4_5), GETLENGTH(*(lenet->weight4_5)));

    #pragma omp parallel for		                                                                        
    FOREACH(j, 120)												
        FOREACH(i, 1)											
            ((double *)features->layer5[j])[i] = action(((double *)features->layer5[j])[i] + lenet->bias4_5[j]);	
    // printf("GETLENGTH(features->layer5)  %d  %d\n", GETLENGTH(features->layer5), GETCOUNT(features->layer5[0]));

	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);

//	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	#pragma omp parallel for
    for(int x = 0; x < GETLENGTH(lenet->weight4_5); ++x)
        for (int y = 0; y < GETLENGTH(*(lenet->weight4_5)); ++y)
            CONVOLUTE_FULL(errors->layer5[y], errors->layer4[x], lenet->weight4_5[x][y]);

    #pragma omp parallel for        	
    FOREACH(i, GETCOUNT(errors->layer4))											
        ((double *)errors->layer4)[i] *= actiongrad(((double *)features->layer4)[i]);

    #pragma omp parallel for    	
    FOREACH(j, GETLENGTH(errors->layer5))											
        FOREACH(i, GETCOUNT(errors->layer5[j]))									
            deltas->bias4_5[j] += ((double *)errors->layer5[j])[i];

    #pragma omp parallel for        								
    for (int x = 0; x < GETLENGTH(lenet->weight4_5); ++x)								
        for (int y = 0; y < GETLENGTH(*(lenet->weight4_5)); ++y)						
            CONVOLUTE_VALID(features->layer4[x], deltas->weight4_5[x][y], errors->layer5[y]);	

	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);

	// CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
    #pragma omp parallel for
    for(int x = 0; x < GETLENGTH(lenet->weight2_3); ++x)
        for (int y = 0; y < GETLENGTH(*(lenet->weight2_3)); ++y)
            CONVOLUTE_FULL(errors->layer3[y], errors->layer2[x], lenet->weight2_3[x][y]);

    #pragma omp parallel for        	
    FOREACH(i, GETCOUNT(errors->layer2))											
        ((double *)errors->layer2)[i] *= actiongrad(((double *)features->layer2)[i]);

    #pragma omp parallel for    	
    FOREACH(j, GETLENGTH(errors->layer3))											
        FOREACH(i, GETCOUNT(errors->layer3[j]))									
            deltas->bias2_3[j] += ((double *)errors->layer3[j])[i];

    #pragma omp parallel for        								
    for (int x = 0; x < GETLENGTH(lenet->weight2_3); ++x)								
        for (int y = 0; y < GETLENGTH(*(lenet->weight2_3)); ++y)						
            CONVOLUTE_VALID(features->layer2[x], deltas->weight2_3[x][y], errors->layer3[y]);	

	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);

		/*
#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
} */
	// CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
    #pragma omp parallel for
    for(int x = 0; x < GETLENGTH(lenet->weight0_1); ++x)
        for (int y = 0; y < GETLENGTH(*(lenet->weight0_1)); ++y)
            CONVOLUTE_FULL(errors->layer1[y], errors->input[x], lenet->weight0_1[x][y]);

    #pragma omp parallel for        	
    FOREACH(i, GETCOUNT(errors->input))											
        ((double *)errors->input)[i] *= actiongrad(((double *)features->input)[i]);

    #pragma omp parallel for    	
    FOREACH(j, GETLENGTH(errors->layer1))											
        FOREACH(i, GETCOUNT(errors->layer1[j]))									
            deltas->bias0_1[j] += ((double *)errors->layer1[j])[i];

    #pragma omp parallel for        								
    for (int x = 0; x < GETLENGTH(lenet->weight0_1); ++x)								
        for (int y = 0; y < GETLENGTH(*(lenet->weight0_1)); ++y)						
            CONVOLUTE_VALID(features->input[x], deltas->weight0_1[x][y], errors->layer1[y]);	

	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);    	
}

static inline void load_input(Feature *features, image input)
{
    double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
    
    const long sz = sizeof(image) / sizeof(**input);
    double mean = 0, std = 0;

    #pragma omp parallel for reduction(+:mean) reduction(+:std)
    FOREACH(j, sizeof(image) / sizeof(*input))
        FOREACH(k, sizeof(*input) / sizeof(**input))
	    {
		    mean = mean + input[j][k];
		    std  = std + (input[j][k] * input[j][k]);
	    }
    mean /= sz;
    std = sqrt(std / sz - mean*mean);
    
    #pragma omp parallel for
    FOREACH(j, sizeof(image) / sizeof(*input))
        FOREACH(k, sizeof(*input) / sizeof(**input))
        {
            layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
        }
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
    double inner = 0;

    // #pragma omp parallel for reduction(-:inner)	
    for (int i = 0; i < count; ++i)
    {
        double res = 0;
        for (int j = 0; j < count; ++j)
        {
             res += exp(input[j] - input[i]);
        }
        loss[i] = 1. / res;
        // #pragma omp atomic
        inner = inner - (loss[i] * loss[i]);
    }
    inner += loss[label];

    // #pragma omp parallel for
    for (int i = 0; i < count; ++i)
    {
        loss[i] *= (i == label) - loss[i] - inner;
    }
}

static void load_target(Feature *features, Feature *errors, int label)
{
    double *output = (double *)features->output;
    double *error = (double *)errors->output;
    softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
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


void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
    double buffer[GETCOUNT(LeNet5)] = { 0 };
    int i = 0;
    int num =  (sizeof(LeNet5)/sizeof(double));   

    #pragma omp parallel for
    for (i = 0; i < batchSize; ++i)
    {
        Feature features = { 0 };
        Feature errors = { 0 };
        LeNet5	deltas = { 0 };
        load_input(&features, inputs[i]);
        forward(lenet, &features, relu);
        load_target(&features, &errors, labels[i]);
        backward(lenet, &deltas, &errors, &features, relugrad);
        #pragma omp critical  
        for (int j = 0; j < num; ++j)     
            buffer[j] += ((double *)&deltas)[j];
    }
    double k = ALPHA / batchSize;

    #pragma omp parallel for
    FOREACH(i, num)
        ((double *)lenet)[i] += k * buffer[i];
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

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
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
