#include "Incs.cuh"

using namespace std;

int read_DataSet(string filepath, vector<string> &value)
{
	int LineNumber = 0;
	ifstream file(filepath);
	string str;
	while (getline(file, str))
	{
		if (str.size() > 0)
		{
			value.push_back(str);
		}
		LineNumber++;
	}
	return LineNumber;
}

int main(int argc, char** argv)
{
	cudaError_t err = cudaSuccess;
	//cudaEvent_t Coverage_Start, Coverage_Stop, Reduction_Start, Reduction_Stop, Fitness_Start, Fitness_Stop;

	int Iter, NumofAttr, NumofClass, TestMode;
	int InitRuleRows = 4;

	string DataSetStr, ClassStr;
	
	//float msecTemp = 0.0f;
	int CPU_StartTime, CPU_StopTime;

	float GPUCoverage_msec = 0.0f;
	float GPUReduction_msec = 0.0f;
	float GPUFitness_msec = 0.0f;

	float CPUInitialization_msec = 0.0f;
	float CPUCoverage_msec = 0.0f;
	float CPUAVGCoverage_msec = 0.0f;
	float CPUReduction_msec = 0.0f;
	float CPUFitness_msec = 0.0f;
	float CPUCrossover_msec = 0.0f;
	float CPUMutation_msec = 0.0f;
	float CPUTest_msec = 0.0f;

	/*float CPUCoverage_msec = 0.0f;
	float CPUReduction_msec = 0.0f;
	float CPUFitness_msec = 0.0f;*/

	if (checkCmdLineFlag(argc, (const char **)argv, "Iter"))
	{
		Iter = getCmdLineArgumentInt(argc, (const char **)argv, "Iter");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "Attr"))
	{
		NumofAttr = getCmdLineArgumentInt(argc, (const char **)argv, "Attr");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "Class"))
	{
		NumofClass = getCmdLineArgumentInt(argc, (const char **)argv, "Class");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "RI"))
	{
		if (getCmdLineArgumentInt(argc, (const char **)argv, "RI") >= 4)
		{
			InitRuleRows = getCmdLineArgumentInt(argc, (const char **)argv, "RI");
		}
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "TM"))
	{
		TestMode = getCmdLineArgumentInt(argc, (const char **)argv, "TM");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "TM"))
	{
		TestMode = getCmdLineArgumentInt(argc, (const char **)argv, "TM");
	}
	/*if (checkCmdLineFlag(argc, (const char **)argv, "TRD"))
	{
		DataSetStr = getCmdLineArgumentInt(argc, (const char **)argv, "TRD");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "TRC"))
	{
		ClassStr = getCmdLineArgumentInt(argc, (const char **)argv, "TRC");
	}*/
	DataSetStr = argv[6];
	ClassStr = argv[7];

	float *DataSet;
	int *Class, *d_Class;
	vector<string> DataSet_Row;
	vector<string> Class_Row;
	int NumofRows = read_DataSet(DataSetStr, DataSet_Row);
	if (NumofRows != (read_DataSet(ClassStr, Class_Row)))
	{
		::exit(EXIT_FAILURE);
	}

	int SizeofDataSet = NumofAttr*NumofRows;
	size_t MemSizeDataSet = sizeof(float)*SizeofDataSet;
	size_t ClassMemSize = sizeof(int)*NumofRows;
	cudaMallocHost((void **)&DataSet, MemSizeDataSet);
	cudaMallocHost((void **)&Class, ClassMemSize);
	vector<float> AttrVec;
	vector<int> ClassVec;

	for (int i = 0; i <= NumofRows; i++)
	{
		stringstream StrStm1(DataSet_Row[i]);
		for (float j; StrStm1 >> j;)
		{
			AttrVec.push_back(j);
			if (StrStm1.peek() == ',')
			{
				StrStm1.ignore();
			}
		}
	}

	//printf("%i , %i\n", SizeofDataSet, AttrVec.size());
	for (int i = 0;i < SizeofDataSet;i++)
	{
		DataSet[i] = AttrVec[i];
	}

	if (TestMode == 1)
	{
		::printf("i&a|");
		for (int a = 0;a < NumofAttr;a++)
		{
			::printf("%i|", a);
		}
		::printf("\n");

		for (int i = 0;i < NumofRows;i++)
		{
			::printf(" %i |", i);
			for (int a = 0;a < NumofAttr;a++)
			{
				::printf("%.0f|", DataSet[(i*NumofAttr) + a]);
			}
			::printf("\n");
		}

	}

	for (int i = 0; i <= NumofRows; i++)
	{
		stringstream StrStm2(Class_Row[i]);
		for (int j; StrStm2 >> j;)
		{
			ClassVec.push_back(j);
			if (StrStm2.peek() == ',')
			{
				StrStm2.ignore();
			}
		}
	}

	for (int i = 0;i < NumofRows;i++)
	{
		Class[i] = ClassVec[i];
	}
	cudaMalloc((void **)&d_Class, ClassMemSize);

	int *NumClass = new int[NumofClass];

	for (int i = 0;i < NumofClass;i++)
	{
		NumClass[i] = 0;
	}

	NumofEachClass(NumClass, NumofRows, Class);

	for (int i = 0;i < NumofClass;i++)
	{
		::printf("Number of Instances of Class %i = %i\n", i, NumClass[i]);
	}
	::printf("*********************************************\n");

	float *MinValue, *MaxValue;
	int MinMaxSize = NumofAttr;
	size_t MinMax_MemSize = MinMaxSize*sizeof(char);
	cudaMallocHost((void **)&MinValue, MinMax_MemSize);
	cudaMallocHost((void **)&MaxValue, MinMax_MemSize);

	int *Rule_Conditions, *Rule_ActiveAttr, *Coverage;
	float *Rule_LowerBound, *Rule_UpperBound;
	int InitialRuleSize = NumofAttr* InitRuleRows*NumofClass;

	size_t InitialRule_MemSize = InitialRuleSize*sizeof(float);
	size_t InitialRuleCond_MemSize = InitialRuleSize*sizeof(int);
	size_t Coverage_MemSize = InitialRuleSize*NumofRows*sizeof(int);

	cudaMallocHost((void **)&Rule_Conditions, InitialRuleCond_MemSize);
	cudaMallocHost((void **)&Rule_LowerBound, InitialRule_MemSize);
	cudaMallocHost((void **)&Rule_UpperBound, InitialRule_MemSize);
	cudaMallocHost((void **)&Rule_ActiveAttr, InitialRuleCond_MemSize);
	cudaMallocHost((void **)&Coverage, Coverage_MemSize);
	int *CPUCoverage = (int *)malloc(Coverage_MemSize);

	int *d_Rule_Conditions, *d_Rule_ActiveAttr, *d_Coverage;
	float *d_DataSet, *d_Rule_LowerBound, *d_Rule_UpperBound;
	//float *d_MinValue, *d_MaxValue;

	cudaMalloc((void **)&d_DataSet, MemSizeDataSet);
	cudaMalloc((void **)&d_Rule_Conditions, InitialRuleCond_MemSize);
	cudaMalloc((void **)&d_Rule_LowerBound, InitialRule_MemSize);
	cudaMalloc((void **)&d_Rule_UpperBound, InitialRule_MemSize);
	cudaMalloc((void **)&d_Rule_ActiveAttr, InitialRuleCond_MemSize);
	cudaMalloc((void **)&d_Coverage, Coverage_MemSize);

	//Fitness Allocation

	dim3 ThreadsPerBlock(32, 32, 1);
	dim3 BlocksPerGrid(((NumofAttr + 32 - 1) / 32), ((NumofRows + 32 - 1) / 32), (InitRuleRows*NumofClass));

	/*dim3 TPB(NumofAttr, 32);
	dim3 BPG(((NumofAttr + 32 - 1) / 32), ((NumofRows*InitRuleRows*NumofClass + 32 - 1) / 32));*/
	dim3 TPB(32, NumofAttr);
	dim3 BPG(((NumofRows*InitRuleRows*NumofClass + 32 - 1) / 32), ((NumofAttr + 32 - 1) / 32));

	dim3 TPB_Fitness(1, 32, InitRuleRows);
	dim3 BPG_Fitness(1, ((NumofRows + 32 - 1) / 32), NumofClass);

	int *GPU_CoverageResult, *d_GPU_CoverageResult;
	size_t CoverageResult_MemSize = InitRuleRows*NumofClass*NumofRows*sizeof(int);
	cudaMallocHost((void **)&GPU_CoverageResult, CoverageResult_MemSize);
	int *CPU_CoverageMatrix = (int *)malloc(Coverage_MemSize);

	cudaMalloc((void **)&d_GPU_CoverageResult, CoverageResult_MemSize);

	int *GPU_CoverageMatrix;
	err = cudaMallocHost((void **)&GPU_CoverageMatrix, Coverage_MemSize);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate vector GPU_CoverageMatrix from device to host (error code %s)!\n", cudaGetErrorString(err));
		::exit(EXIT_FAILURE);
	}

	int *CPU_CoverageResult = (int *)malloc(CoverageResult_MemSize);
	float *GPU_TP, *GPU_FP, *GPU_TN, *GPU_FN;
	float *GPU_Precision, *GPU_TruePositiveRate, *GPU_TrueNegativeRate, *GPU_AccuracyRate, *GPU_Fitness_Value;
	float *d_GPU_TP, *d_GPU_FP, *d_GPU_TN, *d_GPU_FN;
	float *d_GPU_Precision, *d_GPU_TruePositiveRate, *d_GPU_TrueNegativeRate, *d_GPU_AccuracyRate, *d_GPU_Fitness_Value;
	size_t TF_PN_MemSize = InitRuleRows*NumofClass*sizeof(float);
	size_t Fitness_MemSize = InitRuleRows*NumofClass*sizeof(float);
	cudaMallocHost((void **)&GPU_TP, TF_PN_MemSize);
	cudaMallocHost((void **)&GPU_FP, TF_PN_MemSize);
	cudaMallocHost((void **)&GPU_TN, TF_PN_MemSize);
	cudaMallocHost((void **)&GPU_FN, TF_PN_MemSize);
	cudaMallocHost((void **)&GPU_Precision, Fitness_MemSize);
	cudaMallocHost((void **)&GPU_TruePositiveRate, Fitness_MemSize);
	cudaMallocHost((void **)&GPU_TrueNegativeRate, Fitness_MemSize);
	cudaMallocHost((void **)&GPU_AccuracyRate, Fitness_MemSize);
	cudaMallocHost((void **)&GPU_Fitness_Value, Fitness_MemSize);
	float *CPU_TP = (float *)malloc(TF_PN_MemSize);
	float *CPU_FP = (float *)malloc(TF_PN_MemSize);
	float *CPU_TN = (float *)malloc(TF_PN_MemSize);
	float *CPU_FN = (float *)malloc(TF_PN_MemSize);
	float *CPU_Precision = (float *)malloc(Fitness_MemSize);
	float *CPU_TruePositiveRate = (float *)malloc(Fitness_MemSize);
	float *CPU_TrueNegativeRate = (float *)malloc(Fitness_MemSize);
	float *CPU_AccuracyRate = (float *)malloc(Fitness_MemSize);
	float *CPU_Fitness_Value = (float *)malloc(Fitness_MemSize);

	cudaMalloc((void **)&d_GPU_TP, TF_PN_MemSize);
	cudaMalloc((void **)&d_GPU_FP, TF_PN_MemSize);
	cudaMalloc((void **)&d_GPU_TN, TF_PN_MemSize);
	cudaMalloc((void **)&d_GPU_FN, TF_PN_MemSize);
	cudaMalloc((void **)&d_GPU_Precision, Fitness_MemSize);
	cudaMalloc((void **)&d_GPU_TruePositiveRate, Fitness_MemSize);
	cudaMalloc((void **)&d_GPU_TrueNegativeRate, Fitness_MemSize);
	cudaMalloc((void **)&d_GPU_AccuracyRate, Fitness_MemSize);
	cudaMalloc((void **)&d_GPU_Fitness_Value, Fitness_MemSize);

	//Crossover Allocation

	int *SortedFitnessID = (int *)malloc(TF_PN_MemSize);
	float *FitnessSort = (float *)malloc(Fitness_MemSize);

	size_t AvgCoverage_Memsize = InitRuleRows*NumofClass*NumofAttr*sizeof(float);
	float *CPU_AvgCoverage = (float *)malloc(AvgCoverage_Memsize);

	int *Discovered_RC, *Discovered_RA;
	float *Discovered_RL, *Discovered_RU;
	int Discoverded_RI = Iter*NumofAttr*(InitRuleRows / 2)*NumofClass;
	size_t Discoverded_RI_MemSize = Discoverded_RI*sizeof(float);
	size_t Discoverded_RICond_MemSize = Discoverded_RI*sizeof(int);

	cudaMallocHost((void **)&Discovered_RC, Discoverded_RICond_MemSize);
	cudaMallocHost((void **)&Discovered_RL, Discoverded_RI_MemSize);
	cudaMallocHost((void **)&Discovered_RU, Discoverded_RI_MemSize);
	cudaMallocHost((void **)&Discovered_RA, Discoverded_RICond_MemSize);

	bool *MetChromosomes = (bool *)malloc(InitRuleRows*NumofClass*sizeof(bool));

	//Allocations of Testing Phase

	float *LastGenerationError = (float *)malloc(InitRuleRows*NumofClass*sizeof(float));

	//int rnd;
	//for (int i = 0;i < NumofClass;i++)
	//{
	//	for (int j = 0;j < 100;j++)
	//	{
	//		rnd = rand() % NumClass[i];
	//		//printf("iter = %i  |  rnd = %i\n", j, rnd);
	//		RuleInit(DataSet, Rule_Conditions, Rule_LowerBound, Rule_UpperBound, Rule_ActiveAttr, rnd);
	//	}
	//}

	SetMinMax(DataSet, MinValue, MaxValue, NumofAttr, NumofRows);
	for (int i = 0;i < InitialRuleSize*NumofRows;i++)
	{
		Coverage[i] = 8;
		CPUCoverage[i] = 8;
	}
	/*for (int i = 0;i < NumofAttr - 1;i++)
	{
	printf("Min[%i] = %.2f | Max[%i] = %.2f\n", i, MinValue[i], i, MaxValue[i]);
	}*/

	/*dim3 InitThreads(NumofAttr, 3);
	dim3 InitGrid(InitRuleRows, 1);

	RuleInit << <InitGrid, InitThreads >> >(d_Rule_Conditions, d_Rule_LowerBound, d_Rule_UpperBound, d_Rule_ActiveAttr, d_MinValue, d_MaxValue);

	cudaMemcpy(Rule_Conditions, d_Rule_Conditions, InitialRule_MemSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(Rule_LowerBound, d_Rule_LowerBound, InitialRule_MemSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(Rule_UpperBound, d_Rule_UpperBound, InitialRule_MemSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(Rule_ActiveAttr, d_Rule_ActiveAttr, InitialRule_MemSize, cudaMemcpyDeviceToHost);*/

	/*err = cudaEventCreate(&Coverage_Start);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(err));
		::exit(EXIT_FAILURE);
	}
	err = cudaEventCreate(&Coverage_Stop);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(err));
		::exit(EXIT_FAILURE);
	}

	err = cudaEventCreate(&Reduction_Start);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(err));
		::exit(EXIT_FAILURE);
	}
	err = cudaEventCreate(&Reduction_Stop);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(err));
		::exit(EXIT_FAILURE);
	}

	err = cudaEventCreate(&Fitness_Start);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(err));
		::exit(EXIT_FAILURE);
	}
	err = cudaEventCreate(&Fitness_Stop);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(err));
		::exit(EXIT_FAILURE);
	}*/

	::printf("Executing Training Phase..\n");
	::printf("Initializing Population..\n");
	CPU_StartTime = clock();

	CPU_InitPopulation_FirstGeneration(Rule_ActiveAttr, Rule_Conditions, Rule_LowerBound, Rule_UpperBound, MinValue, MaxValue, NumofAttr, InitRuleRows, NumofClass);

	CPU_StopTime = clock();

	::printf("done.\n");

	CPUInitialization_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));
	
	for (int GAIteration = 0;GAIteration < Iter;GAIteration++)
	{
		/*if (GAIteration == 0)
		{
			CPU_InitPopulation_FirstGeneration(Rule_ActiveAttr, Rule_Conditions, Rule_LowerBound, Rule_UpperBound, MinValue, MaxValue, NumofAttr, InitRuleRows, NumofClass);
		}
		else
		{
			CPU_InitPopulation(Rule_ActiveAttr, Rule_Conditions, Rule_LowerBound, Rule_UpperBound, MinValue, MaxValue, NumofAttr, InitRuleRows, NumofClass, MetChromosomes);
		}*/

		for (int c = 0;c < NumofClass;c++)
		{
			for (int r = 0;r < InitRuleRows;r++)
			{
				MetChromosomes[(c*InitRuleRows) + r] = false;
			}
		}

		if (TestMode == 2)
		{
			for (int c = 0;c < NumofClass;c++)
			{
				for (int i = 0;i < InitRuleRows;i++)
				{
					for (int j = 0;j < NumofAttr;j++)
					{
						::printf("ActiveAttr[%i] = %i\n", c*InitRuleRows*NumofAttr + (i*NumofAttr) + j, Rule_ActiveAttr[c*InitRuleRows*NumofAttr + (i*NumofAttr) + j]);
						::printf("Condition[%i] = %i\n", c*InitRuleRows*NumofAttr + (i*NumofAttr) + j, Rule_Conditions[c*InitRuleRows*NumofAttr + (i*NumofAttr) + j]);
						::printf("LowerBound[%i] = %.2f\n", c*InitRuleRows*NumofAttr + (i*NumofAttr) + j, Rule_LowerBound[c*InitRuleRows*NumofAttr + (i*NumofAttr) + j]);
						::printf("UpperBound[%i] = %.2f\n", c*InitRuleRows*NumofAttr + (i*NumofAttr) + j, Rule_UpperBound[c*InitRuleRows*NumofAttr + (i*NumofAttr) + j]);
						::printf("*********************************************\n");
					}
				}
			}
		}

		err = cudaMemcpy(d_DataSet, DataSet, MemSizeDataSet, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy device vector d_DataSet (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}
		err = cudaMemcpy(d_Rule_Conditions, Rule_Conditions, InitialRuleCond_MemSize, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy device vector d_Rule_Conditions (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}
		err = cudaMemcpy(d_Rule_LowerBound, Rule_LowerBound, InitialRule_MemSize, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy device vector d_Rule_LowerBound (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}
		err = cudaMemcpy(d_Rule_UpperBound, Rule_UpperBound, InitialRule_MemSize, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy device vector d_Rule_UpperBound (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}
		err = cudaMemcpy(d_Rule_ActiveAttr, Rule_ActiveAttr, InitialRuleCond_MemSize, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy device vector d_Rule_ActiveAttr (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}
		err = cudaMemcpy(d_Coverage, Coverage, Coverage_MemSize, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy device vector d_Coverage (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		/*err = cudaEventRecord(Coverage_Start, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}*/

		CPU_StartTime = clock();
		
		GPU_CoverageKernel << < BlocksPerGrid, ThreadsPerBlock >> > (d_DataSet, d_Coverage, d_Rule_ActiveAttr, d_Rule_Conditions, d_Rule_LowerBound, d_Rule_UpperBound, NumofRows, NumofAttr);
		err = cudaGetLastError();
		cudaDeviceSynchronize();
		
		CPU_StopTime = clock();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch GPU_CoverageKernel (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}
		/*err = cudaEventRecord(Coverage_Stop, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		err = cudaEventSynchronize(Coverage_Stop);

		err = cudaEventElapsedTime(&msecTemp, Coverage_Start, Coverage_Stop);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		GPUCoverage_msec += msecTemp;*/
		GPUCoverage_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));

		/*err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch GPU_CoverageKernel kernel (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}*/

		err = cudaMemcpy(Coverage, d_Coverage, Coverage_MemSize, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector Coverage from device to host (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		/*err = cudaEventRecord(Coverage_Start, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}*/

		CPU_StartTime = clock();

		CPU_CoverageFunction(DataSet, CPUCoverage, Rule_ActiveAttr, Rule_Conditions, Rule_LowerBound, Rule_UpperBound, NumofAttr, NumofRows, InitRuleRows*NumofClass);

		CPU_StopTime = clock();

		/*err = cudaEventRecord(Coverage_Stop, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		err = cudaEventSynchronize(Coverage_Stop);

		err = cudaEventElapsedTime(&msecTemp, Coverage_Start, Coverage_Stop);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		CPUCoverage_msec += msecTemp;*/
		CPUCoverage_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));


		for (int i = 0;i < InitialRuleSize*NumofRows;i++)
		{
			GPU_CoverageMatrix[i] = Coverage[i];
			CPU_CoverageMatrix[i] = CPUCoverage[i];
		}

		//int cnt = 0;
		//for (int i = 0;i < InitialRuleSize*NumofRows;i++)
		//{
		//	if (Coverage[i] == 1)
		//	{
		//		//printf("Coverage[%i] = %i\n", i, Coverage[i]);
		//		cnt++;
		//	}
		//}
		//printf("# of elements in Coverage = %i\n", InitialRuleSize*NumofRows);
		//printf("# of covered elements = %i\n", cnt);
		//int rndData, rndRule;
		//int sep = 1;
		int cnt = 0;
		if (TestMode == 3)
		{
			//Test Start

			//for (int i = 0;i < 1;i++)
			//{
			//	/*rndData = rand() % NumofRows;
			//	rndRule = rand() % InitRuleRows*NumofClass;*/
			//	rndData = 1;
			//	rndRule = 0;
			//	printf("Selected instance #%i of DataSet and chromosome #%i of intialized rules\n", rndData, rndRule);
			//	for (int j = 0;j < NumofAttr;j++)
			//	{
			//		printf("E = %i |", (rndRule*NumofAttr*NumofRows) + (rndData*NumofAttr) + j);
			//		printf(" D = %.2f |", DataSet[(rndData*NumofAttr) + j]);
			//		printf(" RA = %i |", Rule_ActiveAttr[(rndRule*NumofAttr) + j]);
			//		printf(" RC = %i |", Rule_Conditions[(rndRule*NumofAttr) + j]);
			//		printf(" RL = %.2f |", Rule_LowerBound[(rndRule*NumofAttr) + j]);
			//		printf(" RU = %.2f |", Rule_UpperBound[(rndRule*NumofAttr) + j]);
			//		printf(" C = %i |", CPU_CoverageMatrix[(rndRule*NumofAttr*NumofRows) + (rndData*NumofAttr) + j]);
			//		printf(" C = %i |", GPU_CoverageMatrix[(rndRule*NumofAttr*NumofRows) + (rndData*NumofAttr) + j]);
			//		printf("\n");
			//	}
			//	printf("*********************************************\n");
			//}

			for (int i = 0;i < InitialRuleSize*NumofRows;i++)
			{
				/*if ((sep*NumofAttr) <= i)
				{
					::printf("Passed Row #%i\n", sep);
					sep++;
				}*/
				if (GPU_CoverageMatrix[i] != CPU_CoverageMatrix[i])
				{
					::printf("GPU_CoverageMatrix[%i] != CPU_CoverageMatrix[%i]\n", i, i);
					cnt++;
				}
			}
			::printf("Iter #%i: # of miss-matches in CoverageMatrix = %i\n", GAIteration, cnt);

			//Test Stop
		}

		/*for (int i = 0;i < InitialRuleSize*NumofRows;i++)
		{
			Coverage[i] = CPUCoverage[i];
		}*/
		err = cudaMemcpy(d_Coverage, Coverage, Coverage_MemSize, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector Coverage from host to device (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		/*err = cudaEventRecord(Reduction_Start, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}*/

		CPU_StartTime = clock();

		GPU_CoverageReduction << <BPG, TPB >> >(d_Coverage, NumofAttr, NumofRows, InitRuleRows*NumofClass);
		err = cudaGetLastError();
		cudaDeviceSynchronize();

		CPU_StopTime = clock();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch GPU_CoverageReduction (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}
		/*err = cudaEventRecord(Reduction_Stop, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		err = cudaEventSynchronize(Reduction_Stop);

		err = cudaEventElapsedTime(&msecTemp, Reduction_Start, Reduction_Stop);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		GPUReduction_msec += msecTemp;*/
		GPUReduction_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));

		err = cudaMemcpy(Coverage, d_Coverage, Coverage_MemSize, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector Coverage from device to host (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		/*err = cudaEventRecord(Reduction_Start, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}*/

		CPU_StartTime = clock();

		CPU_CoverageReduction(CPUCoverage, NumofAttr, NumofRows, InitRuleRows*NumofClass);

		CPU_StopTime = clock();

		/*err = cudaEventRecord(Reduction_Stop, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		err = cudaEventSynchronize(Reduction_Stop);

		err = cudaEventElapsedTime(&msecTemp, Reduction_Start, Reduction_Stop);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		CPUReduction_msec += msecTemp;*/
		CPUReduction_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));

		for (int i = 0, j = 0;i < InitialRuleSize*NumofRows;i += NumofAttr, j++)
		{
			GPU_CoverageResult[j] = Coverage[i];
			CPU_CoverageResult[j] = CPUCoverage[i];
		}

		if (TestMode == 4)
		{
			cnt = 0;
			for (int i = 0;i < InitRuleRows*NumofClass*NumofRows;i++)
			{
				if (GPU_CoverageResult[i] != CPU_CoverageResult[i])
				{
					::printf("Passed Row #%i\n", i);
					::printf("GPU_CoverageResult[%i] != CPU_CoverageResult[%i] | %i != %i\n", i, i, GPU_CoverageResult[i], CPU_CoverageResult[i]);
					cnt++;
				}
				/*else
				{
				printf("Passed Row #%i\n", i);
				printf("GPU_CoverageResult[%i] = CPU_CoverageResult[%i] | %i = %i\n", i, i, GPU_CoverageResult[i], CPU_CoverageResult[i]);
				}*/
			}
			::printf("# of miss-matches in CoverageResults = %i\n", cnt);
		}

		for (int i = 0;i < InitRuleRows*NumofClass;i++)
		{
			GPU_TP[i] = 0;
			GPU_FP[i] = 0;
			GPU_TN[i] = 0;
			GPU_FN[i] = 0;
			GPU_Precision[i] = 0;
			GPU_TruePositiveRate[i] = 0;
			GPU_TrueNegativeRate[i] = 0;
			GPU_AccuracyRate[i] = 0;
			GPU_Fitness_Value[i] = 0;

			CPU_TP[i] = 0;
			CPU_FP[i] = 0;
			CPU_TN[i] = 0;
			CPU_FN[i] = 0;
			CPU_Precision[i] = 0;
			CPU_TruePositiveRate[i] = 0;
			CPU_TrueNegativeRate[i] = 0;
			CPU_AccuracyRate[i] = 0;
			CPU_Fitness_Value[i] = 0;
		}

		err = cudaMemcpy(d_GPU_CoverageResult, GPU_CoverageResult, CoverageResult_MemSize, cudaMemcpyHostToDevice);
		err = cudaMemcpy(d_Class, Class, ClassMemSize, cudaMemcpyHostToDevice);
		err = cudaMemcpy(d_GPU_TP, GPU_TP, TF_PN_MemSize, cudaMemcpyHostToDevice);
		err = cudaMemcpy(d_GPU_FP, GPU_FP, TF_PN_MemSize, cudaMemcpyHostToDevice);
		err = cudaMemcpy(d_GPU_TN, GPU_TN, TF_PN_MemSize, cudaMemcpyHostToDevice);
		err = cudaMemcpy(d_GPU_FN, GPU_FN, TF_PN_MemSize, cudaMemcpyHostToDevice);

		err = cudaMemcpy(d_GPU_Precision, GPU_Precision, Fitness_MemSize, cudaMemcpyHostToDevice);
		err = cudaMemcpy(d_GPU_TruePositiveRate, GPU_TruePositiveRate, Fitness_MemSize, cudaMemcpyHostToDevice);
		err = cudaMemcpy(d_GPU_TrueNegativeRate, GPU_TrueNegativeRate, Fitness_MemSize, cudaMemcpyHostToDevice);
		err = cudaMemcpy(d_GPU_Fitness_Value, GPU_Fitness_Value, Fitness_MemSize, cudaMemcpyHostToDevice);

		/*err = cudaEventRecord(Fitness_Start, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}*/

		CPU_StartTime = clock();

		GPU_Fitness << <BPG_Fitness, TPB_Fitness >> >(d_GPU_CoverageResult, d_Class, NumofRows, NumofAttr, NumofClass, InitRuleRows, d_GPU_TP, d_GPU_FP, d_GPU_TN, d_GPU_FN, d_GPU_Precision, d_GPU_TruePositiveRate, d_GPU_TrueNegativeRate, d_GPU_AccuracyRate, d_GPU_Fitness_Value);
		err = cudaGetLastError();
		cudaDeviceSynchronize();

		CPU_StopTime = clock();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch GPU_Fitness (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}
		/*err = cudaEventRecord(Fitness_Stop, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		err = cudaEventSynchronize(Fitness_Stop);

		err = cudaEventElapsedTime(&msecTemp, Fitness_Start, Fitness_Stop);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		GPUFitness_msec += msecTemp;*/
		GPUFitness_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));

		err = cudaMemcpy(GPU_TP, d_GPU_TP, TF_PN_MemSize, cudaMemcpyDeviceToHost);
		err = cudaMemcpy(GPU_FP, d_GPU_FP, TF_PN_MemSize, cudaMemcpyDeviceToHost);
		err = cudaMemcpy(GPU_TN, d_GPU_TN, TF_PN_MemSize, cudaMemcpyDeviceToHost);
		err = cudaMemcpy(GPU_FN, d_GPU_FN, TF_PN_MemSize, cudaMemcpyDeviceToHost);

		err = cudaMemcpy(GPU_Precision, d_GPU_Precision, Fitness_MemSize, cudaMemcpyDeviceToHost);
		err = cudaMemcpy(GPU_TruePositiveRate, d_GPU_TruePositiveRate, Fitness_MemSize, cudaMemcpyDeviceToHost);
		err = cudaMemcpy(GPU_TrueNegativeRate, d_GPU_TrueNegativeRate, Fitness_MemSize, cudaMemcpyDeviceToHost);
		err = cudaMemcpy(GPU_Fitness_Value, d_GPU_Fitness_Value, Fitness_MemSize, cudaMemcpyDeviceToHost);

		/*err = cudaEventRecord(Fitness_Start, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}*/

		CPU_StartTime = clock();

		CPU_Fitness(CPU_CoverageResult, Class, NumofRows, NumofAttr, NumofClass, InitRuleRows, CPU_TP, CPU_FP, CPU_TN, CPU_FN, CPU_Precision, CPU_TruePositiveRate, CPU_TrueNegativeRate, CPU_AccuracyRate, CPU_Fitness_Value);

		CPU_StopTime = clock();

		/*err = cudaEventRecord(Fitness_Stop, NULL);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		err = cudaEventSynchronize(Fitness_Stop);

		err = cudaEventElapsedTime(&msecTemp, Fitness_Start, Fitness_Stop);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(err));
			::exit(EXIT_FAILURE);
		}

		CPUFitness_msec += msecTemp;*/
 		CPUFitness_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));

		if (TestMode == 5)
		{
			for (int i = 0;i < InitRuleRows*NumofClass;i++)
			{
				if (GPU_TP[i] != CPU_TP[i])
				{
					::printf("GPU_TP[%i] != CPU_TP[%i] | %.0f != %.0f\n", i, i, GPU_TP[i], CPU_TP[i]);
				}
				else
				{
					::printf("GPU_TP[%i] = CPU_TP[%i] | %.0f = %.0f\n", i, i, GPU_TP[i], CPU_TP[i]);
				}
				if (GPU_FP[i] != CPU_FP[i])
				{
					::printf("GPU_FP[%i] != CPU_FP[%i] | %.0f != %.0f\n", i, i, GPU_FP[i], CPU_FP[i]);
				}
				else
				{
					::printf("GPU_FP[%i] = CPU_FP[%i] | %.0f = %.0f\n", i, i, GPU_FP[i], CPU_FP[i]);
				}
				if (GPU_TN[i] != CPU_TN[i])
				{
					::printf("GPU_TN[%i] != CPU_TN[%i] | %.0f != %.0f\n", i, i, GPU_TN[i], CPU_TN[i]);
				}
				else
				{
					::printf("GPU_TN[%i] = CPU_TN[%i] | %.0f = %.0f\n", i, i, GPU_TN[i], CPU_TN[i]);
				}
				if (GPU_FN[i] != CPU_FN[i])
				{
					::printf("GPU_FN[%i] != CPU_FN[%i] | %.0f != %.0f\n", i, i, GPU_FN[i], CPU_FN[i]);
				}
				else
				{
					::printf("GPU_FN[%i] = CPU_FN[%i] | %.0f = %.0f\n", i, i, GPU_FN[i], CPU_FN[i]);
				}
			}
		}

		if (TestMode == 6)
		{
			for (int i = 0;i < InitRuleRows*NumofClass;i++)
			{
				if (GPU_Precision[i] != CPU_Precision[i])
				{
					::printf("GPU_Precision[%i] != CPU_Precision[%i] | %.2f != %.2f\n", i, i, GPU_Precision[i], CPU_Precision[i]);
				}
				else
				{
					::printf("GPU_Precision[%i] = CPU_Precision[%i] | %.2f = %.2f\n", i, i, GPU_Precision[i], CPU_Precision[i]);
				}
				if (GPU_TruePositiveRate[i] != CPU_TruePositiveRate[i])
				{
					::printf("GPU_TruePositiveRate[%i] != CPU_TruePositiveRate[%i] | %.2f != %.2f\n", i, i, GPU_TruePositiveRate[i], CPU_TruePositiveRate[i]);
				}
				else
				{
					::printf("GPU_TruePositiveRate[%i] = CPU_TruePositiveRate[%i] | %.2f = %.2f\n", i, i, GPU_TruePositiveRate[i], CPU_TruePositiveRate[i]);
				}
				if (GPU_TrueNegativeRate[i] != CPU_TrueNegativeRate[i])
				{
					::printf("GPU_TrueNegativeRate[%i] != CPU_TrueNegativeRate[%i] | %.2f != %.2f\n", i, i, GPU_TrueNegativeRate[i], CPU_TrueNegativeRate[i]);
				}
				else
				{
					::printf("GPU_TrueNegativeRate[%i] = CPU_TrueNegativeRate[%i] | %.2f = %.2f\n", i, i, GPU_TrueNegativeRate[i], CPU_TrueNegativeRate[i]);
				}
				if (GPU_AccuracyRate[i] != CPU_AccuracyRate[i])
				{
					::printf("GPU_AccuracyRate[%i] != CPU_AccuracyRate[%i] | %.2f != %.2f\n", i, i, GPU_AccuracyRate[i], CPU_AccuracyRate[i]);
				}
				else
				{
					::printf("GPU_AccuracyRate[%i] = CPU_AccuracyRate[%i] | %.2f = %.2f\n", i, i, GPU_AccuracyRate[i], CPU_AccuracyRate[i]);
				}
				if (GPU_Fitness_Value[i] != CPU_Fitness_Value[i])
				{
					::printf("GPU_Fitness_Value[%i] != CPU_Fitness_Value[%i] | %.2f != %.2f\n", i, i, GPU_Fitness_Value[i], CPU_Fitness_Value[i]);
				}
				else
				{
					::printf("GPU_Fitness_Value[%i] = CPU_Fitness_Value[%i] | %.2f = %.2f\n", i, i, GPU_Fitness_Value[i], CPU_Fitness_Value[i]);
				}
			}
		}

		for (int i = 0;i < InitRuleRows*NumofClass;i++)
		{
			FitnessSort[i] = CPU_Fitness_Value[i];
		}
		for (int c = 0;c < NumofClass;c++)
		{
			thrust::sort(FitnessSort + (c*InitRuleRows), FitnessSort + ((c + 1)*InitRuleRows));
		}

		CPU_RuleSelection(CPU_Fitness_Value, FitnessSort, SortedFitnessID, NumofClass, InitRuleRows);

		if (TestMode == 7)
		{
			for (int i = 0;i < InitRuleRows*NumofClass;i++)
			{
				::printf("FitnessSort[%i] = %.2f | CPU_Fitness_Value[%i]\n", i, FitnessSort[i], SortedFitnessID[i]);
			}
		}

		for (int i = 0;i < InitRuleRows*NumofClass;i++)
		{
			for (int j = 0;j < NumofAttr;j++)
			{
				CPU_AvgCoverage[(i*NumofAttr) + j] = 0;
			}
		}

		CPU_StartTime = clock();

		CPU_AverageCoverage(CPU_CoverageMatrix, CPU_AvgCoverage, NumClass, NumofClass, InitRuleRows, NumofRows, NumofAttr);

		CPU_StopTime = clock();

		CPUAVGCoverage_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));

		if (TestMode == 8)
		{
			::printf("Attribute ID =         |");
			for (int i = 0;i < NumofAttr;i++)
			{
				if (i < 10)
				{
					::printf("__%i_|", i);
				}
				else
				{
					::printf("_%i_|", i);
				}
			}
			::printf("\n");
			for (int i = 0;i < InitRuleRows*NumofClass;i++)
			{
				if (i < 10)
				{
					::printf("CPU_AvgCoverage[%i] =   |", i);
				}
				else
				{
					if (i < 100)
					{
						::printf("CPU_AvgCoverage[%i] =  |", i);
					}
				}
				if (i >= 100)
				{
					::printf("CPU_AvgCoverage[%i] = |", i);
				}
				for (int j = 0;j < NumofAttr;j++)
				{
					::printf("%.2f|", CPU_AvgCoverage[(i*NumofAttr) + j]);
				}
				::printf("\n");
			}
		}
		CPU_StartTime = clock();

		CPU_Crossover(CPU_AvgCoverage, SortedFitnessID, MetChromosomes, GAIteration, NumofClass, InitRuleRows, NumofAttr, Rule_ActiveAttr, Rule_Conditions, Rule_LowerBound, Rule_UpperBound, Discovered_RA, Discovered_RC, Discovered_RL, Discovered_RU);

		CPU_StopTime = clock();

		CPUCrossover_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));

		CPU_StartTime = clock();

		CPU_Mutation(CPU_AvgCoverage, MetChromosomes, MinValue, MaxValue, NumofClass, InitRuleRows, NumofAttr, Rule_ActiveAttr, Rule_Conditions, Rule_LowerBound, Rule_UpperBound);

		CPU_StopTime = clock();

		CPUMutation_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));
	}
	
	::printf("\nTraining is complete.\n");

	DataSetStr = argv[8];
	ClassStr = argv[9];

	/*if (checkCmdLineFlag(argc, (const char **)argv, "TED"))
	{
		DataSetStr = getCmdLineArgumentInt(argc, (const char **)argv, "TED");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "TEC"))
	{
		ClassStr = getCmdLineArgumentInt(argc, (const char **)argv, "TEC");
	}*/

	/*vector<string> TestDataSet_Row;
	vector<string> TestClass_Row;*/
	DataSet_Row.empty();
	DataSet_Row.clear();
	Class_Row.empty();
	Class_Row.clear();
	NumofRows = (read_DataSet(ClassStr, Class_Row));
	//printf("Rows = %i\n", NumofRows);
	if (NumofRows != (read_DataSet(DataSetStr, DataSet_Row)))
	{
		::exit(EXIT_FAILURE);
	}

	float *TestDataSet;
	int *TestClass;
	
	SizeofDataSet = NumofAttr*NumofRows;
	MemSizeDataSet = sizeof(float)*SizeofDataSet;
	ClassMemSize = sizeof(int)*NumofRows;
	cudaMallocHost((void **)&TestDataSet, MemSizeDataSet);
	cudaMallocHost((void **)&TestClass, ClassMemSize);

	/*vector<float> TestAttrVec;
	vector<int> TestClassVec;*/
	AttrVec.empty();
	AttrVec.clear();
	ClassVec.empty();
	ClassVec.clear();
	for (int i = 0; i <= NumofRows; i++)
	{
		stringstream StrStm3(DataSet_Row[i]);
		for (float j; StrStm3 >> j;)
		{
			AttrVec.push_back(j);
			if (StrStm3.peek() == ',')
			{
				StrStm3.ignore();
			}
		}
	}

	//printf("%i , %i\n", SizeofDataSet, AttrVec.size());
	for (int i = 0;i < SizeofDataSet;i++)
	{
		TestDataSet[i] = AttrVec[i];
	}

	for (int i = 0; i <= NumofRows; i++)
	{
		stringstream StrStm4(Class_Row[i]);
		for (int j; StrStm4 >> j;)
		{
			ClassVec.push_back(j);
			if (StrStm4.peek() == ',')
			{
				StrStm4.ignore();
			}
		}
	}

	for (int i = 0;i < NumofRows;i++)
	{
		TestClass[i] = ClassVec[i];
	}

	for (int i = 0;i < NumofClass;i++)
	{
		NumClass[i] = 0;
	}

	NumofEachClass(NumClass, NumofRows, TestClass);

	::printf("\nExecuting Testing Phase..\n");
	if (TestMode == 9)
	{
		::printf("i&a|");
		for (int a = 0;a < NumofAttr;a++)
		{
			::printf("%i|", a);
		}
		::printf("\n");

		for (int i = 0;i < NumofRows;i++)
		{
			::printf(" %i |", i);
			for (int a = 0;a < NumofAttr;a++)
			{
				::printf("%.0f|", TestDataSet[(i*NumofAttr) + a]);
			}
			::printf("\n");
		}

	}
	for (int i = 0;i < NumofClass;i++)
	{
		::printf("Number of Instances of Class %i = %i\n", i, NumClass[i]);
	}
	::printf("*********************************************\n");

	InitRuleRows = InitRuleRows / 2;

	CPU_StartTime = clock();

	TestLastGeneration(LastGenerationError, TestDataSet, TestClass, NumClass, Iter, NumofClass, InitRuleRows, NumofRows, NumofAttr, Discovered_RA, Discovered_RC, Discovered_RL, Discovered_RU);

	CPU_StopTime = clock();

	CPUTest_msec += (CPU_StopTime - CPU_StartTime) / ((double)(CLOCKS_PER_SEC));

	::printf("Chromosome #X of Class #Y        Error\n");
	for (int c = 0;c < NumofClass;c++)
	{
		for (int r = 0;r < InitRuleRows;r++)
		{
			::printf("Chromosome #%i of Class %i        %.2f\n", r, c, LastGenerationError[(c*InitRuleRows) + r]);
		}
	}

	if (TestMode == 10)
	{
		for (int c = 0;c < NumofClass;c++)
		{
			for (int r = 0;r < InitRuleRows;r++)
			{
				::printf("Chromosome #%i of Class #%i\n|", r, c);
				for (int a = 0;a < NumofAttr;a++)
				{
					::printf("|%i: %i_%i_%.2f_%.2f|", a, Discovered_RA[((Iter - 1)*NumofClass*(InitRuleRows / 2)*NumofAttr) + (c*InitRuleRows*NumofAttr) + (r*NumofAttr) + a], Discovered_RC[((Iter - 1)*NumofClass*(InitRuleRows / 2)*NumofAttr) + (c*InitRuleRows*NumofAttr) + (r*NumofAttr) + a], Discovered_RL[((Iter - 1)*NumofClass*(InitRuleRows / 2)*NumofAttr) + (c*InitRuleRows*NumofAttr) + (r*NumofAttr) + a], Discovered_RU[((Iter - 1)*NumofClass*(InitRuleRows / 2)*NumofAttr) + (c*InitRuleRows*NumofAttr) + (r*NumofAttr) + a]);
					/*::printf("ActiveAttr[%i][%i][%i] = %i\n", c, r, a, Discovered_RA[((Iter - 1)*NumofClass*(InitRuleRows / 2)*NumofAttr) + (c*InitRuleRows*NumofAttr) + (r*NumofAttr) + a]);
					::printf("Condition[%i][%i][%i] = %i\n", c, r, a, Discovered_RC[((Iter - 1)*NumofClass*(InitRuleRows / 2)*NumofAttr) + (c*InitRuleRows*NumofAttr) + (r*NumofAttr) + a]);
					::printf("LowerBound[%i][%i][%i] = %.2f\n", c, r, a, Discovered_RL[((Iter - 1)*NumofClass*(InitRuleRows / 2)*NumofAttr) + (c*InitRuleRows*NumofAttr) + (r*NumofAttr) + a]);
					::printf("UpperBound[%i][%i][%i] = %.2f\n", c, r, a, Discovered_RU[((Iter - 1)*NumofClass*(InitRuleRows / 2)*NumofAttr) + (c*InitRuleRows*NumofAttr) + (r*NumofAttr) + a]);*/
				}
				::printf("|\n*********************************************\n");
			}
		}
	}
	::printf("Analysis of GPU execution time:\n");
	::printf("Execution time of CoverageKernel = %.2f msec\n", (((float)GPUCoverage_msec / Iter) * 1000));
	::printf("Execution time of ReductionKernel = %.2f msec\n", (((float)GPUReduction_msec / Iter) * 1000));
	::printf("Execution time of FitnessKernel = %.2f msec\n", (((float)GPUFitness_msec / Iter) * 1000));
	::printf("*********************************************\n");
	::printf("Analysis of CPU execution time:\n");
	::printf("Execution time of CPU_Initialization = %.2f msec\n", (((float)CPUInitialization_msec / Iter) * 1000));
	::printf("Execution time of CPU_Coverage = %.2f msec\n", (((float)CPUCoverage_msec / Iter) * 1000));
	::printf("Execution time of CPU_Reduction = %.2f msec\n", (((float)CPUReduction_msec / Iter) * 1000));
	::printf("Execution time of CPU_Fitness = %.2f msec\n", (((float)CPUFitness_msec / Iter) * 1000));
	::printf("Execution time of CPU_AVGCoverage = %.2f msec\n", (((float)CPUAVGCoverage_msec / Iter) * 1000));
	::printf("Execution time of CPU_Crossover = %.4f msec\n", (((float)CPUCrossover_msec / Iter) * 1000));
	::printf("Execution time of CPU_Mutation = %.4f msec\n", (((float)CPUMutation_msec / Iter) * 1000));
	::printf("Execution time of CPU_Test = %.2f msec\n", (((float)CPUTest_msec / Iter) * 1000));
	::printf("*********************************************\n");
	::printf("Speedup = (GPU time / CPU time):\n");
	::printf("CoverageKernel Speedup = %.2f\n", ((float)CPUCoverage_msec / GPUCoverage_msec));
	::printf("ReductionKernel Speedup = %.2f\n", ((float)CPUReduction_msec / GPUReduction_msec));
	::printf("FitnessKernel Speedup = %.2f\n", ((float)CPUFitness_msec / GPUFitness_msec));

	//Destroy CUDA events

	/*cudaEventDestroy(Coverage_Start);
	cudaEventDestroy(Coverage_Stop);
	cudaEventDestroy(Reduction_Start);
	cudaEventDestroy(Reduction_Stop);
	cudaEventDestroy(Fitness_Start);
	cudaEventDestroy(Fitness_Stop);*/

	//Clear Device Memory
	
	cudaFree(d_DataSet);
	cudaFree(d_Coverage);
	cudaFree(d_Rule_Conditions);
	cudaFree(d_Rule_LowerBound);
	cudaFree(d_Rule_UpperBound);
	cudaFree(d_Rule_ActiveAttr);
	cudaFree(d_GPU_TP);
	cudaFree(d_GPU_FP);
	cudaFree(d_GPU_TN);
	cudaFree(d_GPU_FN);
	cudaFree(d_GPU_Precision);
	cudaFree(d_GPU_TruePositiveRate);
	cudaFree(d_GPU_TrueNegativeRate);
	cudaFree(d_GPU_AccuracyRate);
	cudaFree(d_GPU_Fitness_Value);
	
	//Clear Host Memory
	cudaFreeHost(DataSet);
	cudaFreeHost(Coverage);
	cudaFreeHost(MinValue);
	cudaFreeHost(MaxValue);
	cudaFreeHost(GPU_CoverageMatrix);
	cudaFreeHost(GPU_CoverageResult);
	cudaFreeHost(Rule_Conditions);
	cudaFreeHost(Rule_LowerBound);
	cudaFreeHost(Rule_UpperBound);
	cudaFreeHost(Rule_ActiveAttr);
	cudaFreeHost(GPU_TP);
	cudaFreeHost(GPU_FP);
	cudaFreeHost(GPU_TN);
	cudaFreeHost(GPU_FN);
	cudaFreeHost(GPU_Precision);
	cudaFreeHost(GPU_TruePositiveRate);
	cudaFreeHost(GPU_TrueNegativeRate);
	cudaFreeHost(GPU_AccuracyRate);
	cudaFreeHost(GPU_Fitness_Value);
	cudaFreeHost(TestDataSet);
	cudaFreeHost(TestClass);

	/*free(CPU_newRA);
	free(CPU_newRC);
	free(CPU_newRL);
	free(CPU_newRU);*/
	::free(CPUCoverage);
	::free(CPU_CoverageMatrix);
	::free(CPU_CoverageResult);
	::free(CPU_TP);
	::free(CPU_FP);
	::free(CPU_TN);
	::free(CPU_FN);
	::free(CPU_Precision);
	::free(CPU_TruePositiveRate);
	::free(CPU_TrueNegativeRate);
	::free(CPU_AccuracyRate);
	::free(CPU_Fitness_Value);
	::free(SortedFitnessID);
	::free(FitnessSort);
	::free(CPU_AvgCoverage);
	::free(MetChromosomes);
	::free(LastGenerationError);
	
	//Exit GA
	::exit(EXIT_SUCCESS);
}

