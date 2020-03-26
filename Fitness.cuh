//GPU Code

__global__ void GPU_CoverageKernel(float *DS, int *Coverage, int *RA, int *RC, float *RL, float *RU, int RN, int Attr)
{
	int Row = ((blockIdx.y*blockDim.y) + threadIdx.y);
	int Col = ((blockIdx.x*blockDim.x) + threadIdx.x);
	int ChromosomeID = blockIdx.z;

	if ((Row < RN) && (Col < Attr))
	{
		if (RA[ChromosomeID*Attr + Col] == 1)
		{
			switch (RC[ChromosomeID*Attr + Col])
			{
			case 0:
				if (DS[Row*Attr + Col] <= RL[ChromosomeID*Attr + Col])
				{
					Coverage[ChromosomeID*RN*Attr + Row*Attr + Col] = 1;
				}
				else
				{
					Coverage[ChromosomeID*RN*Attr + Row*Attr + Col] = 0;
				}
				break;
			case 1:
				if (RU[ChromosomeID*Attr + Col] <= DS[Row*Attr + Col])
				{
					Coverage[ChromosomeID*RN*Attr + Row*Attr + Col] = 1;
				}
				else
				{
					Coverage[ChromosomeID*RN*Attr + Row*Attr + Col] = 0;
				}
				break;
			case 2:
				if ((DS[Row*Attr + Col] >= RL[ChromosomeID*Attr + Col]) && (RU[ChromosomeID*Attr + Col] >= DS[Row*Attr + Col]))
				{
					Coverage[ChromosomeID*RN*Attr + Row*Attr + Col] = 1;
				}
				else
				{
					Coverage[ChromosomeID*RN*Attr + Row*Attr + Col] = 0;
				}
				break;
			default:
				break;
			}
		}
		else
		{
			Coverage[ChromosomeID*RN*Attr + Row*Attr + Col] = 1;
		}
	}
}

__global__ void GPU_CoverageReduction(int *Coverage, int Attr, int RN, int RR)
{
	int Row = (blockIdx.x*blockDim.x) + threadIdx.x;
	int Col = (blockIdx.y*blockDim.y) + threadIdx.y;

	if ((Row < RN*RR) && (Col < Attr))
	{
		for (int a = 0;a <= log2f(Attr);a++)
		{
			if ((2 * (int)powf(2, a)*Col) + (int)powf(2, a) < Attr)
			{
				Coverage[Row*Attr + (2 * (int)powf(2, a)*Col)] += Coverage[Row*Attr + (2 * (int)powf(2, a)*Col) + (int)powf(2, a)];
			}
			__syncthreads();
		}
	}
}

__global__ void GPU_Fitness(int *Coverage, int *Class, int RN, int Attr, int CN, int RI, float *TP, float *FP, float *TN, float *FN, float *Precision, float *TruePositiveRate, float *TrueNegativeRate, float *AccuracyRate, float *Fitness_Value)
{
	int Instance = (blockIdx.y*blockDim.y) + threadIdx.y;
	int ChromosomeID = blockIdx.z*blockDim.z + threadIdx.z;
	if (Instance < RN)
	{
		if (blockIdx.z == Class[Instance])
		{
			if (Coverage[(ChromosomeID*RN) + Instance] == Attr)
			{
				atomicAdd(&TP[ChromosomeID], 1);
			}
			else
			{
				atomicAdd(&FN[ChromosomeID], 1);
			}
		}
		else
		{
			if (Coverage[(ChromosomeID*RN) + Instance] == Attr)
			{
				atomicAdd(&FP[ChromosomeID], 1);
			}
			else
			{
				atomicAdd(&TN[ChromosomeID], 1);
			}
		}
	}
	__syncthreads();
	if ((TP[ChromosomeID] == 0) && (FP[ChromosomeID] == 0))
	{
		Precision[ChromosomeID] = 0;
	}
	else
	{
		Precision[ChromosomeID] = TP[ChromosomeID] / (TP[ChromosomeID] + FP[ChromosomeID]);
	}
	TruePositiveRate[ChromosomeID] = TP[ChromosomeID] / (TP[ChromosomeID] + FN[ChromosomeID]);
	if ((TN[ChromosomeID] == 0) && (FP[ChromosomeID] == 0))
	{
		TrueNegativeRate[ChromosomeID] = 0;
	}
	else
	{
		TrueNegativeRate[ChromosomeID] = TN[ChromosomeID] / (TN[ChromosomeID] + FP[ChromosomeID]);
	}
	AccuracyRate[ChromosomeID] = (TP[ChromosomeID] + TN[ChromosomeID]) / (TP[ChromosomeID] + TN[ChromosomeID] + FP[ChromosomeID] + FN[ChromosomeID]);
	Fitness_Value[ChromosomeID] = Precision[ChromosomeID] * TruePositiveRate[ChromosomeID];
}

// CPU Code

void CPU_CoverageFunction(float *DS, int *CPUCoverage, int *RA, int *RC, float *RL, float *RU, int Attr, int RN, int RR)
{
	for (int r = 0;r < RR;r++)
	{
		for (int d = 0;d < RN;d++)
		{
			for (int a = 0;a < Attr;a++)
			{
				if (RA[(r*Attr) + a] == 1)
				{
					switch (RC[(r*Attr) + a])
					{
					case 0:
						if (DS[(d*Attr) + a] <= RL[(r*Attr) + a])
						{
							CPUCoverage[(r*RN*Attr) + (d*Attr) + a] = 1;
						}
						else
						{
							CPUCoverage[(r*RN*Attr) + (d*Attr) + a] = 0;
						}
						break;
					case 1:
						if (RU[(r*Attr) + a] <= DS[(d*Attr) + a])
						{
							CPUCoverage[(r*RN*Attr) + (d*Attr) + a] = 1;
						}
						else
						{
							CPUCoverage[(r*RN*Attr) + (d*Attr) + a] = 0;
						}
						break;
					case 2:
						if ((DS[(d*Attr) + a] >= RL[(r*Attr) + a]) && (RU[(r*Attr) + a] >= DS[(d*Attr) + a]))
						{
							CPUCoverage[(r*RN*Attr) + (d*Attr) + a] = 1;
						}
						else
						{
							CPUCoverage[(r*RN*Attr) + (d*Attr) + a] = 0;
						}
						break;
					default:
						break;
					}
				}
				else
				{
					CPUCoverage[(r*RN*Attr) + (d*Attr) + a] = 1;
				}
			}
		}
	}
}

void CPU_CoverageReduction(int *Coverage, int Attr, int RN, int RR)
{
	int Stride, Row;
	for (int r = 0;r < RR;r++)
	{
		Stride = r*Attr*RN;
		for (int i = 0;i < RN;i++)
		{
			Row = i*Attr;
			for (int a = 0;a <= log2(Attr);a++)
			{
				for (int t = 0;t < Attr;t++)
				{
					if ((2 * (int)pow(2, a)*t) + (int)pow(2, a) < Attr)
					{
						Coverage[Stride + Row + (2 * (int)pow(2, a)*t)] += Coverage[Stride + Row + (2 * (int)pow(2, a)*t) + (int)pow(2, a)];
						//::printf("CPUCoverage[%i] += CPUCoverage[%i]\n", Stride + Row + (2 * (int)pow(2, a)*t), Stride + Row + (2 * (int)pow(2, a)*t) + (int)pow(2, a));
					}
				}
			}
			//::printf("CPUCoverage[%i] = %i\n", Stride + Row, Coverage[Stride + Row]);
		}
	}
}

void CPU_Fitness(int *Coverage, int *Class, int RN, int Attr, int CN, int RI, float *TP, float *FP, float *TN, float *FN, float *Precision, float *TruePositiveRate, float *TrueNegativeRate, float *AccuracyRate, float *Fitness_Value)
{
	for (int c = 0;c < CN;c++)
	{
		for (int r = 0;r < RI;r++)
		{
			for (int i = 0;i < RN;i++)
			{
				if (c == Class[i])
				{
					if (Coverage[(((c*RI) + r)*RN) + i] == Attr)
					{
						TP[(c*RI) + r]++;
					}
					else
					{
						FN[(c*RI) + r]++;
					}
				}
				else
				{
					if (Coverage[(((c*RI) + r)*RN) + i] == Attr)
					{
						FP[(c*RI) + r]++;
					}
					else
					{
						TN[(c*RI) + r]++;
					}
				}
			}
		}
	}
	for (int c = 0;c < CN;c++)
	{
		for (int r = 0;r < RI;r++)
		{
			if ((TP[(c*RI) + r] == 0) && (FP[(c*RI) + r] == 0))
			{
				Precision[(c*RI) + r] = 0;
			}
			else
			{
				Precision[(c*RI) + r] = TP[(c*RI) + r] / (TP[(c*RI) + r] + FP[(c*RI) + r]);
			}
			TruePositiveRate[(c*RI) + r] = TP[(c*RI) + r] / (TP[(c*RI) + r] + FN[(c*RI) + r]);
			if ((TN[(c*RI) + r] == 0) && (FP[(c*RI) + r] == 0))
			{
				TrueNegativeRate[(c*RI) + r] = 0;
			}
			else
			{
				TrueNegativeRate[(c*RI) + r] = TN[(c*RI) + r] / (TN[(c*RI) + r] + FP[(c*RI) + r]);
			}
			AccuracyRate[(c*RI) + r] = (TP[(c*RI) + r] + TN[(c*RI) + r]) / (TP[(c*RI) + r] + TN[(c*RI) + r] + FP[(c*RI) + r] + FN[(c*RI) + r]);
			Fitness_Value[(c*RI) + r] = Precision[(c*RI) + r] *TruePositiveRate[(c*RI) + r];
		}
	}
}