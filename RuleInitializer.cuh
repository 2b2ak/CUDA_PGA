
void NumofEachClass(int *NumClass, int NumofRows, int *Class)
{
	for (int i = 0;i < NumofRows;i++)
	{
		NumClass[(int)Class[i]]++;
	}
}

void SetMinMax(float *DataSet, float *MinValue, float *MaxValue, int NumofAttr, int NumofRows)
{
	for (int i = 0;i < NumofAttr;i++)
	{
		//::printf("Min[%i] = %.2f | Max[%i] = %.2f\n", i, DataSet[i], i, DataSet[i]);
		MinValue[i] = DataSet[i];
		MaxValue[i] = DataSet[i];
	}

	for (int i = 1;i < NumofRows;i++)
	{
		for (int j = 0;j < NumofAttr;j++)
		{
			if (DataSet[(i*NumofAttr) + j] < MinValue[j])
			{
				MinValue[j] = DataSet[(i*NumofAttr) + j];
			}
			if (DataSet[(i*NumofAttr) + j] > MaxValue[j])
			{
				MaxValue[j] = DataSet[(i*NumofAttr) + j];
			}
		}
	}
}

void CPU_InitPopulation_FirstGeneration(int *RA, int *RC, float *RL, float *RU, float *MinValue, float *MaxValue, int Attr, int RI, int CN)
{
	srand(time(NULL));

	for (int c = 0;c < CN;c++)
	{
		for (int r = 0;r < RI;r++)
		{
			for (int a = 0;a < Attr;a++)
			{
				RA[(((c*RI) + r)*Attr) + a] = (rand() % 2);
				if (RA[(((c*RI) + r)*Attr) + a] == 0)
				{
					RC[(((c*RI) + r)*Attr) + a] = 0;
					RL[(((c*RI) + r)*Attr) + a] = 0;
					RU[(((c*RI) + r)*Attr) + a] = 0;
				}
				else
				{
					RC[(((c*RI) + r)*Attr) + a] = (rand() % 3);
					switch (RC[(((c*RI) + r)*Attr) + a])
					{
					case 0:
						if (((MaxValue[a] - MinValue[a]) + MinValue[a]) == 0)
						{
							RL[(((c*RI) + r)*Attr) + a] = 0;
						}
						else
						{
							RL[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
						}
						RU[(((c*RI) + r)*Attr) + a] = 0;
						break;
					case 1:
						if (((MaxValue[a] - MinValue[a]) + MinValue[a]) == 0)
						{
							RU[(((c*RI) + r)*Attr) + a] = 0;
						}
						else
						{
							RU[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
						}
						RL[(((c*RI) + r)*Attr) + a] = 0;
						break;
					case 2:
						if (((MaxValue[a] - MinValue[a]) + MinValue[a]) == 0)
						{
							RL[(((c*RI) + r)*Attr) + a] = 0;
							RU[(((c*RI) + r)*Attr) + a] = 0;
						}
						else
						{
							RL[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
							RU[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
							while (RU[(((c*RI) + r)*Attr) + a] < RL[(((c*RI) + r)*Attr) + a])
							{
								RL[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
								RU[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
							}
						}
						break;
					default:
						break;
					}
				}
			}
		}
	}
}

//bool CrossedChromosomes(int *MetChromosomes, int ch, int CN, int RI)
//{
//	for (int c = 0;c < CN;c++)
//	{
//		for (int r = 0;r < (RI / 2);r++)
//		{
//			if (MetChromosomes[(c*(RI / 2)) + r] == ch)
//			{
//				return false;
//			}
//		}
//	}
//	return true;
//}
//
//void CPU_InitPopulation(int *RA, int *RC, float *RL, float *RU, float *MinValue, float *MaxValue, int Attr, int RI, int CN, int *MetChromosomes)
//{
//	srand(time(NULL));
//
//	for (int c = 0;c < CN;c++)
//	{
//		for (int r = 0;r < RI;r++)
//		{
//			if (CrossedChromosomes(MetChromosomes, r, CN, RI))
//			{
//				for (int a = 0;a < Attr;a++)
//				{
//					RA[(((c*RI) + r)*Attr) + a] = (rand() % 2);
//					if (RA[(((c*RI) + r)*Attr) + a] == 0)
//					{
//						RC[(((c*RI) + r)*Attr) + a] = 0;
//						RL[(((c*RI) + r)*Attr) + a] = 0;
//						RU[(((c*RI) + r)*Attr) + a] = 0;
//					}
//					else
//					{
//						RC[(((c*RI) + r)*Attr) + a] = (rand() % 3);
//						switch (RC[(((c*RI) + r)*Attr) + a])
//						{
//						case 0:
//							RL[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
//							RU[(((c*RI) + r)*Attr) + a] = 0;
//							break;
//						case 1:
//							RL[(((c*RI) + r)*Attr) + a] = 0;
//							RU[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
//							break;
//						case 2:
//							RL[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
//							RU[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
//							while (RU[(((c*RI) + r)*Attr) + a] < RL[(((c*RI) + r)*Attr) + a])
//							{
//								RL[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
//								RU[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
//							}
//							break;
//						default:
//							break;
//						}
//					}
//				}
//			}
//		}
//	}
//}

//__global__ void GPU_InitPopulation(int *RA, int *RC, float *RL, float *RU, float *MinValue, float *MaxValue)
//{
//	curandState *state;
//	__device__ unsigned int curand(curandState_t *state);
//	int Attr = (blockDim.x*blockIdx.x) + threadIdx.x;
//	curand_init(0, Attr, 0, &state[Attr]);
//	int j = threadIdx.y;
//	if (j == 0)
//	{
//		RC[Attr] = (curand(&state) % 2);
//	}
//	if (j == 1)
//	{
//		RL[Attr] = (curand_uniform_double(&state);(&ThreadState) % (int)(MaxValue[Attr] - MinValue[Attr])) + MinValue[Attr];
//		RU[Attr] = (curand(&ThreadState) % (int)(MaxValue[Attr] - MinValue[Attr])) + MinValue[Attr];
//		while (RU[Attr] < RL[Attr])
//		{
//			RL[Attr] = (curand(&ThreadState) % (int)(MaxValue[Attr] - MinValue[Attr])) + MinValue[Attr];
//			RU[Attr] = (curand(&ThreadState) % (int)(MaxValue[Attr] - MinValue[Attr])) + MinValue[Attr];
//		}
//	}
//	if (j == 2)
//	{
//		RA[Attr] = (curand(&ThreadState) % 2);
//	}
//}
