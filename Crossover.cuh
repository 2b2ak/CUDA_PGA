
// GPU

// CPU

void CPU_RuleSelection(float *FitnessValue, float *HV, int *SortedFitnessID, int CN, int RI)
{
	for (int c = 0;c < CN;c++)
	{
		for (int i = 0;i < RI;i++)
		{
			for (int m = (c*RI);m < ((c + 1)*RI);m++)
			{
				if (HV[(c*RI) + i] == FitnessValue[m])
				{
					FitnessValue[m] = -1;
					SortedFitnessID[(c*RI) + i] = m;
					break;
				}
			}
		}
	}
}

void CPU_AverageCoverage(int *Coverage, float *AvgCoverage, int *Class, int CN, int RI, int RN, int Attr)
{
	int ClassStride = 0;
	for (int c = 0;c < CN;c++)
	{
		for (int r = 0;r < RI;r++)
		{
			for (int a = 0;a < Attr;a++)
			{
				for (int i = 0;i < Class[c];i++)
				{
					AvgCoverage[(((c*RI) + r)*Attr) + a] += Coverage[(((c*RI) + r)*RN*Attr) + (ClassStride*Attr) + (i*Attr) + a];
					/*if (r == 1 && c == 1)
					{
						printf("AvgCoverage[%i] += Coverage[%i]\n", (((c*RI) + r)*Attr) + a, (((c*RI) + r)*RN*Attr) + (ClassStride*Attr) + (i*Attr) + a);
					}*/
				}
			}
		}
		ClassStride += Class[c];
	}

	for (int c = 0;c < CN;c++)
	{
		for (int r = 0;r < RI;r++)
		{
			for (int a = 0;a < Attr;a++)
			{
				AvgCoverage[(((c*RI) + r)*Attr) + a] /= Class[c];
			}
		}
	}
}

void CPU_Crossover(float *AvgCoverage, int *SortedFitnessID, bool *MetChromosome, int Iter, int CN, int RI, int Attr, int *RA, int *RC, float *RL, float *RU, int *newRA, int *newRC, float *newRL, float *newRU)
{
	int StackPtr;
	for (int c = 0;c < CN;c++)
	{
		StackPtr = 0;
		for (int r = RI - (RI / 2);r < RI;r++)
		{
			for (int a = 0;a < Attr;a++)
			{
				if (AvgCoverage[((SortedFitnessID[((c*RI) + r)])*Attr) + a] < AvgCoverage[((SortedFitnessID[((c*RI) + r - 1)])*Attr) + a])
				{

					if (RA[((SortedFitnessID[((c*RI) + r - 1)])*Attr) + a] != 0)
					{
						RA[((SortedFitnessID[((c*RI) + r)])*Attr) + a] = RA[((SortedFitnessID[((c*RI) + r - 1)])*Attr) + a];
						RC[((SortedFitnessID[((c*RI) + r)])*Attr) + a] = RC[((SortedFitnessID[((c*RI) + r - 1)])*Attr) + a];
						RL[((SortedFitnessID[((c*RI) + r)])*Attr) + a] = RL[((SortedFitnessID[((c*RI) + r - 1)])*Attr) + a];
						RU[((SortedFitnessID[((c*RI) + r)])*Attr) + a] = RU[((SortedFitnessID[((c*RI) + r - 1)])*Attr) + a];
					}
				}
			}
			MetChromosome[SortedFitnessID[((c*RI) + r)]] = true;
			//printf("MetChromosome[%i] = %i | SortedFitnessID[%i] = %i\n", (c*(RI / 2)) + StackPtr, MetChromosome[(c*(RI / 2)) + StackPtr], ((c*RI) + r), SortedFitnessID[((c*RI) + r)]);
			for (int a = 0;a < Attr;a++)
			{
				newRA[(Iter*CN*(RI / 2)*Attr) + (c*(RI / 2)*Attr) + (StackPtr*Attr) + a] = RA[((SortedFitnessID[((c*RI) + r)])*Attr) + a];
				newRC[(Iter*CN*(RI / 2)*Attr) + (c*(RI / 2)*Attr) + (StackPtr*Attr) + a] = RC[((SortedFitnessID[((c*RI) + r)])*Attr) + a];
				newRL[(Iter*CN*(RI / 2)*Attr) + (c*(RI / 2)*Attr) + (StackPtr*Attr) + a] = RL[((SortedFitnessID[((c*RI) + r)])*Attr) + a];
				newRU[(Iter*CN*(RI / 2)*Attr) + (c*(RI / 2)*Attr) + (StackPtr*Attr) + a] = RU[((SortedFitnessID[((c*RI) + r)])*Attr) + a];
			}
			StackPtr++;
		}
	}
	//srand(time(NULL));

	//bool *notSelected = new bool[RI - (RI / 2)];
	//for (int b = 0;b < (RI - (RI / 2));b++)
	//{
	//	notSelected[b] = true;
	//}

	//int r1, r2, StackPtr;
	//for (int c = 0;c < CN;c++)
	//{
	//	StackPtr = 0;
	//	while (StackPtr < (RI / 2))
	//	{
	//		r1 = (rand() % (RI - (RI / 2))) + (RI / 2);
	//		r2 = (rand() % (RI - (RI / 2))) + (RI / 2);
	//		if ((r1 != r2) && notSelected[r1 - (RI / 2)])
	//		{
	//			notSelected[r1 - (RI / 2)] = false;

	//			for (int a = 0;a < Attr;a++)
	//			{
	//				if (AvgCoverage[((SortedFitnessID[((c*RI) + r1)])*Attr) + a] < AvgCoverage[((SortedFitnessID[((c*RI) + r2)])*Attr) + a])
	//				{

	//					if (RA[((SortedFitnessID[((c*RI) + r2)])*Attr) + a] != 0)
	//					{
	//						RA[((SortedFitnessID[((c*RI) + r1)])*Attr) + a] = RA[((SortedFitnessID[((c*RI) + r2)])*Attr) + a];
	//						RC[((SortedFitnessID[((c*RI) + r1)])*Attr) + a] = RC[((SortedFitnessID[((c*RI) + r2)])*Attr) + a];
	//						RL[((SortedFitnessID[((c*RI) + r1)])*Attr) + a] = RL[((SortedFitnessID[((c*RI) + r2)])*Attr) + a];
	//						RU[((SortedFitnessID[((c*RI) + r1)])*Attr) + a] = RU[((SortedFitnessID[((c*RI) + r2)])*Attr) + a];
	//						//printf("newRA[%i] = %i | RA[%i] = %i\n", (Iter*CN*(RI / 2)*Attr) + (c*(RI / 2)*Attr) + (StackPtr*Attr) + a, newRA[(Iter*CN*(RI / 2)*Attr) + (c*(RI / 2)*Attr) + (StackPtr*Attr) + a], ((SortedFitnessID[((c*RI) + r2)])*Attr) + a, RA[((SortedFitnessID[((c*RI) + r2)])*Attr) + a]);
	//					}
	//				}
	//			}
	//			MetChromosome[(c*(RI / 2)) + StackPtr] = SortedFitnessID[((c*RI) + r1)];
	//			//printf("MetChromosome[%i] = %i | SortedFitnessID[%i] = %i\n", (c*(RI / 2)) + StackPtr, MetChromosome[(c*(RI / 2)) + StackPtr], ((c*RI) + r), SortedFitnessID[((c*RI) + r)]);
	//			for (int a = 0;a < Attr;a++)
	//			{
	//				newRA[(Iter*CN*(RI / 2)*Attr) + (c*(RI / 2)*Attr) + (StackPtr*Attr) + a] = RA[((SortedFitnessID[((c*RI) + r1)])*Attr) + a];
	//				newRC[(Iter*CN*(RI / 2)*Attr) + (c*(RI / 2)*Attr) + (StackPtr*Attr) + a] = RC[((SortedFitnessID[((c*RI) + r1)])*Attr) + a];
	//				newRL[(Iter*CN*(RI / 2)*Attr) + (c*(RI / 2)*Attr) + (StackPtr*Attr) + a] = RL[((SortedFitnessID[((c*RI) + r1)])*Attr) + a];
	//				newRU[(Iter*CN*(RI / 2)*Attr) + (c*(RI / 2)*Attr) + (StackPtr*Attr) + a] = RU[((SortedFitnessID[((c*RI) + r1)])*Attr) + a];
	//			}
	//			StackPtr++;
	//		}
	//	}
	//	for (int b = 0;b < (RI - (RI / 2));b++)
	//	{
	//		notSelected[b] = true;
	//	}
	//}
}