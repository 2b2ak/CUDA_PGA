
//GPU

//CPU

void CPU_Mutation(float *AvgCoverage, bool *MetChromosome, float *MinValue, float *MaxValue, int CN, int RI, int Attr, int *RA, int *RC, float *RL, float *RU)
{
	srand(time(NULL));

	for (int c = 0;c < CN;c++)
	{
		for (int r = 0;r < RI;r++)
		{
			if (!(MetChromosome[(c*RI) + r]))
			{
				for (int a = 0;a < Attr;a++)
				{
					if (AvgCoverage[(((c*RI) + r)*Attr) + a] < 1)
					{
						RC[(((c*RI) + r)*Attr) + a] = (rand() % 3);
						switch (RC[(((c*RI) + r)*Attr) + a])
						{
						case 0:
							RL[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
							RU[(((c*RI) + r)*Attr) + a] = 0;
							break;
						case 1:
							RL[(((c*RI) + r)*Attr) + a] = 0;
							RU[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
							break;
						case 2:
							RL[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
							RU[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
							while (RU[(((c*RI) + r)*Attr) + a] < RL[(((c*RI) + r)*Attr) + a])
							{
								RL[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
								RU[(((c*RI) + r)*Attr) + a] = (rand() % (int)(MaxValue[a] - MinValue[a])) + MinValue[a];
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
}