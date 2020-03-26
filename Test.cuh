
//GPU

//CPU

void TestLastGeneration(float *LastGenerationError, float * DS, int *TC, int *Class, int Iter, int CN, int RI, int DR, int Attr, int *RA, int *RC, float *RL, float *RU)
{
	int GeneID, ResID, DataID;
	int Stride = 0;
	int *Result = new int[CN*RI*DR];
	for (int i = 0;i < CN*RI*DR;i++)
	{
		Result[i] = 0;
	}

	int *Correct = new int[CN*RI];
	for (int i = 0;i < CN*RI;i++)
	{
		Correct[i] = 0;
	}

	for (int c = 0;c < CN;c++)
	{
		//printf("Class[%i] = %i\n", c, Class[c]);
		for (int r = 0;r < RI;r++)
		{
			for (int d = Stride;d < Stride + Class[c];d++)
			{
				for (int a = 0;a < Attr;a++)
				{
					GeneID = ((Iter - 1)*CN*RI*Attr) + (c*RI*Attr) + (r*Attr) + a;
					DataID = (d*Attr) + a;
					ResID = (c*RI*DR) + (r*DR) + d;
					if (RA[GeneID] == 0)
					{
						Result[ResID]++;
					}
					else
					{
						switch (RC[GeneID])
						{
						case 0:
							if (DS[DataID] <= RL[GeneID])
							{
								Result[ResID]++;
							}
							break;
						case 1:
							if (RU[GeneID] <= DS[DataID])
							{
								Result[ResID]++;
							}
							break;
						case 2:
							if ((DS[DataID] <= RL[GeneID]) && (RU[GeneID] <= DS[DataID]))
							{
								Result[ResID]++;
							}
							break;
						default:
							break;
						}
					}
				}
			}
		}
		Stride += Class[c];
	}

	Stride = 0;
	for (int c = 0;c < CN;c++)
	{
		for (int r = 0;r < RI;r++)
		{
			for (int d = Stride;d < Stride + Class[c];d++)
			{
				GeneID = (c*RI) + r;
				ResID = (c*RI*DR) + (r*DR) + d;
				//printf("GeneID = %i | Res ID = %i\n", GeneID, ResID);
				if (Result[ResID] == Attr)
				{
					Correct[GeneID]++;
					//printf("Correct[%i] = %i\n", GeneID, Correct[GeneID]);
				}
			}
		}
		Stride += Class[c];
	}

	::printf("Test Results for Generation #%i\n", Iter);

	for (int c = 0;c < CN;c++)
	{
		for (int r = 0;r < RI;r++)
		{
			GeneID = (c*RI) + r;
			LastGenerationError[(c*RI) + r] = (((float)Class[c] - (float)Correct[GeneID]) / (float)Class[c]) * 100;
			//::printf("Chromosome[%i][%i] error = %.2f\n", c, r, (((float)Class[c] - (float)Correct[GeneID]) / (float)Class[c]) * 100);
		}
	}
}