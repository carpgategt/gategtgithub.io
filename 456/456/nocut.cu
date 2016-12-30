#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include<iostream>
#include <windows.h>
#define GF 256
#define PP 301
using namespace std;
//以__global__開頭的函式，代表CUDA_square這個函式是由GPU執行
//----------------------------------------------
//加密
int *Log = new int[256];
int *ALog = new int[256];
void FillLogArrays(void)
{
	ALog[0] = 1;
	Log[0] = 1 - GF;
	for (int i = 1; i<GF; i++)
	{
		ALog[i] = ALog[i - 1] * 2;
		if (ALog[i] >= GF) ALog[i] ^= PP;
		Log[ALog[i]] = i;
	}
}
//CPU函數----------------------------------------------------------
int cpu_modProduct(int A, int B)
{
	if ((A == 0) || (B == 0)) return (0);
	else
	{
		return (ALog[(Log[A] + Log[B]) % (GF - 1)]);
	}
}
int cpu_modPower(int x, int a)
{
	if (a == 0)return 1;
	else return cpu_modProduct(x, cpu_modPower(x, a - 1));//x*power(x,a-1);
}
int cpu_modQuotient(int A, int B)// namely A divided by B
{
	if (B == 0) return (1 - GF);
	else if (A == 0) return (0);
	else
	{
		return (ALog[(Log[A] - Log[B] + (GF - 1)) % (GF - 1)]);
	}
}
int cpu_modSum(int A, int B) { return (A ^ B); }
int cpu_modDifference(int A, int B) { return (A ^ B); }
//----------------------------------------------------------------
__device__ int modProduct(int A, int B, int * gpu_Log, int * gpu_ALog)
{
	if ((A == 0) || (B == 0)) return (0);
	else
	{
		return (gpu_ALog[((gpu_Log[A] + gpu_Log[B]) % (GF - 1))]);
	}
}
__device__ int modPower(int x, int a, int * gpu_Log, int * gpu_ALog)
{
	int test = x;
	for (int i = 1; i <a; i++) {
		test = modProduct(x, test, gpu_Log, gpu_ALog);
	}
	return test;
}
__device__ int modQuotient(int A, int B, int * gpu_Log, int * gpu_ALog)// namely A divided by B
{
	if (B == 0) return (1 - GF);
	else if (A == 0) return (0);
	else
	{
		return (gpu_ALog[(gpu_Log[A] - gpu_Log[B] + (GF - 1)) % (GF - 1)]);
	}
}
__device__ int modSum(int A, int B) { return (A ^ B); }
__device__ int modDifference(int A, int B) { return (A ^ B); }
/*__device__ int datarandom(){

}*/
__global__ void gpu_sharing(int k, int n, int dataDim, int *gpu_dataA, int * gpu_share, int *gpu_Log, int *gpu_ALog, int *gpu_rand)
{
	int block_id = blockIdx.x;//讀取當前block的編號
	int thread_id = threadIdx.x; //讀取當前thread的編號
	int index = block_id *blockDim.x + thread_id;
	int *ans = new int[n];
	for (int i = 0; i < n; i++) {
		ans[i] = 0;
	}
	//亂數------------------------------------------------------------------------
	/*	curandState_t state;
	curand_init(0, index, 0, &state);

	for (int i = 0; i < k - 1; i++) {						//迴圈k-1圈，產生k-1個亂數且小於P之係數
	randCoefficient[i] = curand(&state) % 256;
	}*/
	//------------------------------------------------------------------------
	for (int h = 0; h < n; h++) {						//n份，每一份的常數
		for (int j = 0; j < k - 1; j++) {				//係數*X的次方數
			ans[h] = modSum(ans[h], modProduct(gpu_rand[j], modPower(h + 1, j + 1, gpu_Log, gpu_ALog), gpu_Log, gpu_ALog));//pow(h+1,j+1)--->h+1=基底 j+1=次方數
		}
		gpu_share[h*dataDim + index] = modSum(gpu_dataA[index], ans[h]); //gpu_dataA[index];																		 //= ans[h];
	}

	free(ans);
	//gpu_share[index] = gpu_dataB[index];
	//gpu_share[index + dataDim] = gpu_share[index];
	//gpu_share[index + 2*dataDim] = gpu_share[index];
	//gpu_share[index + 4* dataDim] = gpu_share[index];
}
__global__ void gpu_desharing(int k, int n, int dataDim, int *gpu_dataB, int * gpu_share, int *gpu_Log, int *gpu_ALog)
{
	int block_id = blockIdx.x;//讀取當前block的編號
	int thread_id = threadIdx.x; //讀取當前thread的編號
	int index = block_id *blockDim.x + thread_id;
	float L_end = 0;
	int L_Product = 1;


	for (int h = 0; h < k; h++) {							//K個還原
		L_Product = 1;
		for (int j = 0; j < k; j++) {
			if (h == j)continue;
			L_Product = modProduct(L_Product, modQuotient(j + 1, modDifference(j + 1, h + 1), gpu_Log, gpu_ALog), gpu_Log, gpu_ALog);
		}
		L_Product = modProduct(gpu_share[index + h*dataDim], L_Product, gpu_Log, gpu_ALog);
		L_end = modSum(L_end, L_Product);
	}
	gpu_dataB[index] = L_end;
}
int main()
{
	//輸入threadnum 數和blocknum 數----------------------------------------
	int  blocknum, threadnum;
	int  k, n;
	int i, j;
	cout << "threadnum 數:";
	cin >> threadnum;
	cout << endl;
	cout << "blocknum 數:";
	cin >> blocknum;
	cout << endl;
	cout << "k = ";
	cin >> k;
	cout << endl;
	cout << "n = ";
	cin >> n;
	cout << endl;
	int dataDim = blocknum * threadnum;
	cout << "資料數:" << dataDim << endl;
	//宣告-----------------------------------------------------------
	int *gpu_Log;
	int *gpu_ALog;
	// int * gpu_k;
	// int * gpu_n;
	int * gpu_dataA;
	//int * share;
	int * gpu_share;
	int *gpu_dataB;
	int *gpu_rand;
	//記憶體大小-----------------------------------------------------------------
	int *data = (int*)malloc(sizeof(int)*(dataDim));
	int *dataB = (int*)malloc(sizeof(int)*(dataDim));
	int *randCoefficient = (int*)malloc(sizeof(int)*k);
	// int *data2 = (int*) malloc (sizeof(int)*dataDim);
	int *share = (int*)malloc(sizeof(int)*n*dataDim);
	//share = new int[n*dataDim];
	// for (i = 0; i < n; i++)
	//	share[i] = new int [dataDim];
	//灑參數-----------------------------------------------------------------
	srand(time(NULL));
	FillLogArrays();
	for (i = 0; i < dataDim; i++)
	{
		data[i] = (rand() % 256) + 1;
	}
		for (i = 0; i < 2; i++)
	{
	cout<<"test data["<<i<<"]:"<<data[i]<<endl;
	}
	cout<<endl<< "後2筆:-------------------------"<<endl;
	for (i = dataDim - 1; i >dataDim - 3; i--)
	{
	cout << "test data[" << i << "]:" << data[i] << endl;
	}
	cout << endl;
	for (int x = 0; x < k - 1; x++) {
		randCoefficient[x] = (rand() % 256) + 1;
	}
	//GPU記憶體大小-----------------------------------------------------------------
	cudaMalloc((int**)&gpu_Log, sizeof(int)*GF);
	cudaMalloc((int**)&gpu_ALog, sizeof(int)*GF);
	cudaMalloc((int**)&gpu_dataA, sizeof(int)*dataDim);
	cudaMalloc((int**)&gpu_share, sizeof(int)*n*dataDim);
	cudaMalloc((int**)&gpu_rand, sizeof(int)*k);
	cout << "位元:" << sizeof(int)*k + sizeof(int)*n*dataDim + sizeof(int)*dataDim + sizeof(int)*GF + sizeof(int)*GF << endl;
	//-----------------------------------------------------------------------
	/*float time__tran, time__tran2,costime1,costime2;
	cudaEvent_t start_tran, stop_tran, start_tran2, stop_tran2,time_start,time_end,time_start2, time_end2;
	
	cudaEventCreate(&start_tran);
	cudaEventCreate(&stop_tran);
	cudaEventCreate(&start_tran2);
	cudaEventCreate(&stop_tran2);
	cudaEventCreate(&time_start);
	cudaEventCreate(&time_end);
	cudaEventCreate(&time_start2);
	cudaEventCreate(&time_end2);

	cudaEventRecord(start_tran, 0);*/
	/*	for (i = 0; i < n; i++) {
	for (int j = 0; j < 5; j++)
	{
	//printf("GPU_share[%2d]: %d \n ", j, share[i * dataDim + j]);//, Log[j], ALog[j]
	cout << "TEST GPU_share[" << j << "]" << share[i * dataDim + j] << endl;
	}
	printf("\n");
	}*/
	//GPU加密資料CPU傳到GPU------------------------------------------------------------------------------
	cudaMemcpy(gpu_Log, Log, sizeof(int)*GF, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_ALog, ALog, sizeof(int)*GF, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_dataA, data, sizeof(int)*dataDim, cudaMemcpyHostToDevice);
	//cudaMemcpy(gpu_dataB, dataB, sizeof(int)*dataDim, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_rand, randCoefficient, sizeof(int)*k, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_share, share, sizeof(int)*n*dataDim, cudaMemcpyHostToDevice);
	
	/*cudaEventRecord(stop_tran, 0);
	cudaEventSynchronize(stop_tran);
	cudaEventElapsedTime(&time__tran, start_tran, stop_tran);*/
	
	/*	free(data);
	free(randCoefficient);
	free(ALog);
	free(Log);*/
	//GPU加密計算----------------------------------------------------------------------------
	gpu_sharing << <blocknum, threadnum, 0 >> >(k, n, dataDim, gpu_dataA, gpu_share, gpu_Log, gpu_ALog, gpu_rand);
	
	cudaMemcpy(share, gpu_share, sizeof(int)*n*dataDim, cudaMemcpyDeviceToHost);
	cudaFree(gpu_dataA);
	cudaFree(gpu_rand);

	//GPU解密計算----------------------------------------------------------------------------
	cudaMalloc((int**)&gpu_dataB, sizeof(int)*dataDim);
	gpu_desharing << <blocknum, threadnum, 0 >> >(k, n, dataDim, gpu_dataB, gpu_share, gpu_Log, gpu_ALog);

	cudaThreadSynchronize();//同步
							//GPU傳到CPU--------------------------------------------------------------------------------------
							// for (i=0; i<n; i++)
							//GPU解密資料GPU傳到CPU------------------------------------------------------------------------------
	cudaMemcpy(dataB, gpu_dataB, sizeof(int)*dataDim, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(data, gpu_dataA, sizeof(int)*dataDim, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(ALog, gpu_ALog, sizeof(int)*256, cudaMemcpyDeviceToHost);

	//----------------------------------------------------------------------------------------------------------
	/*cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << "秒:" << elapsed_time_ms / 1000 << endl;*/
	
	//輸出---------------------------------------------------------------------
	//printf("CPU %f \n", L_end);//GPU
	cout << "-------------------------"<<endl;
	for (i = 0; i < n; i++) {
	for (int j = 0; j < 2; j++)
	{
	cout << "GPU_share[" << j << "]" << share[i * dataDim + j]<<endl;//printf("GPU_share[%2d]: %d \n ", j, share[i * dataDim + j]);//, Log[j], ALog[j]
	}
	printf("\n");
	}
	cout << endl << "後2筆:-------------------------" << endl;
	for (i = 0; i < n; i++) {
	for (int j = dataDim-1; j >dataDim-3; j--)
	{
	//printf("GPU_share[%2d]: %d \n ", j, share[i * dataDim + j]);//, Log[j], ALog[j]
	cout << "GPU_share[" << j << "]" << share[i * dataDim + j] << endl;

	}
	printf("\n");
	}
	cout << "----------------------" << endl;
	for (i = 0; i < 2; i++)
	{
	cout << "GPU_data[" << i << "]:" << dataB[i] << endl;

	}


	cout << endl << "後2筆:-------------------------" << endl;
	for (i = dataDim-1; i >dataDim-3; i--)
	{
	cout << "GPU_data[" << i << "]:" << dataB[i] << endl;

	}
	//cout << "秒:" << elapsed_time_ms/1000 << endl;
	cout << "----------------------" << endl;

	
	cudaFree(gpu_Log);
	cudaFree(gpu_ALog);
	cudaFree(gpu_dataB);
	cudaFree(gpu_share);
	//CPU加密解密運算----------------------------------------------------------------------
	//printf("CPU 加密與解密時間\n");

	clock_t end2;
	clock_t start2;
	double costTime_encode = 0;
	double costTime_decode = 0;

	int* cpu_ans = new int[n];
	int *L_end = new int[dataDim];
	for (int i = 0; i < dataDim; i++) {
		L_end[i] = 0;
	}
	int L_Product = 1;
	for (int h = 0; h < dataDim; h++) {
		start2 = clock();
		for (j = 0; j < n; j++) {
			cpu_ans[j] = 0;
		}
		for (i = 0; i < n; i++) {						//n份，每一份的常數
			for (j = 0; j < k - 1; j++) {				//係數*X的次方數
				cpu_ans[i] = cpu_modSum(cpu_ans[i], cpu_modProduct(randCoefficient[j], cpu_modPower(i + 1, j + 1)));//pow(i+1,j+1)--->i+1=基底 j+1=次方數
			}
			cpu_ans[i] = cpu_modSum(cpu_ans[i], data[h]);
		}
		end2 = clock();
		costTime_encode = costTime_encode + ((double)(end2 - start2) / CLK_TCK);
		//Decover----------------------------------------------------------------------
		start2 = clock();
		L_Product = 1;
		for (i = 0; i < k; i++) {							//K個還原
			L_Product = 1;										//累乘基底
			for (j = 0; j < k; j++) {
				if (i == j)continue;
				L_Product = cpu_modProduct(L_Product, cpu_modQuotient(j + 1, cpu_modDifference(j + 1, i + 1)));
			}
			L_Product = cpu_modProduct(cpu_ans[i + 1 - 1], L_Product);
			L_end[h] = cpu_modSum(L_end[h], L_Product);
		}
		end2 = clock();
		costTime_decode = costTime_decode + ((double)(end2 - start2) / CLK_TCK);
	}
	for (i = 0; i <2 ; i++)
	{
		cout <<"CPU_data[" << i << "]:" << L_end[i] << endl;
	}
	cout<<endl<<"後2筆:-------------------------" << endl;
	for (i = dataDim - 1; i >dataDim - 3; i--)
	{
		cout << "CPU_data[" << i << "]:" << L_end[i] << endl;
	}
	
	cout << "加密共 ： " << costTime_encode << " 秒\n";
	
	cout << "解密共 ： " << costTime_decode << " 秒\n";
	
	/*for (i = 0; i < 10; i++) {
	cout << "Data[" << i << "] ：" << cpu_data[i] << "\n";
	}
	for (i = 0; i < 10; i++) {
	cout << "Decode[" << i << "] ：" << L_end[i] << "\n";
	}*/
	system("pause");
}