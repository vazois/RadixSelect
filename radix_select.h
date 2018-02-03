#ifndef RADIX_SELECT_H
#define RADIX_SELECT_H

#include <inttypes.h>
#include "CudaHelper.h"

template<uint32_t block_size>
__global__ void radix_select_uint32_t_count(
		uint32_t *data,//unsorted data
		uint64_t n,//vector size
		uint64_t prefix,//discovered prefix for current iteration
		uint64_t prefix_mask,//matching digit position for discovered prefix
		uint64_t digit_mask,//position of digit being processed
		uint32_t digit_shf,//shift places of digit being processed
		uint32_t *bins//bins to aggregate digit occurences
);

__global__ void clear(uint32_t *bins);

//__device__ uint32_t GPU_BINS[16];
template<class T>
class RadixSelect{
	const static uint32_t VALUE_PER_THREAD = 224;
	const static uint32_t BLOCK_SIZE = 128;

public:
	static T Select(uint64_t num_items, T *d_in, uint64_t k){
		uint32_t *cpu_bins;
		cudaMallocHost(&cpu_bins, sizeof(uint32_t)*16);
		//cutil::safeMallocHost<uint32_t,uint64_t>(&(cpu_bins),sizeof(uint32_t)*16,"cpu_bins alloc");
		//for(int i = 0;i<16;i++) cpu_bins[i]=0;

		uint32_t *gpu_bins;
		cudaMalloc(&gpu_bins, sizeof(uint32_t)*16);
		//cutil::safeMalloc<uint32_t,uint64_t>(&(gpu_bins),sizeof(uint32_t)*16,"gpu_bins alloc");

		//cutil::safeCopyToDevice<uint32_t,uint64_t>(gpu_bins,cpu_bins,sizeof(uint32_t)*16, " copy from cpu_bins to gpu_bins");
		//cudaMemcpy(gpu_bins,cpu_bins,sizeof(uint32_t)*16,cudaMemcpyHostToDevice);
		clear<<<1,1>>>(gpu_bins);

		uint32_t GRID_SIZE = (num_items-1)/(VALUE_PER_THREAD * BLOCK_SIZE) + 1;
		dim3 grid(GRID_SIZE,1,1);
		dim3 block(BLOCK_SIZE,1,1);

		//Default for 32-bit values;
		uint64_t prefix=0x00000000;
		uint64_t prefix_mask=0x00000000;
		uint64_t digit_mask=0xF0000000;
		uint32_t digit_shf=28;
		uint8_t digit =0x0;
		//

		uint64_t tmpK = k;
		for(uint32_t i = 0;i <8;i++){
			radix_select_uint32_t_count<BLOCK_SIZE><<<grid,block>>>(
					d_in,
					num_items,
					prefix,
					prefix_mask,
					digit_mask,
					digit_shf,
					gpu_bins
			);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing radix_select_count");
			cutil::safeCopyToHost<uint32_t,uint64_t>(cpu_bins,gpu_bins,sizeof(uint32_t)*16, " copy from gpu_bins to cpu_bins");

			if (cpu_bins[0] > tmpK){
				digit = 0x0;
			}else{
				for(uint32_t i = 1;i < 16;i++){
					cpu_bins[i]+=cpu_bins[i-1];
					if( cpu_bins[i] >= tmpK ){
						tmpK = tmpK-cpu_bins[i-1];
						digit=i;
						break;
					}
				}
			}

			//for(int i = 0;i<16;i++) cpu_bins[i]=0;
			//cutil::safeCopyToDevice<uint32_t,uint64_t>(gpu_bins,cpu_bins,sizeof(uint32_t)*16, " copy from cpu_bins to gpu_bins");
			clear<<<1,1>>>(gpu_bins);

			prefix = prefix | (digit << digit_shf);
			prefix_mask=prefix_mask | digit_mask;
			digit_mask>>=4;
			digit_shf-=4;
		}

		cudaFree(gpu_bins);
		cudaFreeHost(cpu_bins);
		return prefix;
	}

	static T All(uint64_t num_items, T *d_in, uint64_t k){
		uint32_t *cpu_bins;
		cudaMallocHost(&cpu_bins, sizeof(uint32_t)*16);
		//cutil::safeMallocHost<uint32_t,uint64_t>(&(cpu_bins),sizeof(uint32_t)*16,"cpu_bins alloc");
		//for(int i = 0;i<16;i++) cpu_bins[i]=0;

		uint32_t *gpu_bins;
		cudaMalloc(&gpu_bins, sizeof(uint32_t)*16);
		//cutil::safeMalloc<uint32_t,uint64_t>(&(gpu_bins),sizeof(uint32_t)*16,"gpu_bins alloc");

		//cutil::safeCopyToDevice<uint32_t,uint64_t>(gpu_bins,cpu_bins,sizeof(uint32_t)*16, " copy from cpu_bins to gpu_bins");
		//cudaMemcpy(gpu_bins,cpu_bins,sizeof(uint32_t)*16,cudaMemcpyHostToDevice);
		clear<<<1,1>>>(gpu_bins);

		//Default for 32-bit values;///////
		uint64_t prefix=0x00000000;
		uint64_t prefix_mask=0x00000000;
		uint64_t digit_mask=0xF0000000;
		uint32_t digit_shf=28;
		uint8_t digit =0x0;
		uint64_t tmpK = k;
		if( std::is_same<T,float>::value || std::is_same<T,uint32_t>::value || std::is_same<T,int32_t>::value ){
			prefix=0x0000000000000000;
			prefix_mask=0x0000000000000000;
			digit_mask=0xF000000000000000;
			digit_shf=28;
			digit =0x0;
		}else if( std::is_same<T,double>::value || std::is_same<T,uint64_t>::value || std::is_same<T,int64_t>::value ){
			prefix=0x0000000000000000;
			prefix_mask=0x0000000000000000;
			digit_mask=0xF000000000000000;
			digit_shf=60;
			digit =0x0;
		}
		////////////////////////////////////////////

		uint32_t GRID_SIZE = (num_items-1)/(VALUE_PER_THREAD * BLOCK_SIZE) + 1;
		dim3 grid(GRID_SIZE,1,1);
		dim3 block(BLOCK_SIZE,1,1);

		for(uint32_t i = 0;i <8;i++){
			//Count Occurrences//
			if (std::is_same<T,uint32_t>::value){
				radix_select_uint32_t_count<BLOCK_SIZE><<<grid,block>>>(
						d_in,
						num_items,
						prefix,
						prefix_mask,
						digit_mask,
						digit_shf,
						gpu_bins
				);
			}else if (std::is_same<T,float>::value){

			}
			//

			//Gather results
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing radix_select_count");
			cutil::safeCopyToHost<uint32_t,uint64_t>(cpu_bins,gpu_bins,sizeof(uint32_t)*16, " copy from gpu_bins to cpu_bins");
			if (cpu_bins[0] > tmpK){
				digit = 0x0;
			}else{
				for(uint32_t i = 1;i < 16;i++){
					cpu_bins[i]+=cpu_bins[i-1];
					if( cpu_bins[i] >= tmpK ){
						tmpK = tmpK-cpu_bins[i-1];
						digit=i;
						break;
					}
				}
			}
			//

			//Update parameters for next iteration//
			for(int i = 0;i<16;i++) cpu_bins[i]=0;
			cutil::safeCopyToDevice<uint32_t,uint64_t>(gpu_bins,cpu_bins,sizeof(uint32_t)*16, " copy from cpu_bins to gpu_bins");
			//clear<<<1,1>>>(gpu_bins);
			prefix = prefix | (digit << digit_shf);
			prefix_mask=prefix_mask | digit_mask;
			digit_mask>>=4;
			digit_shf-=4;
		}

		cudaFree(gpu_bins);
		cudaFreeHost(cpu_bins);
		return prefix;
	}

	static T Flagged(uint64_t num_items, T *d_in, uint64_t k){

	}
};

__global__ void clear(uint32_t *bins){
	bins[0]=0; bins[1]=0; bins[2]=0; bins[3]=0;
	bins[4]=0; bins[5]=0; bins[6]=0; bins[7]=0;
	bins[8]=0; bins[9]=0; bins[10]=0; bins[11]=0;
	bins[12]=0; bins[13]=0; bins[14]=0; bins[15]=0;
}

template<uint32_t block_size>
__global__ void radix_select_uint32_t_count(
		uint32_t *data,//unsorted data
		uint64_t n,//vector size
		uint64_t prefix,//discovered prefix for current iteration
		uint64_t prefix_mask,//matching digit position for discovered prefix
		uint64_t digit_mask,//position of digit being processed
		uint32_t digit_shf,//shift places of digit being processed
		uint32_t *bins//bins to aggregate digit occurences
		){
	__shared__ uint32_t sbins[block_size][16];//Counting bins
	uint64_t offset = block_size * blockIdx.x + threadIdx.x;//Vector offset

	for(int i = 0;i<16;i++) sbins[threadIdx.x][i] = 0;
	while (offset < n){
		uint32_t v = data[offset];
		uint8_t digit = (v & digit_mask) >> digit_shf;
		sbins[threadIdx.x][digit]+= ((v & prefix_mask) == prefix);
		offset+=gridDim.x*block_size;
	}
	__syncthreads();

	if(threadIdx.x < 16){
		for(int i = 1; i < block_size;i++){
			sbins[0][threadIdx.x]+= sbins[i][threadIdx.x];
		}
		//gpu_bins[threadIdx.x] = tbins[threadIdx.x];
		atomicAdd(&bins[threadIdx.x],sbins[0][threadIdx.x]);
	}
}

#endif
