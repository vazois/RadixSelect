#include "radix_select.h"
#include "cpu_select.h"
#include <cstdlib>
#include <ctime>

#include <cub/cub.cuh>
#include "Time.h"

#define N (32*1024*1024)
#define K (100)

template<class T>
void sample_print(uint64_t n, T *cdata, uint64_t k){
	uint64_t limit = k <= n ? k : n;
	for(uint64_t i = 0; i < limit; i++){
		std::cout << i << ": " <<cdata[i] << std::endl;
	}
}

template<class T>
void fill_vector(uint64_t n, T *&cdata){
	srand (time(NULL));
	for(uint64_t i = 0;i<n;i++){
		cdata[i] =  rand() % n + n/2;
		//cdata[i] =  i;
	}
}

template<class T>
T select_gpu_sort(uint64_t n, T *gdata, uint64_t k){
	T *cdata_out,*gdata_out;
	void *d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;

	cutil::safeMalloc<T,uint64_t>(&(gdata_out),sizeof(T)*n,"gdata_out alloc");
	cutil::safeMallocHost<T,uint64_t>(&(cdata_out),sizeof(T)*k,"cdata_out alloc");
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, gdata, gdata_out, n);

	cutil::safeMalloc<void,uint64_t>(&(d_temp_storage),temp_storage_bytes,"temp storage alloc");
	//cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, gdata, gdata_out, n);
	cutil::safeCopyToHost<T,uint64_t>(cdata_out,gdata_out,sizeof(T)*k, "copy from gdata_out to cdata_out");

	T t_gpu_s = cdata_out[k-1];
	cudaFree(gdata_out);
	cudaFreeHost(cdata_out);

	return t_gpu_s;
}

void test_uint32_t(int round){
	uint32_t *cdata,*gdata;
	cutil::safeMalloc<uint32_t,uint64_t>(&(gdata),sizeof(uint32_t)*N,"gdata alloc");
	cutil::safeMallocHost<uint32_t,uint64_t>(&(cdata),sizeof(uint32_t)*N,"cdata alloc");

	fill_vector<uint32_t>(N,cdata);
	//sample_print<uint32_t>(N,cdata,10);
	cutil::safeCopyToDevice<uint32_t,uint64_t>(gdata,cdata,sizeof(uint32_t)*N, " copy from cdata to gdata");

	uint32_t t_cpu = select_cpu(N,cdata,K);
	uint32_t t_cpu_s = select_cpu_sort(N,cdata,K);
	uint32_t t_cpu_r = select_cpu_radix(cdata,N,K);
	uint32_t t_gpu_s =select_gpu_sort(N,cdata,K);
	Time<msecs> t;
	t.start();
	uint32_t t_gpu_r = RadixSelect<uint32_t>::Select(N, gdata,K);
	if(round == 0) std::cout << "elapsed(ms) : " << t.lap() << std::endl;

	if(t_cpu!=t_cpu_s || t_cpu!=t_cpu_r || t_cpu!=t_gpu_s || t_cpu!=t_gpu_r){
		std::cout << "t_cpu: " << t_cpu << std::endl;
		std::cout << "t_cpu_s: " << t_cpu_s << std::endl;
		std::cout << "t_cpu_r: " << t_cpu_r << std::endl;
		std::cout << "t_gpu_s: " << t_gpu_s << std::endl;
		std::cout << "t_gpu_r: " << t_gpu_r << std::endl;
	}else{
		std::cout <<"("<< round<<")PASSED!!!" << std::endl;
	}

	cudaFree(gdata);
	cudaFreeHost(cdata);
}

int main(){

	for(int i = 0;i<1; i++) test_uint32_t(i);

	return 0;
}
