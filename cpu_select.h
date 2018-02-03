#ifndef CPU_SELECT_H
#define CPU_SELECT_H

#include <queue>
#include <algorithm>

template<class T>
T select_cpu(uint64_t n, T *cdata, uint64_t k){
	std::priority_queue<T> q;

	for(uint64_t i = 0;i<n;i++){
		if(q.size() < k){
			q.push(cdata[i]);
		}else if(q.top() > cdata[i]){
			q.pop();
			q.push(cdata[i]);
		}
	}
	return q.top();
}

template<class T>
T select_cpu_sort(uint64_t n, T *cdata, uint64_t k){
	std::sort(cdata, cdata + n);
	return cdata[k-1];
}

uint32_t radix_select_count(uint32_t *col_ui,uint64_t n,uint64_t &k,uint32_t prefix, uint32_t prefix_mask,uint32_t digit_mask, uint32_t digit_shf){
	uint32_t bins[16];
	for(int i = 0;i < 16;i++) bins[i]=0;

	for(uint64_t i = 0;i<n;i++){
		uint32_t vi = col_ui[i];
		uint8_t digit = (vi & digit_mask) >> digit_shf;
		bins[digit]+= ((vi & prefix_mask) == prefix);
	}

	if (bins[0] > k) return 0x0;
	for(int i = 1;i < 16;i++){
		bins[i]+=bins[i-1];
		if( bins[i] >= k ){
			k = k-bins[i-1];
			return i;
		}
	}
	return 0xF;
}

uint32_t select_cpu_radix(uint32_t *col_ui,uint64_t n, uint64_t k){
	uint32_t prefix=0x00000000;
	uint32_t prefix_mask=0x00000000;
	uint32_t digit_mask=0xF0000000;
	uint32_t digit_shf=28;

	uint64_t tmpK = k;
	for(int i = 0;i <8;i++){
		//printf("0x%08x,0x%08x,0x%08x,%02d, %"PRIu64"\n",prefix,prefix_mask,digit_mask,digit_shf,tmpK);

		uint32_t digit = radix_select_count(col_ui,n,tmpK,prefix,prefix_mask,digit_mask,digit_shf);

		prefix = prefix | (digit << digit_shf);
		prefix_mask=prefix_mask | digit_mask;
		digit_mask>>=4;
		digit_shf-=4;
	}
//	printf("k largest: 0x%08x\n",prefix);
	return prefix;
}

#endif
