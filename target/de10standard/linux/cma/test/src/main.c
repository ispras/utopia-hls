#include <stdio.h>
#include <stdlib.h>

#include "cma_api.h"

#include "timer/timer.h"

#define CMA_ALLOC_SIZE 		(2*1024*1024+1)


int main(void)
{
	int *mem_cached, *mem_noncached;
	struct custom_timer t_cached = {"Cached memory"};
	struct custom_timer t_noncached = {"Noncached memory"};

	printf("Initializing CMA API\n");
	if( cma_init() == -1){
		printf("FAILED!\n");
		return -1;		
	}

	printf("Allocating 0x%x bytes contigous memory, cached\n",CMA_ALLOC_SIZE);
	mem_cached = cma_alloc_cached(CMA_ALLOC_SIZE);
	if(mem_cached == NULL){
		printf("FAILED!\n");
		return -1;
	}
	
	printf("Allocating 0x%x bytes contigous memory, noncached\n",CMA_ALLOC_SIZE);
	mem_noncached = cma_alloc_noncached(CMA_ALLOC_SIZE);
	if(mem_noncached == NULL){
		printf("FAILED!\n");
		return -1;
	}

	printf("Initializing cached memory\n");
	timer_start(&t_cached);
	for(int i=0; i<CMA_ALLOC_SIZE/sizeof(int); i++)
		mem_cached[i] = i;
	timer_end(&t_cached);

	printf("Initializing noncached memory\n");
	timer_start(&t_noncached);
	for(int i=0; i<CMA_ALLOC_SIZE/sizeof(int); i++)
		mem_noncached[i] = i;
	timer_end(&t_noncached);

	printf("Initialization timings\n");
	print_timer(&t_cached);
	print_timer(&t_noncached);

	printf("Releasing contigous memory, cached\n");
	if(cma_free(mem_cached) == -1){
		printf("FAILED!\n");
		return -1;
	}

	printf("Releasing contigous memory, noncached\n");
	if(cma_free(mem_noncached) == -1){
		printf("FAILED!\n");
		return -1;
	}

	printf("Releasing CMA API\n");
	if( cma_release() == -1)
		printf("FAILED!\n");
		return -1;

	return 0;
}