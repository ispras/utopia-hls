#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "timer.h"



void timer_start(struct custom_timer *timer)
{
	clock_gettime(CLOCK_MONOTONIC_RAW , &(timer->start));
}

void timer_end(struct custom_timer *timer)
{
	clock_gettime(CLOCK_MONOTONIC_RAW , &(timer->end));
}

void print_timer(struct custom_timer *timer)
{
	printf("%-40s: %f sec\n", 	timer->name,
							timer->end.tv_sec -timer->start.tv_sec + 
					(float)(timer->end.tv_nsec-timer->start.tv_nsec)/1000000000);
}

float timer_get_value(struct custom_timer *timer)
{
	return timer->end.tv_sec -timer->start.tv_sec + 
		   (float)(timer->end.tv_nsec-timer->start.tv_nsec)/1000000000;
}
