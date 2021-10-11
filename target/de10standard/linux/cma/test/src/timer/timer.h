#ifndef TIMER_H_
#define TIMER_H_



struct custom_timer{
	char 			*name;
	struct timespec	start;
	struct timespec	end;
};



void timer_start(struct custom_timer *timer);
void timer_end(struct custom_timer *timer);
void print_timer(struct custom_timer *timer);
float timer_get_value(struct custom_timer *timer);


#endif