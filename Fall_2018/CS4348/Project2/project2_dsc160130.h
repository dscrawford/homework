#ifndef PROJECT2_H_
#define PROJECT2_H_

#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <semaphore.h>
#include <unistd.h>
#include <random>
#define NUM_CUSTOMERS 25
#define NUM_FRONTDESK 2
#define NUM_BELLHOP   2

extern sem_t checkIn, fdAvailable, bhAvailable, custExchanged, getBH,
  gotBags[NUM_CUSTOMERS], giveBags[NUM_CUSTOMERS], entersRoom[NUM_CUSTOMERS],
  giveTip[NUM_CUSTOMERS], giveRoom[NUM_CUSTOMERS];

struct customer {
  int room = -1; //room number
  int bags = -1; //number of bags
  int fd = -1; //front desk
  int bh = -1; //bellhop
};

extern int shareCust,  currRoom;

extern customer customers[NUM_CUSTOMERS];

//print functions
void CheckIn(int, int);
void GiveRoom(int, int, int);
void GetRoom(int, int, int);
void GetBellHop(int);
void EnterRoom(int, int);
void GetBags(int, int);
void GiveBags(int, int);
void GetBackBags(int, int);
void Retire(int);

//semaphore functions
void wait(sem_t&);
void send(sem_t&);
void init_semaphore(sem_t&, int);

//random Generators
int getNumber();

//thread functions
void *Customer(void*);
void *FrontDesk(void*);
void *Bellhop(void*);
void joinThreads(pthread_t[],int);

#endif //PROJECT2_H
