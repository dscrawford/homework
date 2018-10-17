#ifndef PROJECT2_H
#define PROJECT2_H

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

sem_t frontDeskAvailable, checkIn, exchangeDone,guestShared,valuesReady,
  empExchange,giveRoom,getBellhop,entersRoom,bellhopReady,bellExchange,
  bellhopAvailable,giveBags,gotBags, frontDeskExchangeDone, custExchangeDone,
  bellhopExchangeDone, custBellExchangeDone;

int tempcust = -1, temproom = -1, tempbags = -1, tempfrontdesk = -1,
  tempbellhop = -1, currRoom = 0;

void CheckIn(int, int);
void GiveRoom(int, int, int);
void GetRoom(int, int, int);
void GetBellHop(int);
void EnterRoom(int, int);
void GetBags(int, int);
void GiveBags(int, int);
void GetBackBags(int, int);
void Retire(int);


#endif //PROJECT2_H
