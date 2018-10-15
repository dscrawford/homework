#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <semaphore.h>
#include <unistd.h>
#include <random>
#define NUM_CUSTOMERS 3
#define NUM_FRONTDESK 2
#define NUM_BELLHOP   2

sem_t checkIn, gaveRoom, getBellhop, enterRoom, giveBags;

int customerNo = -1, roomNo[NUM_CUSTOMERS], bags[NUM_CUSTOMERS],
  frontdeskNo = -1, currRoom = 0;

int getNumber() {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0,5);
  return distribution(generator);
}

void CheckIn() {
  printf("Guest %d enters hotel with %d bags\n", customerNo, bags);
}

void giveRoom() {
  printf("Front desk employee %d registers guest %d and assigns room %d\n",
	 frontdeskNo, customerNo, roomNo);
}
void getRoom() {
  printf("Guest %d receives key for room %d from front desk employee %d\n",
	 customerNo, roomNo, frontdeskNo);
}

void *customer(void *arg) {
  int lcustNo = 0;
  int *pnum = (int *) arg;
  int num = *pnum;
  free(arg);


  pthread_mutex_lock(&mtx);
  while (customerNo != -1) {
    //wait
    }
  customerNo = lcustNo = num;
  bags[customerNo] = getNumber();
  CheckIn();
  pthread_mutex_unlock(&mtx);
  
  if(sem_post (&checkIn) == -1) {
    printf("Post checkIn\n");
    exit(1);
  }
  if(sem_wait(&gaveRoom) == -1) {
    printf("Wait on gaveRoom");
    exit(1);
  }

  getRoom();

  return NULL;
}

void *FrontDesk(void *arg) {
  int lfrontdeskNo;
  int *pnum = (int *) arg;
  int num = *pnum;
  free(arg);
  frontdeskNo = lfrontdeskNo = num;

  while (true) {
    if (sem_wait (&checkIn) == -1) {
      printf("Post semaphore");
      exit(1);
    }

    roomNo[customerNo] = ++currRoom;
    giveRoom();

    if (sem_post (&gaveRoom) == -1) {
      printf("post on gaveRoom");
    }
  }

  return NULL;
}

void joinThreads(pthread_t thread[], int numberOfThreads) {
  int status;
  for (int thread_count = 0; thread_count < numberOfThreads; ++thread_count) {
    status = pthread_join(thread[thread_count],NULL);
    if (status != 0) {
      printf("ERROR: Could not join threads");
      exit(1);
    }
  }
}

int main() {
  int status;
  pthread_t customers[NUM_CUSTOMERS];
  pthread_t frontdesk[NUM_FRONTDESK];
  pthread_t bellhops   [NUM_BELLHOP];
  
  if (sem_init (&checkIn,0,0) == -1)
    exit(1);
  if (sem_init (&gaveRoom,0,0) == -1)
    exit(1);
  if (sem_init (&getBellhop,0,0) == -1)
    exit(1);
  if (sem_init (&enterRoom,0,0) == -1)
    exit(1);
  if (sem_init (&giveBags,0,0) == -1)
    exit(1);

  for (int frontdesk_count = 0; frontdesk_count < NUM_FRONTDESK; ++frontdesk_count) {
    int *pnum = (int*)malloc(sizeof(int));
    *pnum = frontdesk_count;
    status = pthread_create(&frontdesk[frontdesk_count], NULL, FrontDesk,
			    (void*)pnum);
    printf("Front desk employee %d created\n", *pnum);
    if (status != 0) {
      printf("Create thread\n");
      exit(1);
    }
  }
  
  for (int customer_count = 0; customer_count < NUM_CUSTOMERS; ++customer_count) {
    int *pnum = (int*)malloc(sizeof(int));
    *pnum = customer_count;
    status = pthread_create(&customers[customer_count], NULL, customer,
			    (void*)pnum);
    printf("Guest %d created\n", *pnum);
    if (status != 0) {
      printf("Create thread\n");
      exit(1);
    }
  }

  int thread_count = NUM_FRONTDESK + NUM_CUSTOMERS + NUM_BELLHOP;

  joinThreads(customers, NUM_CUSTOMERS);
  joinThreads(frontdesk, NUM_FRONTDESK);

  return 0;
}
