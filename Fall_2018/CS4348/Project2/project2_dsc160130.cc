#include "project2_dsc160130.h"

int getNumber() {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0,5);
  return distribution(generator);
}

void wait(sem_t &semaphore) {
  if (sem_wait(&semaphore) == -1) {
    printf("ERROR: waiting for semaphore");
    exit(1);
  }
}

void send(sem_t &semaphore) {
  if (sem_post(&semaphore) == -1) {
    printf("ERROR: sending semaphore");
    exit(1);
  }
}
void *customer(void *arg) {
  int *pnum = (int *) arg;
  int cust = *pnum, bags = getNumber(), room, frontdesk, bellhop;
  free(arg);

  //checkin()
  CheckIn(cust, bags);
  //wait employeeavailable
  wait(frontDeskAvailable);
  //send checkin
  send(checkIn);
  //wait exchangedone
  wait(exchangeDone);//1->0
  //share guest
  tempcust = cust;
  //send guestshared
  send(guestShared);
  //wait room&frontdesk
  wait(valuesReady);
  //get  room&frontdesk
  room = temproom;
  frontdesk = tempfrontdesk;
  //wait frontDeskExchangeDone
  wait(frontDeskExchangeDone);
  //wait custExchangeDone
  send(custExchangeDone);
  //send empExchange
  send(empExchange);

  //wait giveroom
  wait(giveRoom);
  //getroom()
  GetRoom(frontdesk, cust, room);

  //if # bags > 2 get and singal for bellhop
  if (bags > 2) {
    //wait bellhopAvailable
    wait(bellhopAvailable);//1
    printf("1\n");
    //GetBellHop()
    GetBellHop(cust);
    //send getBellhop
    send(getBellhop);//2
    printf("2\n");
    //wait exchangeDone
    wait(exchangeDone);//getting stuck
    printf("3\n");
    //share guest
    tempcust = cust;
    //send guestShared
    send(guestShared);//4
    printf("4\n");
    //wait bellhopReady
    wait(bellhopReady);//5
    printf("5\n");
    //get bellhop
    bellhop = tempbellhop;
    //send custBellExchangeDone
    send(custBellExchangeDone);
    //waitbellhopExchangeDone
    wait(bellhopExchangeDone);
    //send bellExchange
    send(bellExchange);//6
    printf("6\n");
    //wait gotBags
    wait(gotBags);//7
    printf("7\n");
  }
  //enterRoom()
  EnterRoom(cust, room);
  if (bags > 2) {
    //send entersRoom
    send(entersRoom);
    //wait giveBags
    wait(giveBags);
    //getBackBags
    GetBackBags(bellhop, cust);
  }
  //then Retire()
  Retire(cust);  
  
  return NULL;
}

void *FrontDesk(void *arg) {
  int *pnum = (int *) arg;
  int frontdesk = *pnum, cust, room;
  free(arg);

  while (true) {
    //send frontAvailable
    send(frontDeskAvailable);
    //wait checkIn
    wait(checkIn);
    //wait empExchange
    wait(empExchange);//1
    //wait guestShared
    wait(guestShared);//1
    //get guest
    cust = tempcust;
    //share room, frontdesk
    room = ++currRoom;
    temproom = room;
    tempfrontdesk = frontdesk;
    //send valuesReady
    send(valuesReady);
    //send frontDeskExchangeDone
    send(frontDeskExchangeDone);
    wait(custExchangeDone);
    //send exchangeDone
    send(exchangeDone);

    //giveRoom()
    GiveRoom(frontdesk, cust, room);
    //send giveRoom
    send(giveRoom);
  }

  return NULL;
}

void *Bellhop(void *arg) {
  int *pnum = (int *) arg;
  int bellhop = *pnum, cust;
  while (true) {
    //send bellhopAvailable
    send(bellhopAvailable);//1
    //wait getBellhop
    wait(getBellhop);//2
    //wait bellExchange
    wait(bellExchange);//6
    //wait guestReady
    wait(guestShared);//4
    //get guest
    cust = tempcust;
    //share bellhop
    tempbellhop = bellhop;
    //send bellhopReady
    send(bellhopReady);//5
    //send bellHopExchangeDone
    send(bellhopExchangeDone);
    //wait custBellExchangeDone
    wait(custBellExchangeDone);
    //send exchangeDone
    send(exchangeDone);//3
    //getBags()
    GetBags(bellhop, cust);
    //send gotBags
    send(gotBags);//7
    //wait entersRoom
    wait(entersRoom);
    //giveBags()
    GiveBags(bellhop,cust);
    //send givebags
    send(giveBags);
  }
}

void joinThreads(pthread_t thread[], int numberOfThreads) {
  int status;
  for (int thread_count = 0; thread_count < numberOfThreads; ++thread_count) {
    status = pthread_join(thread[thread_count],NULL);
    if (status != 0) {
      printf("ERROR: Could not join threads\n");
      exit(1);
    }
    printf("Guest %d joined\n", thread_count);
  }
}

int main() {
  int status;
  pthread_t customers[NUM_CUSTOMERS];
  pthread_t frontdesk[NUM_FRONTDESK];
  pthread_t bellhops   [NUM_BELLHOP];

  if (sem_init (&frontDeskAvailable,0,0) == -1)
    exit(1);
  if (sem_init (&checkIn,0,0) == -1)
    exit(1);
  if (sem_init (&exchangeDone,0,1) == -1)
    exit(1);
  if (sem_init (&guestShared,0,0) == -1)
    exit(1);
  if (sem_init (&valuesReady,0,0) == -1)
    exit(1);
  if (sem_init (&empExchange,0,1) == -1)
    exit(1);
  if (sem_init (&giveRoom,0,0) == -1)
    exit(1);
  if (sem_init (&getBellhop,0,0) == -1)
    exit(1);
  if (sem_init (&entersRoom,0,0) == -1)
    exit(1);
  if (sem_init (&bellhopReady,0,0) == -1)
    exit(1);
  if (sem_init (&bellExchange,0,1) == -1)
    exit(1);
  if (sem_init (&giveBags,0,0) == -1)
    exit(1);
  if (sem_init (&gotBags,0,0) == -1)
    exit(1);
  if (sem_init (&frontDeskExchangeDone,0,0) == -1)
    exit(1);
  if (sem_init (&custExchangeDone,0,0) == -1)
    exit(1);
  if (sem_init (&bellhopExchangeDone,0,0) == -1)
    exit(1);
  if (sem_init (&custBellExchangeDone,0,0) == -1)
    exit(1);

  printf("Simulation starts\n");
  for (int frontdesk_count = 0; frontdesk_count < NUM_FRONTDESK;
       ++frontdesk_count) {
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

  for (int bellhop_count = 0; bellhop_count < NUM_FRONTDESK; ++bellhop_count) {
    int *pnum = (int*)malloc(sizeof(int));
    *pnum = bellhop_count;
    status = pthread_create(&bellhops[bellhop_count], NULL, Bellhop,
			    (void*)pnum);
    printf("Bellhop %d created\n", *pnum);
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

  joinThreads(customers, NUM_CUSTOMERS);

  printf("Simulation ends\n");
  return 0;
}
