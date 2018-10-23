#include "project2_dsc160130.h"

int getNumber() {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0,5);
  return distribution(generator);
}

void *customer(void *arg) {
  int *pnum = (int *) arg;
  int cust = *pnum, bags = getNumber(), room, frontdesk, bellhop;
  free(arg);

  //checkin()
  CheckIn(cust, bags);
  //send checkin
  if ( sem_post(&checkIn) == -1) {
    printf("ERROR: sending semaphore");
    exit(1);
  }
  
  //wait employeeavailable
  wait(frontDeskAvailable);
  if ( sem_wait(&frontDeskAvailable) == -1) {
    printf("ERROR: waiting for semaphore");
    exit(1);
    }

  //wait exchangedone
  if ( sem_wait(&exchangeDone) == -1) {
    printf("ERROR: waiting for semaphore");
    exit(1);
  }
  //share guest
  tempcust = cust;
  //send guestshared
  if ( sem_post(&guestShared) == -1) {
    printf("ERROR: sending semaphore");
    exit(1);
  }
  //wait room&frontdesk
  if ( sem_wait(&valuesReady) == -1) {
    printf("ERROR: waiting for semaphore");
    exit(1);
  }
  //get  room&frontdesk
  room = temproom;
  frontdesk = tempfrontdesk;
  //send empExchange
  if ( sem_post(&empExchange) == -1) {
    printf("ERROR: sending semaphore");
    exit(1);
  }
  //send exchangedone
  if ( sem_post(&exchangeDone) == -1) {
    printf("ERROR: sending semaphore");
    exit(1);
  }
  //wait giveroom
  if ( sem_wait(&giveRoom) == -1) {
    printf("ERROR: waiting for semaphore");
    exit(1);
  }
  //getroom()
  GetRoom(frontdesk, cust, room);

  //if # bags > 2 get and singal for bellhop
  if (bags > 2) {
    GetBellHop(cust);
    if ( sem_post(&getBellhop) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
    }
  }
  //enterRoom()
  EnterRoom(cust, room);
  if (bags > 2) {
    //  signal EntersRoom
    if ( sem_post(&entersRoom) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
    }
    //  wait exchangedone
    if ( sem_wait(&exchangeDone) == -1) {
      printf("ERROR: waiting for semaphore");
      exit(1);
    }
    //  share guest
    tempcust = cust;
    //  send guestshared
    if ( sem_post(&guestShared) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
    }
    //  wait bellhopready
    if ( sem_wait(&bellhopReady) == -1) {
      printf("ERROR: waiting for semaphore");
      exit(1);
    }
    //  get bellhop
    bellhop = tempbellhop;
    //  send bellexchange
    if ( sem_post(&bellExchange) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
    }
    //  send exchangedone
    if ( sem_post(&exchangeDone) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
    }
    //  wait givebags
    if ( sem_wait(&giveBags) == -1) {
      printf("ERROR: waiting for semaphore");
      exit(1);
    }
    //  getBags()
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
    if ( sem_post(&frontDeskAvailable) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
      }
    //wait checkIn
    if ( sem_wait(&checkIn) == -1) {
      printf("ERROR: waiting for semaphore");
      exit(1);
    }
    room = ++currRoom;
    //wait empExchange
    if ( sem_wait(&empExchange) == -1) {
      printf("ERROR: waiting for semaphore");
      exit(1);
    }
    //get guest
    cust = tempcust;
    //share room, frontdesk
    temproom = room;
    tempfrontdesk = frontdesk;
    //send valuesReady
    if ( sem_post(&valuesReady) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
    }
    //giveRoom()
    GiveRoom(frontdesk, cust, room);
    //send giveRoom
    if ( sem_post(&giveRoom) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
    }
  }

  return NULL;
}

void *Bellhop(void *arg) {
  int *pnum = (int *) arg;
  int bellhop = *pnum, cust;
  while (true) {
    //send bellhopavailable
    if ( sem_post(&bellhopReady) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
    }
    //wait getBellhop
    if ( sem_wait(&getBellhop) == -1) {
      printf("ERROR: waiting for semaphore");
      exit(1);
    }
    //wait bellexchange
    if ( sem_wait(&bellExchange) == -1) {
      printf("ERROR: waiting for semaphore");
      exit(1);
    }
    //wait guestshared
    if ( sem_wait(&guestShared) == -1) {
      printf("ERROR: waiting for semaphore");
      exit(1);
    }
    //get guest
    cust = tempcust;
    //getBags()
    GetBags(bellhop, cust);
    //share bellhop
    tempbellhop = bellhop;
    //send bellhopready
    if ( sem_post(&bellhopReady) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
    }
    //wait enterRoom
    if ( sem_wait(&entersRoom) == -1) {
	  printf("ERROR: waiting for semaphore");
	  exit(1);
    }
    //giveBags()
    GiveBags(bellhop, cust);
    //send giveBags
    send(giveBags);
    if ( sem_post(&giveBags) == -1) {
      printf("ERROR: sending semaphore");
      exit(1);
    }
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

  printf("Simulation starts\n");
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
