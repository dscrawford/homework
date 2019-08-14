#include "project2_dsc160130.h"
void *Customer(void *arg) {
  //Get the value of the ith customer.
  int *pnum = (int *) arg;
  int cust = *pnum;
  free(arg);

  customers[cust].bags = getNumber();

  CheckIn(cust, customers[cust].bags);

  wait(fdAvailable);

  wait(custExchanged);

  shareCust = cust;

  send(checkIn);

  wait(giveRoom[cust]);

  GetRoom(customers[cust].fd,cust,customers[cust].room);

  if (customers[cust].bags > 2) {
    GetBellHop(cust);

    wait(bhAvailable);

    wait(custExchanged);

    shareCust = cust;

    send(getBH);

    wait(gotBags[cust]);
  }

  EnterRoom(cust, customers[cust].room);

  if(customers[cust].bags > 2) {
    send(entersRoom[cust]);

    wait(giveBags[cust]);

    GetBackBags(customers[cust].bh, cust);

    send(giveTip[cust]);
  }
  
  Retire(cust);
  return NULL;
}

void *FrontDesk(void *arg) {
  int *pnum = (int *) arg;
  int frontdesk = *pnum, cust;
  free(arg);

  while (true) {
    send(fdAvailable);
    
    wait(checkIn);
    
    cust = shareCust;
    customers[cust].fd = frontdesk;
    customers[cust].room = ++currRoom;
    
    send(custExchanged);

    GiveRoom(customers[cust].fd, cust, customers[cust].room);

    send(giveRoom[cust]);
  }

  return NULL;
}

void *Bellhop(void *arg) {
  int *pnum = (int *) arg;
  int bellhop = *pnum, cust;
  while (true) {
    send(bhAvailable);

    wait(getBH);

    cust = shareCust;
    customers[cust].bh = bellhop;

    send(custExchanged);

    GetBags(customers[cust].bh, cust);

    send(gotBags[cust]);

    wait(entersRoom[cust]);

    GiveBags(customers[cust].bh, cust);

    send(giveBags[cust]);

    wait(giveTip[cust]);
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
