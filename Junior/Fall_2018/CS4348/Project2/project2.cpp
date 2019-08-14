/* Made by Daniel Crawford on 10/18/2018
 * Email: dsc160130@utdallas.edu 
 * Project simulates hotel interaction between guests, front desk employees and
 * bellhops using threads(in parallel with semaphores)
 */
#include "project2.h"
//Semaphores
sem_t checkIn, fdAvailable, bhAvailable, custExchanged, getBH,
  gotBags[NUM_CUSTOMERS], giveBags[NUM_CUSTOMERS], entersRoom[NUM_CUSTOMERS],
  giveTip[NUM_CUSTOMERS], giveRoom[NUM_CUSTOMERS];

//shareCust = variable that can be exchanged
//currRoom  = starting from 0, variable that identifies each room
int shareCust,  currRoom;

customer customers[NUM_CUSTOMERS];

int main() {

  //initialize values
  shareCust = -1;
  currRoom  = 0;

  //Create thread arrays
  int status;
  pthread_t customerThreads [NUM_CUSTOMERS];
  pthread_t frontdeskThreads[NUM_FRONTDESK];
  pthread_t bellhopThreads  [NUM_BELLHOP];


  /*
   * INITIALIZE ALL SEMAPHORES
   */
  init_semaphore(checkIn,0);
  init_semaphore(fdAvailable,0);
  init_semaphore(bhAvailable,0);
  init_semaphore(custExchanged,1); //Needs to be initialized to 1
  init_semaphore(getBH,0);
  for (int i = 0; i < NUM_CUSTOMERS; ++i) {
    init_semaphore(giveRoom[i], 0);
    init_semaphore(gotBags[i], 0);
    init_semaphore(giveBags[i], 0);
    init_semaphore(entersRoom[i], 0);
    init_semaphore(giveTip[i], 0);
  }

  /*
   * START SIMULATION
   */
  printf("Simulation starts\n");

  //Create all the front desk employees
  for (int frontdesk_count = 0; frontdesk_count < NUM_FRONTDESK;
       ++frontdesk_count) {
    //set variable pnum to the current increment
    int *pnum = (int*)malloc(sizeof(int));
    *pnum = frontdesk_count;
    status = pthread_create(&frontdeskThreads[frontdesk_count], NULL, FrontDesk,
			    (void*)pnum);
    printf("Front desk employee %d created\n", *pnum);
    //if error status
    if (status != 0) {
      printf("Create thread\n");
      exit(1);
    }
  }

  //Create all the bellhops
  for (int bellhop_count = 0; bellhop_count < NUM_BELLHOP; ++bellhop_count) {
    //set variable pnum to the current increment
    int *pnum = (int*)malloc(sizeof(int));
    *pnum = bellhop_count;
    status = pthread_create(&bellhopThreads[bellhop_count], NULL, Bellhop,
			    (void*)pnum);
    printf("Bellhop %d created\n", *pnum);
    //if error status
    if (status != 0) {
      printf("Create thread\n");
      exit(1);
    }
  }

  //Create all the customers
  for (int customer_count = 0; customer_count < NUM_CUSTOMERS; ++customer_count) {
    int *pnum = (int*)malloc(sizeof(int));
    //set variable pnum to the current increment
    *pnum = customer_count;
    status = pthread_create(&customerThreads[customer_count], NULL, Customer,
			    (void*)pnum);
    printf("Guest %d created\n", *pnum);
    //if error status
    if (status != 0) {
      printf("Create thread\n");
      exit(1);
    }
  }

  //Join all the customer threads together.
  joinThreads(customerThreads, NUM_CUSTOMERS);

  printf("Simulation ends\n");
  /*
   * END SIMULATION
   */
  return 0;
}
