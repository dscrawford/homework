#include "project2_dsc160130.h"
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

void init_semaphore(sem_t& semaphore, int initialValue) {
  if (sem_init (&semaphore,0,initialValue) == -1) {
    printf("ERROR: Failed to initialize semaphore.");
    exit(1);
  }
}
