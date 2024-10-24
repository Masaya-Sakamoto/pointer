#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NOISE_SIZE 1024
#define PRINT_READ
#define PRINT_WRITE

static float noise[NOISE_SIZE];
static pthread_mutex_t noise_mutex;  //  = PTHREAD_MUTEX_INITIALIZER;

void intNoise() {
  pthread_mutex_lock(&noise_mutex);
  for (int i = 0; i < NOISE_SIZE; i++) noise[i] = rand() / (float)RAND_MAX;
  pthread_mutex_unlock(&noise_mutex);
}

static void *writeNoise() {
  static unsigned int writeCounter = 0;
  while (1) {
    pthread_mutex_lock(&noise_mutex);
    noise[writeCounter % NOISE_SIZE] = rand() / (float)RAND_MAX;
    writeCounter = (writeCounter + 1) % __UINT32_MAX__;
    pthread_mutex_unlock(&noise_mutex);
#ifdef PRINT_WRITE
    printf("write %4d\n", writeCounter % NOISE_SIZE);
    fflush(stdout);
#endif
  }

  return NULL;
}

static void *readNoise() {
  static unsigned int readCounter = 0;
  float rnoise;
  while (1) {
    pthread_mutex_lock(&noise_mutex);
    rnoise = noise[readCounter % NOISE_SIZE];
    readCounter = (readCounter + 1) % __UINT32_MAX__;
    pthread_mutex_unlock(&noise_mutex);
#ifdef PRINT_READ
    printf("read #%4d    %f\n", readCounter % NOISE_SIZE, rnoise);
    fflush(stdout);
#endif
  }

  return NULL;
}

int main() {
  intNoise();
  // Create threads
  pthread_t wtid, rtid;
  pthread_create(&wtid, NULL, writeNoise, NULL);
  pthread_create(&rtid, NULL, readNoise, NULL);

  // Join threads (not really needed here as they run infinitely, but good
  // practice)
  pthread_join(wtid, NULL);
  pthread_join(rtid, NULL);

  return 0;
}
