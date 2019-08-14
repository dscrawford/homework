#ifndef MEMINSTRUCT_H
#define MEMINSTRUCT_H

#include <vector>

class RAM {
 private:
  const static int SIZE = 2000;
  int* arr;
 public:
  RAM();
  int read(int);
  void write(int, int);
  ~RAM();
};

#endif // MEMINSTURCT_H
