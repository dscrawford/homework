#include "project2_dsc160130.h"
int getNumber() {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0,5);
  return distribution(generator);
}
