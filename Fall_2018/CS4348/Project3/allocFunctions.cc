#include "project3.h"

int getFileSize(const char* filename) {
  std::ifstream file(filename, std::ifstream::in | std::ifstream::ate);
  if (!file.is_open())
    return -1;
  
  file.seekg(0, std::ios::end);
  int file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  file.close();
  
  return file_size;
}

void freeContiguous(Disk disk, int fileSize, pair& contiguous) {
  contiguous.beg = -1;
  contiguous.end = -1;
  int sizeToAllocate = fileSize, i;
  std::cout << sizeToAllocate << std::endl;
  //iterate through the bitmap to find open space
  for (i = 2; i < MAXBLOCKSIZE && sizeToAllocate != 0; ++i) {
    std::cout << "iteration: " << i << std::endl;
    //if the bit signals the beginning of an open partition.
    if (disk.read(1).bytes[i] == '0' && contiguous.beg == -1)
      contiguous.beg = i;
    //if the bit signals that there is still open space left
    if (disk.read(1).bytes[i] == '0')
      sizeToAllocate--;
    //current contiguous chain does not have enough space
    else {
      contiguous.beg = -1;
      sizeToAllocate = fileSize;
    }
    //if current contiguous chain has enough space
  }
  contiguous.end = i - 1;
  std::cout << contiguous.beg << ", " << contiguous.end << std::endl;

  //pass arr if not enough
  if (sizeToAllocate != 0) {
    contiguous.beg = -1;
    contiguous.end = -1;
  }
}

int getBlocks(int fileSize) {
  return ceil((double)fileSize / MAXBLOCKSIZE);
}
