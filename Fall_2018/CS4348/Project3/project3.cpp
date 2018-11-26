#include "project3.h"

int main(int argc, char** argv) {
  if (argc > 2) {
    std::cerr << "ERROR: Too many arguments (Should be in format "
	      << "./project3 [consecutive|indexed|chained]" << std::endl;
    return 1;
  }
  Disk disk(0);
  int choice = 0;

  while (choice != 8) {
    printOptions();
    choice = getInt();

    switch(choice) {
    case 1:
      break;
    case 2:
      std::cout << disk.read(0).bytes << std::endl;
      break;
    case 3:
      {
	for (int i = 0; i < MAXBLOCKS / 32; ++i) {
	  for (int j = 0; j < 32; ++j) {
	    std::cout << disk.read(1).bytes[j+(i*32)];
	  }
	  std::cout << std::endl;
	}
      }
      break;
    case 4:
      break;
    case 5:
      break;
    case 6:
      copyFileFromRealSystemToSimulation(disk);
      break;
    case 7:
      std::cout << "hello" << std::endl;
      break;
    case 8:
      std::cout << "end" << std::endl;
      break;
    default:
      std::cerr << "ERROR: Invalid input, re!!!!!!!!!!!!" << std::endl;
    }
  }
}

void copyFileFromRealSystemToSimulation(Disk &disk) {
  std::string fromFile, toFile;
  std::cout << "Copy from: ";
  fromFile = getStr();
  std::cout << "Copy to: ";
  toFile = getStr();
  
  if (fromFile == "" && toFile == "")
    std::cerr << "ERROR: Need a name" << std::endl;

  const int file_size = getFileSize(fromFile.c_str()),
    blocks = getBlocks(file_size);
  std::fstream file(fromFile, std::ofstream::in);
  
  if (!file.is_open()) {
    std::cerr << "ERROR: File \"" <<  fromFile << "\" failed to open."
	      << std::endl;
    return;
  }
  if (blocks > 10 || blocks <= 0) {
    std::cerr << "ERROR: File is empty or too large(10 blocks max)"
	      << std::endl;
    return;
  }

  pair contiguous;
  freeContiguous(disk, blocks, contiguous);

  //If there was a contiguous space available
  if (contiguous.beg < 0) {
    std::cerr << "ERROR: Not enough space available" << std::endl;
    return;
  }
  
  int spaceToAllocate = file_size;
  for (int i = contiguous.beg; i <= contiguous.end; ++i) {
    //Get remaining space needed to allocate in 256 size blocks
    //(ignore null terminator)
    int space = (spaceToAllocate > MAXBLOCKSIZE)
      ? MAXBLOCKSIZE : spaceToAllocate - 1;
    spaceToAllocate -= MAXBLOCKSIZE;
    block b;
    char* cstr = new char[space];
    file.read(cstr, space);
    b.bytes = std::string(cstr);
    std::cout <<"Writing to address " << i
	      << ", with space " << space << std::endl;
    disk.write(b, i, 0);
  }

  disk.addToFAS(toFile, contiguous.beg, file_size);
  
  file.close();
}

void printOptions() {
  std::cout << "1) Display a file\n"
	    << "2) Display the file table\n"
	    << "3) Display the free space bitmap\n"
	    << "4) Display a disk block\n"
	    << "5) Copy a file from the simulation to a file on the real system\n"
	    << "6) Copy a file from the real system to a file in the simulation\n"
	    << "7) Delete a file\n"
	    << "8) Exit" << std::endl;
}
