#include "project3.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "ERROR: Invalid arguments count (Should be in format "
	      << "./project3 [contiguous|indexed|chained]" << std::endl;
    return 1;
  }

  int type = getType(std::string(argv[1]));
  std::cout << "type: " << type << std::endl;
  Disk disk;
  Logic logic(disk, type);

  
  int choice = 0;
  while (choice != 8) {
    printOptions();
    choice = getInt();

    switch(choice) {
    case 1:
      {
	std::cout << "Enter file name: ";
	std::string file_name = getStr();
	std::cout << logic.getFile(file_name) << std::endl;
      }
      break;
    case 2:
      std::cout << logic.ReadBlock(FILEALLOC).bytes << std::endl;
      break;
    case 3:
      {
	block b = logic.ReadBlock(BITMAP);
	for (int i = 0; i < MAXBLOCKS / 32; ++i) {
	  for (int j = 0; j < 32; ++j) {
	    std::cout << b.bytes[j+(i*32)];
	  }
	  std::cout << std::endl;
	}
      }
      break;
    case 4:
      {
	std::cout << "Enter block index: ";
	int index = getInt();
	std::cout << logic.ReadBlock(index).bytes << std::endl;
      }
      break;
    case 5:
      {
	std::cout << "Enter file to write: ";
	std::string input = getStr();
	logic.writeRealFile(input);
      }
      break;
    case 6:
      {
	std::string fromFile, toFile;
	std::cout << "Copy from: ";
	fromFile = getStr();
	std::cout << "Copy to: ";
	toFile = getStr();
	logic.WriteFile(fromFile, toFile);
      }
      break;
    case 7:
      {
	std::cout << "Enter file to remove: ";
	std::string file_name = getStr();
	logic.deleteFile(file_name);
      }
      break;
    case 8:
      break;
    default:
      std::cerr << "ERROR: Invalid input" << std::endl;
    }
  }
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
