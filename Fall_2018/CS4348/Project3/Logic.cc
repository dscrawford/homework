#include "project3.h"

Logic::Logic(Disk& disk, int type) {
  this->disk = disk;
  this->type = type;

  block b;
  b.bytes = std::string(MAXBLOCKS, '0');
  b.bytes[0] = '1';
  b.bytes[1] = '1';
  disk.write(b, BITMAP);
}

void Logic::addToFAS(std::string file_name, int start, int length) {
  block b = disk.read(FILEALLOC);
  b.bytes += file_name + "\t" + std::to_string(start)
    + "\t" + std::to_string(length) + "\n";
  disk.write(b, FILEALLOC);
}

void Logic::deleteFile(std::string file_name) {
  block b = disk.read(FILEALLOC);
  if (b.bytes == "") {
    std::cerr << "ERROR: No files to delete" << std::endl;
    return;
  }
  pair p = findFile(file_name);
  if (p.beg == -1) {
    std::cout << "ERROR: File to delete not found" << std::endl;
    return;
  }

  std::string file_info
  std::queue<int> blocksToDelete;
  int start, length;
  getFileData(file_info, start, length);
  
  //Contiguous implementation
  //
  for (int i = start; i < ceil( ((double)start + length)/MAXBLOCKSIZE); ++i) {
    block b;
    disk.write(b, i);
    changeBitmap(i);
  }
  //
  //

  b.bytes.erase(p.beg, file_info.length());
  disk.write(b, FILEALLOC);
}

block Logic::read(int i) {
  return disk.read(i);
}

void Logic::write(block b, int i) {
  disk.write(b, i);
  changeBitmap(i);
}

void Logic::writeFile(std::string fromFile, std::string toFile) {
  if (fromFile == "" && toFile == "") {
    std::cerr << "ERROR: Need a name" << std::endl;
    return;
  }

  const int file_size = getFileSize(fromFile.c_str()),
    blocks = getBlocks(file_size);
  std::fstream file(fromFile, std::ifstream::in);
  
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
  freeContiguous(blocks, contiguous);

  //If there was no contiguous space available
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
    disk.write(b, i);
    changeBitmap(i);
  }

  addToFAS(toFile, contiguous.beg, file_size);
  
  file.close();
}

void Logic::changeBitmap(int index) {
  block b = disk.read(BITMAP);
  if (b.bytes[index] == '0')
    b.bytes[index] = '1';
  else
    b.bytes[index] = '0';

  disk.write(b, BITMAP);
}
int Logic::getFileSize(const char* file_name) {
  std::ifstream file(file_name, std::ifstream::in | std::ifstream::ate);
  if (!file.is_open())
    return -1;
  
  file.seekg(0, std::ios::end);
  int file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  file.close();
  
  return file_size;
}

void Logic::freeContiguous(int fileSize, pair& contiguous) {
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

int Logic::getBlocks(int fileSize) {
  return ceil((double)fileSize / MAXBLOCKSIZE);
}

pair Logic::findFile(std::string file_name) {
  pair pos;
  pos.beg = -1;
  pos.end = -1;
  std::string str;
  std::stringstream ss(disk.read(FILEALLOC).bytes);
  int lengthCovered = 0;
  while (getline(ss, str)) {
    if (str.substr(0, str.find("\t")) == file_name) {
      pos.beg = lengthCovered;
      pos.end = lengthCovered + str.length() + 1;
      break;
    }
    lengthCovered += (str.length() + 1); //length of current str plus newline
  }
  
  return pos;
}

void Logic::getFileData(std::string file_info, int& start, int& length) {
  std::string file_info = b.bytes.substr(p.beg, p.end);
  std::string vals[2];
  int pos = file_info.substr(0, file_info.find("\t")).length();
  vals[0] = file_info.substr(pos, file_info.find(pos, "\t"));
  pos += file_info.find(pos, "\t");
  vals[1] = file_info.substr(pos, file_info.length());
  start = std::stoi(vals[0]);
  length = std::stoi(vals[1]);
}
