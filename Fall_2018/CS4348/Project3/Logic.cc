#include "project3.h"

int getDigits(int number) {
  int numDigits = 0;
  while (number) {
    numDigits /= 10;
    numDigits++;
  }
  return numDigits;
}

Logic::Logic(Disk& disk, int type) {
  if (type != CONTIGUOUS && type != CHAINED && type != INDEXED) {
    std::cerr << "ERROR: Invalid type" << std::endl;
    exit(1);
  }
  this->disk = disk;
  this->type = type;

  block b;
  b.bytes = std::string(MAXBLOCKS, '0');
  b.bytes[0] = '1';
  b.bytes[1] = '1';
  disk.write(b, BITMAP);

  //If chained or indexed, generate open space and shuffle where it could go.
  if (type == CHAINED || type == INDEXED) {
    for (int i = 2; i < MAXBLOCKS; ++i) {
      openSpace.push_back(i);
    }
    std::random_shuffle(openSpace.begin(), openSpace.end());
  }
}

void Logic::addToFAS(std::string file_name, int start, int length) {
  block b = ReadBlock(FILEALLOC);
  b.bytes += file_name + "\t" + std::to_string(start)
    + "\t" + std::to_string(length) + "\n";
  WriteBlock(b, FILEALLOC);
}

void Logic::deleteFile(std::string file_name) {
  block b = ReadBlock(FILEALLOC);
  if (b.bytes == "") {
    std::cerr << "ERROR: No files to delete" << std::endl;
    return;
  }
  pair p = findFile(file_name);
  if (p.beg == -1) {
    std::cout << "ERROR: File to delete not found" << std::endl;
    return;
  }

  std::string file_info = b.bytes.substr(p.beg, p.end);
  int start, file_size;
  GetFileAllocData(file_info, start, file_size);
  int blocks = getBlocks(file_size);

  //
  //Contiguous implementation
  //
  if (type == CONTIGUOUS) {
    for (int i = start; i < start + blocks; ++i)
      DeleteBlock(i);
  }
  //
  //
  //
  else if (type == CHAINED) {
    int pointerSize = getDigits(MAXBLOCKS);
    int address = start;
    do {
    //Read first pointer
      int i = address;
      address = std::stoi( ReadBlock(start).bytes.substr(0, pointerSize) );
      DeleteChainedOrIndexedBlock(i);
    } while (address != 0);
    
  }
  else if (type == INDEXED) {
  }

  b.bytes.erase(p.beg, file_info.length());
  WriteBlock(b, FILEALLOC);
}

block Logic::ReadBlock(int i) {
  if (i >= 0 && i < MAXBLOCKS)
    return disk.read(i);
  else
    std::cerr << "ERROR: Index out of range" << std::endl;
  return *(new block);
}

void Logic::DeleteBlock(int i) {
  disk.write(*(new block), i);
  changeBitmap(i);
}
void Logic::WriteBlock(block b, int i) {
  disk.write(b, i);
  changeBitmap(i);
}

void Logic::WriteFile(std::string fromFile, std::string toFile) {
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

  if (type == CONTIGUOUS) {
    pair contiguous = freeContiguous(blocks);
    
    //If there was no contiguous space available
    if (contiguous.beg < 0) {
      std::cerr << "ERROR: Not enough contiguous space available" << std::endl;
      return;
    }


    
    int spaceToAllocate = file_size;
    for (int i = contiguous.beg; i <= contiguous.end; ++i) {
      //Get remaining space needed to allocate in 256 size blocks
      int space = (spaceToAllocate > MAXBLOCKSIZE)
	? MAXBLOCKSIZE : spaceToAllocate;
      spaceToAllocate -= MAXBLOCKSIZE;
      block b;
      char* cstr = new char[space];
      file.read(cstr, space);
      b.bytes = std::string(cstr);
      std::cout << "Made it here!" << std::endl;
      WriteBlock(b, i);
    }

    addToFAS(toFile, contiguous.beg, file_size);
  }

  else if (type == CHAINED) {
    int blocks = getBlocks(file_size);
    if ( freeSpace() < blocks ) {
      std::cerr << "ERROR: Not enough space available" << std::endl;
      return;
    }

    int pointerSize = getDigits(MAXBLOCKS), spaceToAllocate = file_size,
      start = openSpace.front();

    std::cout << pointerSize << std::endl;
    for (int i = 0; i < blocks; ++i) {
      //Get remaining space needed to allocate in 256 size blocks
      int space = (spaceToAllocate > MAXBLOCKSIZE - pointerSize)
	? MAXBLOCKSIZE - pointerSize : spaceToAllocate;
      //Space needed to allocate
      spaceToAllocate -= (MAXBLOCKSIZE + pointerSize);
      block b;
      char* cstr = new char[space];

      //Write a pointer at the beginning of the file.
      if (i == blocks - 1)
	b.bytes = "000";
      else {
	std::stringstream ss;
	ss << std::setfill('0') << std::setw(pointerSize) << openSpace.front();
	b.bytes = ss.str();
      }

      file.read(cstr, space);
      b.bytes += std::string(cstr);
      std::cout << b.bytes;
      WriteRandBlock(b);
    }
    addToFAS(toFile, start, file_size);
  }

  else if (type == INDEXED) {
    
  }


  
  file.close();
}

void Logic::changeBitmap(int index) {
  block b = ReadBlock(BITMAP);
  if (b.bytes[index] == '0')
    b.bytes[index] = '1';
  else
    b.bytes[index] = '0';

  WriteBlock(b, BITMAP);
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

pair Logic::freeContiguous(int fileSize) {
  pair contiguous;
  contiguous.beg = -1;
  contiguous.end = -1;
  int sizeToAllocate = fileSize, i;
  //iterate through the bitmap to find open space
  for (i = 2; i < MAXBLOCKSIZE && sizeToAllocate != 0; ++i) {
    //if the bit signals the beginning of an open partition.
    if (ReadBlock(1).bytes[i] == '0' && contiguous.beg == -1)
      contiguous.beg = i;
    //if the bit signals that there is still open space left
    if (ReadBlock(1).bytes[i] == '0')
      sizeToAllocate--;
    //current contiguous chain does not have enough space
    else {
      contiguous.beg = -1;
      sizeToAllocate = fileSize;
    }
    //if current contiguous chain has enough space
  }
  contiguous.end = i - 1;

  //pass arr if not enough
  if (sizeToAllocate != 0) {
    contiguous.beg = -1;
    contiguous.end = -1;
  }

  return contiguous;
}

int Logic::getBlocks(int fileSize) {
  return ceil((double)fileSize / MAXBLOCKSIZE);
}

pair Logic::findFile(std::string file_name) {
  pair pos;
  pos.beg = -1;
  pos.end = -1;
  std::string str;
  std::stringstream ss(ReadBlock(FILEALLOC).bytes);
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

//Split FileData string into three strings and find start and length.
void Logic::GetFileAllocData(std::string file_info, int& start, int& length) {
  char *pch;
  pch = strtok((char*)file_info.c_str(),"\t");
  pch = strtok(NULL,"\t");
  start = stoi(std::string(pch));
  pch = strtok(NULL,"\n");
  length = stoi(std::string(pch));
}

std::string Logic::getFile(std::string file_name) {
  std::string file_data = "";
  int start, length;
  pair pos = findFile(file_name);
  if (pos.beg != -1) {
    std::string file_info = ReadBlock(FILEALLOC).bytes.substr(pos.beg,pos.end);
    GetFileAllocData(file_info, start, length);
    int blocks = getBlocks(length);
    
    if (type == CONTIGUOUS) {
      for (int i = start; i < start + blocks; ++i) {
	file_data += ReadBlock(i).bytes;
      }
    }
    
    else if (type == CHAINED) {
      int pointerSize = getDigits(MAXBLOCKS);
      int address = start;
      do {
	//Read first pointer
	int i = address;
	address = std::stoi( ReadBlock(start).bytes.substr(0, pointerSize) );
	block b = ReadBlock(i);
	file_data += b.bytes.substr(pointerSize, b.bytes.length());
      } while (address != 0);
    }
  return file_data;
  }
}

void Logic::writeRealFile(std::string file_name) {
  std::string file_data = getFile(file_name);
  if (file_data != "") {
    std::fstream file(file_name, std::ios::out);
    file << file_data;
  }
  else
    std::cerr << "ERROR: File " << file_name << " is either empty or does not"
	      << " exist." << std::endl;
}

void Logic::WriteRandBlock(block b) {
  int index = openSpace.front();
  openSpace.erase(openSpace.begin());
  WriteBlock(b, index);
}

void Logic::DeleteChainedOrIndexedBlock(int i) {
  openSpace.push_back(i);
  DeleteBlock(i);
}

int Logic::freeSpace() {
  if (type == CHAINED || type == INDEXED) 
    return openSpace.size();

  int free_space = 0;
  std::string bitmap = ReadBlock(BITMAP).bytes;
  for (int i = 0; i < bitmap.length(); ++i) {
    if (bitmap[i] == '0')
      free_space++;
  }

  return free_space;
}
