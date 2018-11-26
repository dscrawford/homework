#include "project3.h"

Disk::Disk(int allocationMethod) {
  data = new block[MAXBLOCKS];
  data[BITMAP].bytes = std::string(MAXBLOCKS, '0');
  data[BITMAP].bytes[0] = '1';
  data[BITMAP].bytes[1] = '1';
}

block Disk::read(int i) {
  return data[i];
}

void Disk::write(block b, int i, int type) {
  if (b.bytes.length() > MAXBLOCKS) {
    std::cerr << "ERROR: Block too large" << std::endl;
  }
  data[i] = b;
  data[BITMAP].bytes[i] = '1';
}

void Disk::addToFAS(std::string file_name, int start, int length) {
  block b;
  b.bytes = file_name + "\t" + std::to_string(start) + "\t" +
    std::to_string(length) + "\n";
  data[FILEALLOC].bytes += b.bytes;
}

void Disk::deleteFile(std::string file_name) {
  std::string delim1 = "\t", delim2 = "\n", token;
  int pos = 0;
  while ( (token = data[FILEALLOC].bytes.substr(pos,
		   data[FILEALLOC].bytes.find(delim1))) != file_name &&) {
    pos = pos + data[FILEALLOC].bytes.substr(pos,data[FILEALLOC].bytes.find(delim2))
      + 1;
  }
  data[FILEALLOC].bytes.erase(pos, data[FILEALLOC].bytes
  block b;
  data[i] = b;
  data[BITMAP].bytes[i] = '0';
  */
}
