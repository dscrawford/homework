class Logic {
 private:
  Disk disk;
  int type;
  std::vector<int> openSpace;

  void addToFAS(std::string, int, int);
  void GetFileAllocData(std::string, int&, int&);
  int getBlocks(int);
  int getFileSize(const char*);
  int freeSpace();
  pair freeContiguous(int);
  pair findFile(std::string);

  void DeleteChainedOrIndexedBlock(int);
  void WriteRandBlock(block);
  void DeleteBlock(int);
  void changeBitmap(int);

 public:
  Logic(Disk&, int);
  void WriteBlock(block, int);
  block ReadBlock(int);
  void deleteFile(std::string);
  void addFile(std::string);
  void WriteFile(std::string, std::string);
  std::string getFile(std::string);
  void writeRealFile(std::string);
};
