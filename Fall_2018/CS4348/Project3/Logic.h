class Logic {
 private:
  Disk disk;
  int type;

  void changeBitmap(int);
  void freeContiguous(int, pair&);
  pair findFile(std::string);
  int getBlocks(int);
  int getFileSize(const char*);
  void getFileData(std::string, int, int);
 public:
  Logic(Disk&, int);
  void addToFAS(std::string, int, int);
  void deleteFile(std::string);
  void addFile(std::string);
  void write(block, int);
  block read(int);
  void writeFile(std::string, std::string);
};
