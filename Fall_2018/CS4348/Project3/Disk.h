class Disk {
 private:
  block *data;
  int allocationMethod;
 public:
  Disk(int);
  block read(int);
  void write(block, int, int);
  void deleteFile(std::string);
  void addToFAS(std::string, int, int);

};
