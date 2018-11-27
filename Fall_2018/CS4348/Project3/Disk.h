class Disk {
 private:
  block *data;
  int allocationMethod;
 public:
  Disk();
  block read(int);
  void write(block, int);
};
