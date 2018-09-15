#ifndef READINPUT_H
#define READINPUT_H

class InputReader { 
 private:
  fstream file;
 public:
  InputReader(std::string FILENAME);
  file.open(FILENAME);
  if (!file) {
    std::cout << "File failed to open" << std::endl;
  }

  int read() {
    
  }
};

#endif // READINPUT_H
