//Made by Daniel Crawford on 4/7/2018(dsc160130@utdallas.edu)
//CS 3377.501
#include "program5.h"
#include<sys/stat.h>

std::map<int,std::string> myMap;

std::string PATH;
std::ofstream logfile, pidfile;

int main(int argc, char* argv[]) {
  
  //Find the PATH to the current directory
  FILE* getPath = popen("pwd","r");
  char c;
  while ( (c = fgetc(getPath)) != EOF ) {
    if (c != '\n')
      PATH += c;
  }
  PATH += "/";

  //If cs3377dirmond.pid exists, then exit this program. Otherwise, open a pidfile.  
  std::ifstream testfile;
  testfile.open( (PATH + "cs3377dirmond.pid").c_str() );
  if (testfile) {
    std::cerr << "error: another process is current running" << std::endl;
    testfile.close();
    exit(EXIT_FAILURE);
  }
  else {
    pidfile.open( (PATH + "/cs3377dirmond.pid").c_str());
  }
  
  //Try to parse the commandline and config file first, then try to fork the value.
  //Both functions put values into the map
  myMap = parsecmd(argc,argv);
  //Call getconf and tell the function that this is not inside the daemon.
  getconf(false);
  
  //Create a string that will contain results of shell command that tells the user if
  //folder .versions exists. String will contain either "0" or "1"
  FILE* checkfile = popen(("[ ! -e " + PATH + "tempdir/.versions ]; echo $?" ).c_str(), "r");
  std::string checkstr;
  while ( (c = fgetc(checkfile)) != EOF) {
    if (c != '\n')
      checkstr += c;
  }
  //if returns 0, file does not exist
  if ( checkstr == "0" )
    mkdir( ( myMap[WATCHDIR] + "/.versions").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

  processControl(myMap[DAEMON] == "true");


  //Should not get here.
  return 0;
}
