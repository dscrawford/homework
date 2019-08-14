//Made by Daniel Crawford on 4/8/2017 (dsc160130@utdallas.edu)
//CS 3377.501
#include "program5.h"
#include <fstream>

void insertPID(int sid) {
  //Check if pidfile is open, then do operations.
  if (!pidfile)
    pidfile << "error: unable to open cs3377dirmond.pid" << std::endl;
  else {
    pidfile << sid << std::endl;
    pidfile.close();
  }
}

void signalHandler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {

    //Find where the file that needs to be removed is.
    std::string removefile = PATH + std::string("cs3377dirmond.pid");

    remove( removefile.c_str() );
    //Close the logfile.
    logfile.close();

    

    exit(EXIT_SUCCESS);
  }
  else if (signal == SIGHUP) {
    //re-read the configure file, pass true because it will be running from a daemon.
    getconf(true);

    //Close the previous logfile and open the new one.
    logfile << "I DIDNT GET CLOSED" << std::endl;
    if (logfile)
      logfile.close();
    logfile.open( (PATH + myMap[LOGFILE]).c_str(), std::ios::out | std::ios::app);
  }
  
}
