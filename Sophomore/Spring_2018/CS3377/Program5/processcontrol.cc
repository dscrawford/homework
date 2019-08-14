//Made by Daniel Crawford on 4/7/2018(dsc160130@utdallas.edu)
//CS 3377.501
#include "program5.h"
#include <sys/types.h>
#include <sys/stat.h>

//Will do all forking needed
void processControl(bool isDaemon) {

  if (isDaemon) {
    pid_t forkvalue = fork();
    
    if (forkvalue < 0) {
      std::cerr << "error: There was an error in the fork.. No child created.." << std::endl;
      exit(EXIT_FAILURE);
    }
    else if (forkvalue > 0) { //Exit the parent
      exit(EXIT_SUCCESS);
    }
    umask(0);
    
    pid_t sid = setsid();
    if (sid < 0) {
      exit(EXIT_FAILURE);
    }
    
    //Put the process id into the pid file
    insertPID(sid);
    
    if ((chdir("/")) < 0) {
      exit(EXIT_FAILURE);
    }
    //Close filestreams
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

  }
  
  //Listen for these signals in the program
  signal(SIGHUP, signalHandler);
  signal(SIGTERM, signalHandler);
  signal(SIGINT, signalHandler);

  logfile.open( (PATH + myMap[LOGFILE]).c_str(), std::ios::out | std::ios::app );
  
  while (true) {
    sleep(1);
    //open the logfile
    if (!logfile)
      logfile.open( (PATH + myMap[LOGFILE]).c_str(), std::ios::out | std::ios::app );
    inotifyrun();
    if (logfile)
      logfile.close();
    //Operations go in here.
  }
  
  //Probably won't get here seeing as theres an evil infinite loop looming above this
  exit(EXIT_FAILURE);
}
