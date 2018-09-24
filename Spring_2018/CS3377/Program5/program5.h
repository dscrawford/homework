//Made by Daniel Crawford(dsc160130@utdallas.edu) on 4/8/2018
//CS 3377.501
#ifndef PROGRAM5_H
#define PROGRAM5_H

#include <iostream>
#include <map>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <csignal>
#include <unistd.h>

std::map<int,std::string> parsecmd(int, char**);

void getconf(bool);
void signalHandler(int);
void insertPID(pid_t);
void processControl(bool);
void inotifyrun();

enum mapVals {DAEMON, FILENAME, VERBOSE, LOGFILE, PASSWORD, NUMVERSIONS, WATCHDIR};

extern std::map<int, std::string> myMap;
extern std::map<std::string, int> versions_count;
extern std::string PATH;
extern std::ofstream logfile, pidfile;

#endif /* PROGRAM5_H */
