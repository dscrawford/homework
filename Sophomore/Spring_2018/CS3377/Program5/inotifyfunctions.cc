
//Made by Daniel Crawford on 4/7/2018(dsc160130@utdallas.edu)
//CS 3377.501
#include "program5.h"
#include <sys/inotify.h>
#include <string.h>
#include <string>

#define EVENT_SIZE    ( sizeof (struct inotify_event) )
#define BUF_LEN       ( 1024 * (EVENT_SIZE + 16) )
#define MAX_DATE_SIZE 30

int inotifyCreate(char[], int&, int&);
std::string getDate();
void tooManyVersions(std::string);
void removeOldest(std::string);

//function returns true the instantiation of inotify works.
void inotifyrun() {
  char buffer[BUF_LEN];
  int fd, wd, i = 0,
    length = inotifyCreate(buffer, fd, wd);

  std::string date = getDate();

  while ( i < length ) {
    struct inotify_event *event = ( struct inotify_event * ) &buffer[i];
    //If the file coming through isnt a backup file of any sort, then proceed.
    if ( event->name[strlen(event->name) - 1] != '~' && strncmp(event->name, ".#", strlen(".#")) != 0 ) {
      if (event->len) {
	//If modified or closed then written to
	if (event->mask & IN_MODIFY || event->mask & IN_CLOSE_WRITE) {
	    
	  if (event->mask & IN_ISDIR)
	    logfile << "The directory " << event->name << " was modified." << std::endl;
	  else
	    logfile << "The file " << event->name << " was modified." << std::endl;

	  //Check if there are too many versions of the filename, remove oldest files.
	  tooManyVersions(std::string(event->name));
	  
	  //Display what backup is being created if user
	  if (myMap[VERBOSE] == "true")
	    logfile << "Creating backup " << event->name << "." << date << std::endl;
	  
	  //Call cp through system() and put new versions of file with date appended to it.
	  system( ("cp " + myMap[WATCHDIR] + "/" + event->name + " " + myMap[WATCHDIR]
		   + "/.versions/" + event->name + "." + date).c_str() );	
	}
	//If created
	if (event->mask & IN_CREATE) {
	  
	  if (event->mask & IN_ISDIR)
	    logfile << "The directory " << event->name << " was created." << std::endl;
	  else
	    logfile << "The file " << event->name  << " was created." << std::endl;
	  
	}
	//If deleted
	if (event->mask & IN_DELETE) {
	  
	  if (event->mask & IN_ISDIR)
	    logfile << "The directory " << event->name << " was deleted." << std::endl;
	  else
	    logfile << "The file " << event->name  << " was deleted." << std::endl;
	  
	}
	
      }
      //If modified, make a new version of the folder.
    }
    //Go to next buffer
    i += EVENT_SIZE + event->len;
  }
  inotify_rm_watch(fd,wd);
  close(fd);
}

//Create the notifier, and gives information and what has been changed.
int inotifyCreate(char buffer[BUF_LEN], int& fd, int &wd) {
  fd = inotify_init();
  if (fd < 0)
    return 0;

  wd = inotify_add_watch (fd, myMap[WATCHDIR].c_str(), IN_MODIFY | IN_CREATE | IN_DELETE | IN_CLOSE_WRITE);
  if (wd < 0)
    return 0;;

  int length = read (fd, buffer, BUF_LEN);
  if (length < 0)
    return 0;

  return length;
}

//Return a string which contains the current date as YYYY.MM.DD-HH:MM:SS
std::string getDate() {
  FILE* dateReader =
    popen("date +%y.%m.%d-%H:%M:%S","r");
  char str[MAX_DATE_SIZE];
  fgets(str, MAX_DATE_SIZE, dateReader);
  return std::string(str);
}

void tooManyVersions(std::string filename) {
  //Find files with the same pattern as that file
  //find /path/to/.versions -n "filename*" | wc -l
  FILE* filestream =
    popen( ("find " + myMap[WATCHDIR] + "/.versions/ " +
	    "-name \"" + filename + "*\" | wc -l").c_str(),"r");
  std::string str;
  char c;
  while ( (c = fgetc(filestream)) != EOF ) {
    str += c;
  }

  
  int i = atoi(str.c_str());

  if (myMap[VERBOSE] == "true")
    logfile << "Current copies of " << filename << ": " << i << std::endl;

  //If current # of backups for file is greater than limit.
  while (i >= atoi(myMap[NUMVERSIONS].c_str())) {
    removeOldest(filename);
    i--;
  }
}

void removeOldest(std::string filename) {
  FILE* filestream =
    popen ( ("find " + myMap[WATCHDIR] + "/.versions/ -name \"" +
	     filename + "*\" -printf \"%p\\n\" | sort | " +
	     "head -n 1").c_str(), "r");
  std::string oldfile;
  char c;
  while ( (c = fgetc(filestream)) != EOF) {
    if (c != '\n')
      oldfile += c;
  }

  if (myMap[VERBOSE] == "true")
    logfile << "Removing backup file " << oldfile << std::endl;

  //Remove the oldest file(by unix timestamp)
  remove(oldfile.c_str());
}
