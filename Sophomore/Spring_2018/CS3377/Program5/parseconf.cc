/*Made by Daniel Crawford(dsc160130@utdallas.ed) on 4/8/2018
 *CS 3377.501
 */
#include "program5.h"
#include <rude/config.h>

//Read the configuration file and store it into map called myMap
void getconf(bool fromDaemon) {
  try {
    
    rude::Config config;
    config.load( (PATH + myMap[FILENAME]).c_str());
    config.setSection("Parameters");

    //Check if the configuration file is valid, if not, send an error message to the user.
    if (config.exists("Verbose") && config.exists("LogFile") && config.exists("Password")
	&& config.exists("NumVersions") && config.exists("WatchDir")) {

      //Set boolean value for verbosity
      if (config.getBoolValue("Verbose"))
	myMap[VERBOSE] = "true";
      else
	myMap[VERBOSE] = "false";

      //Store rest of the values inside the map.
      myMap[LOGFILE] = std::string(config.getStringValue("LogFile"));
      myMap[NUMVERSIONS] = std::string(config.getStringValue("NumVersions"));
      if (!fromDaemon) {
	myMap[WATCHDIR] = std::string(config.getStringValue("WatchDir"));
	myMap[PASSWORD] = std::string(config.getStringValue("Password"));
      }
      else {
	if (myMap[WATCHDIR] != std::string(config.getStringValue("WatchDir")) ) {
	  logfile << "error: Cannot change directory being watched while running" << std::endl;
	  config.setStringValue("Watchdir",myMap[WATCHDIR].c_str());
	}
	if (myMap[PASSWORD] != std::string(config.getStringValue("Password"))) {
	  logfile << "error: Cannot change the password while running" << std::endl;
	  config.setStringValue("Password",myMap[PASSWORD].c_str());
	}
      }

     
      
    }
    else {
      if (logfile)
	logfile << "error: Invalid configuration file \"" << myMap[FILENAME] << "\"" << std::endl;
      else
	std::cerr << "error: Invalid configuration file" << myMap[FILENAME] << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    config.save();
  }
  catch (std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}
