//Made by Daniel Crawford on 4/7/2018(dsc160130@utdallas.edu)
//CS 3377.501
#include "program5.h"
#include "tclap/CmdLine.h"
#include "tclap/UnlabeledValueArg.h"
#include "tclap/SwitchArg.h"

//Put the arguments of the commandline onto a std::map
std::map<int,std::string> parsecmd(int argc, char* argv[]) {
  
  std::map<int,std::string> commandline;

  //Parse the command line
  try {
    TCLAP::CmdLine cmd("cs377dirmond Directory Monitory Daemon", ' ', "0.1");
    
    TCLAP::SwitchArg daemonArg("d","daemon","Run in daemon mode(forks to run as a daemon)",
			       cmd, false);
    TCLAP::UnlabeledValueArg<std::string> fileArg("config_filename", "The name of the configuration"
						  " file. Defaults to cs3377dirmond.conf", false,
						  "cs3377dirmond.conf", "config filename", false);
    cmd.add(fileArg);
    cmd.parse(argc,argv);

      //Insert the daemon flag as strings true or false
     if (daemonArg.getValue())
       commandline[DAEMON] = "true";
     else
       commandline[DAEMON] = "false";

     //Insert the argument value for the filename
     commandline[FILENAME] = fileArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
  }


  return commandline;
}

