#Made by Daniel Crawford on 4/8/2017(dsc160130@utdallas.edu)
#CS3377.501
EXECFILE=cs3377dirmond
CXX=g++
LDLIBS=-lrudeconfig
LDFLAGS=-L/home/013/d/ds/dsc160130/ext/rudeconfig/lib
CXXFLAGS=-Wall
CPPFLAGS=-I/home/013/d/ds/dsc160130/ext/tclap/include -I/home/013/d/ds/dsc160130/ext/rudeconfig/include
PROJECTNAME=Program5
OBJS=program5.o parsecmd.o parseconf.o signalhandling.o processcontrol.o inotifyfunctions.oOD

.PHONY: clean backup tarball

all: $(EXECFILE)

clean:
	rm -rf ./*~ ./\#* ./*.o

backup: clean
	@mkdir -p ~/backups; chmod 700 ~/backups
	@$(eval CURDIRNAME := $(shell basename `pwd`))
	@$(eval MKBKUPNAME := ~/backups/$(PROJECTNAME)-$(shell date +'%Y.%m.%d-%H:%M:%S').tar.gz)
	@echo
	@echo Writing backup file to: $(MKBKUPNAME)
	@echo
	@-tar zcfv $(MKBKUPNAME) ../$(CURDIRNAME) --exclude="../Program5/tars" --exclude="../Program5/tempdir"
	@echo
	@chmod 600 $(MKBKUPNAME)
	@echo
	@echo Done!

tarball: clean
	@$(eval CURDIRNAME := $(shell basename `pwd`))
	@-tar zcfv ../Program5.tar.gz ../$(CURDIRNAME) --exclude="../Program5/tars/" --exclude="../Program5/tempdir"
	@echo
	@echo Done!

parsecmd.o: parsecmd.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o  $@ $^
program5.o: program5.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $^
parseconf.o: parseconf.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $^

$(EXECFILE): $(OBJS)
	$(CXX) -static $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)
