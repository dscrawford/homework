#Made by Daniel Crawford(dsc160130) on 2/15/2018
#Section 3377.501

CXX = g++
CXXFLAGS = -Wall
OBJS = program3.o gawktostring.o parseResults.o
EXEC = program3
FOLDER = Program3
PROJECTNAME = dsc160130_Program3
DIR = `pwd`

all: program3

tarball: clean
	@echo
	@-tar --exclude='bin' -zcvf ../Program3.tar.gz ../$(FOLDER) 
	@chmod 600 ../Program3.tar.gz
	@echo
	@echo Done!
clean:
	rm -rf ./*~ ./*.o ./\#* $(EXEC)
backup: clean
	@mkdir -p ~/backups; chmod 700 ~/backups
	@$(eval CURDIRNAME := $(shell basename `pwd`))
	@$(eval MKBKUPNAME := ~/backups/$(PROJECTNAME)-$(shell date + '%Y.%m.%d-%H:$M:$S').tar.gz)
	@echo
	@echo Writing backup file to: $(MKBKUPNAME)
	@-tar zcfv $(MKBKUPNAME) ../$(CURDIRNAME)
	@chmod 600 $(MKBKUPNAME)
	@echo
	@echo Done!

program3.o: program3.cc
	$(CXX) -c $(CXXFLAGS) program3.cc

gawktostring.o: gawktostring.cc
	$(CXX) -c $(CXXFLAGS) gawktostring.cc

parseResults.o: parseResults.cc
	$(CXX) -c $(CXXFLAGS) parseResults.cc

program3: $(OBJS)
	$(CXX) $(OBJS) -o $(EXEC)
