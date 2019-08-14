#Made by Daniel Crawford on 4/19/2018 (dsc160130@utdallas.edu)
#CS3377.501
CXX=g++
EXECFILE=program6
LDFLAGS=-L /scratch/perkins/lib
LDLIBS=-lcdk -lcurses
CXXFLAGS=-Wall
CPPFLAGS=-I /scratch/perkins/include
OBJS=program6.o
.PHONY: backup clean
all: $(EXECFILE)

clean: 
	rm -rf *~ \#* $(EXECFILE) $(OBJS)

backup: clean
	@mkdir -p ~/backups; chmod 700 ~/backups
	@$(eval CURDIRNAME := $(shell basename `pwd`))
	@$(eval MKBKUPNAME := ~/backups/$(PROJECTNAME)-$(shell date +'%Y.%m.%d-%H:%M:%S').tar.gz)
	@echo Writing backup file to: $(MKBKUPNAME)
	@-tar zcfv $(MKBKUPNAME) ../$(CURDIRNAME) --exclude="../Homework-6/.git"
	@chmod 600 $(MKBKUPNAME)
	@echo Done!

$(EXECFILE): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)
