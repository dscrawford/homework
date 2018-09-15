PROJECT_NAME=project1_dsc160130
EXEC=$(PROJECT_NAME)
CXX=g++
CPP=g++
CXXFLAGS=-Wall
OBJS=$(PROJECT_NAME).cc readinput.cc meminstruct.cc cpuinstruct.cc processfunctions.cc

all: $(EXEC)

backup: clean
	@mkdir -p ~/backups; chmod 700 ~/backups
	@$(eval CURDIRNAME := $(shell basename `pwd`))
	@$(eval MKBKUPNAME := ~/backups/$(PROJECT_NAME)-$(shell date +'%Y.%m.%d-%H:%M:%S').tar.gz)
	@echo
	@echo Writing backup file to: $(MKBKUPNAME)
	@echo
	@-tar zcfv $(MKBKUPNAME) ../$(CURDIRNAME)
	@echo
	@chmod 600 $(MKBKUPNAME)
	@echo
	@echo Done!

tarball: clean
	@$(eval CURDIRNAME := $(shell basename `pwd`))
	@-tar zcfv ../$(PROJECT_NAME).tar.gz ../$(CURDIRNAME)
	@echo
	@echo Done!

clean:
	rm -f *~ *.o *\# $(PROJECT_NAME)

$(EXEC): $(OBJS)
	g++ -Wall -o $@ $^
