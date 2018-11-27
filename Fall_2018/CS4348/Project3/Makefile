PROJECT_NAME=project3
EXEC=$(PROJECT_NAME)
CXX=g++
CPP=g++
CXXFLAGS=-std=c++11 -Wall -pedantic -g
OBJS=$(PROJECT_NAME).o validation.o Disk.o allocFunctions.o Logic.o
HEADER=Disk.h Logic.h $(PROJECT_NAME).h

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

$(EXEC): $(OBJS) $(HEADER)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS)
