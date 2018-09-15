PROJECT_NAME=project1_dsc160130
EXEC=$(PROJECT_NAME)
CXX=g++
CPP=g++
CXXFLAGS=-Wall
OBJS=$(PROJECT_NAME).cc
HFILES=readinput.h meminstruct.h cpuinstruct.h

all: $(EXEC)

clean:
	rm -f *~ *.o *\# $(PROJECT_NAME)

$(EXEC): $(OBJS) $(HFILES)
	g++ -Wall -o $@ $^
