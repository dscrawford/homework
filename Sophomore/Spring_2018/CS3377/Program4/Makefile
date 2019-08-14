#Made by Daniel Crawford on 3/14/2018(dsc160130@utdallas.edu)
#Section 3377.501
EXEC=program4
CXX=gcc
CC= gcc
CXXFLAGS=-Wall
CPPFLAGS=-Wall
LEX=flex
PARSE=bison
SCAN_FILE=scan
PARSE_FILE=parse
OBJS= $(SCAN_FILE).o $(PARSE_FILE).o

.PHONY: clean tarball

#Create two executables, one for creating the parser, the other for creating the scanner
all: $(EXEC)


clean:
	rm -rf *~ *.o \#* $(EXEC) $(EXEC)_scanner $(SCAN_FILE).c $(PARSE_FILE).tab.h $(PARSE_FILE).tab.c scanner parser
tarball: clean
	@echo "Creating tarball.."
	@-tar -zcvf ../Program4.tar.gz ../Program4
	@chmod 600 ../Program4.tar.gz
	@echo
	@echo "Done!";

$(PARSE_FILE).tab.c $(PARSE_FILE).tab.h: $(PARSE_FILE).y
	$(PARSE) -dtf $<

$(PARSE_FILE).o: $(PARSE_FILE).tab.c
	$(CXX) $(CXXFLAGS) -c $< -o $(PARSE_FILE).o

$(SCAN_FILE).c: $(SCAN_FILE).l $(PARSE_FILE).tab.h
	$(LEX) -o $@ $<

$(SCAN_FILE).o: $(SCAN_FILE).c
	$(CXX) -c $<

$(EXEC).o: $(EXEC).c $(PARSE_FILE).tab.h
	$(CXX) $(CPPFLAGS) -c $<

$(EXEC): $(EXEC).o $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(EXEC).o -o $(EXEC)
	ln -sf ./$(EXEC) parser
	ln -sf ./$(EXEC) scanner
