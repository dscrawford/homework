CXX=g++
CXXFLAGS=-Wall
OBJS=main.o bigram.o
EXEC=bigram

all: $(EXEC)

clean:
	rm -rf ./*~ ./*.o ./\#* $(EXEC)

%.o: %.c
	$(CXX) -c -o $@ $< $(CXXFLAGS)

bigram: $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS)
