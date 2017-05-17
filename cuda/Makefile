PROJ = lenet5
CC = g++
NVCC = nvcc

CFLAGS = -c -g -Wall -I/opt/local/include -I$(HOME)/cppunit/include
LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib
LIBS = -lcppunit -ldl
OBJS = lenet.o 

all: $(PROJ)

$(PROJ): $(OBJS)
	$(NVCC) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o : %.cu %.h
	$(NVCC) -c $< -o $@

clean:
	rm -f $(PROJ) $(OBJS) 

