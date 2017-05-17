PROJ = lenet5
CC = g++

CFLAGS = -c -g -w -I/opt/local/include -I$(HOME)/cppunit/include
LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib
LIBS = -lcppunit -ldl
OBJS = $(patsubst %.c,%.o,$(wildcard *.c))
OMPFLAGS = -fopenmp
SIMDFLAGS = -msse2 -msse3 -mavx -O2 -lm

all: $(PROJ)

$(PROJ): $(OBJS)
	$(CC) $(OMPFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o : %.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $< -o $@ $(SIMDFLAGS)

%.o : %.c %.h
	$(CC) $(OMPFLAGS)  $(CFLAGS) $< -o $@ $(SIMDFLAGS)

clean:
	rm -f $(PROJ) $(OBJS)
