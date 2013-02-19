CC=gcc
CFLAGS=-fopenmp -I.
DEPS = ompsmooth.h
OBJ = smooth.o ompsmooth.o 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

smooth: $(OBJ)
	gcc -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o smooth
