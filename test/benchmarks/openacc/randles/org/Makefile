.SUFFIXES: .exe

SRCS =	dev.poise.nocells.c	\
	dev.shear.cells.c 	\
	dev.shear.nocells.c

EXES = $(SRCS:.c=.exe)

CFLAGS = -Wall -pg

all: $(EXES)

.c.exe:
	gcc $(CFLAGS) $< -o $@ -lm

clean::
	rm -f $(EXES) *.tmp *.o

# eof
