#! /bin/bash

if [ $# -gt 0 ]; then
	mode=$1
else
	mode="util"
fi

make purge
make OMP=1 MODE=normal
make clean
make OMP=1 MODE=profile
make clean
make OMP=0 MODE=normal
make clean
make OMP=0 MODE=profile
make clean
if [ "$mode" = "all" ] || [ "$mode" = "dist" ]; then
	make OMP=1 MODE=normal DIST=1
	make clean
fi
if [ "$mode" = "all" ] || [ "$mode" = "util" ]; then
	if [ "$OPENARC_ARCH" != "5" ]; then
		make binUtil PRINT_LOG=1
		if [ -d ../bin ]; then
			cp -f binBuilder_* ../bin
		fi
		make clean
		make res
	fi
	make omphelp
	gcc -o Timer Timer.c
	make clean
fi
