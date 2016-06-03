#! /bin/bash

function usage()
{
	echo "./batchCleanup.bash"
	echo "List of options:"
	echo -e "\t-h --help"
	echo -e "\tall -clean-all"
	echo -e "\t[list of targets to clean]"
	echo ""
	echo "List of targets:"
	echo -e "\texamples compiler runtime benchmarks"
	echo ""
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;  
        all | -clean-all)
            CLEAN_ALL=1
            CLEAN_TARGETS=( "examples" "compiler" "runtime" "benchmarks" )
            echo "Delete All"
            ;;  
        examples | compiler | runtime | benchmarks )
            if [ ! -n "$CLEAN_ALL" ]; then
                CLEAN_TARGETS=( "${CLEAN_TARGETS[@]}" $PARAM )
            fi  
            ;;  
        *)  
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;  
    esac
    shift
done

if [ ${#CLEAN_TARGETS[@]} -eq 0 ]; then
    CLEAN_TARGETS=( "examples" )
fi


if [ "$openarc" = "" ] || [ ! -d "$openarc" ]; then
    echo "Environment variable, openarc, should be set up correctly to run this script; exit."
    exit
fi

for TARGET in ${CLEAN_TARGETS[@]}
do
	if [ "$TARGET" = "examples" ]; then
		echo "==> Clean up examples in the test directory"
		cd $openarc/test/examples
		examples=( `find . -mindepth 2 -maxdepth 2 -type d` )
		for example in ${examples[@]}
		do
			echo "==> Target: ${example}"
			cd $openarc/test/examples/${example}	
			rm -f openarcConf.txt options.cetus
			echo "${example}" | grep "openarc" > /dev/null
			if [ $? -eq 0 ]; then
				make purge
			else
				make clean
			fi
		done
	fi

	if [ "$TARGET" = "benchmarks" ]; then
		echo "==> Clean up benchmarks in the test directory"
		rm -f "$openarc/*.log" "$openarc/test/bin/*.log"
		if [ -d "$openarc/test/benchmarks" ]; then
			cd $openarc/test/benchmarks
			benchmarks=( `find . -mindepth 3 -maxdepth 3 -type d` )
			for example in ${benchmarks[@]}
			do
				cd $openarc/test/benchmarks/${example}	
				if [ -f Makefile ]; then
					echo "==> Target: ${example}"
					rm -f openarcConf.txt options.cetus
					echo "${example}" | grep "nvl-c" > /dev/null
					if [ $? -eq 0 ]; then
						make clean
					else
						make purge
					fi
				fi
			done
			cd $openarc/test/benchmarks
			benchmarks=( `find . -mindepth 4 -maxdepth 4 -type d` )
			for example in ${benchmarks[@]}
			do
				cd $openarc/test/benchmarks/${example}	
				if [ -f Makefile ]; then
					echo "==> Target: ${example}"
					rm -f openarcConf.txt options.cetus
					echo "${example}" | grep "nvl-c" > /dev/null
					if [ $? -eq 0 ]; then
						make clean
					else
						make purge
					fi
				fi
			done
		fi
	fi


	if [ "$TARGET" = "compiler" ]; then
		echo ""
		echo "==> Clean up OpenARC compiler"
		cd $openarc
		build.sh clean
	fi

	if [ "$TARGET" = "runtime" ]; then
		echo ""
		echo "==> Clean up OpenARC runtime"
		cd $openarc/openarcrt
		make purge
		rm -f binBuilder_* libopenaccrt_* libopenaccrtomp_* libresilience.a
	fi
done
