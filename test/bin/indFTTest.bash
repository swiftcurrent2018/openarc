#! /bin/bash

function usage()
{
    echo "./indFTTest.bash"
    echo "List of options:"
    echo -e "\t-h --help"
    echo -e "\t-t=[targetpath] --target=[targetpath]"
    echo -e "\t-c --compile-only"
    echo -e "\t-r --run-only"
	echo -e "\t-fv=variable --ftvar=variable"
	echo -e "\t-i=N --itr=N"
	echo -e "\t-m=mode --mode=mode"
	echo -e "\t-RR=N"
	echo -e "\t-RM=N"
	echo -e "\t-fk=N --ftkind=N"
    echo ""
    echo "List of possible modes:"
    echo -e "\tTCPU LLVM ACC"
    echo ""
}

ITR=3
TARGET="__NONE__"
FTVAR=-1
COMPILE_ONLY=0
RUN_ONLY=0
MODE="TCPU"
RR=-1
FTKIND=5
RMODE=0
while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;  
        -t | --target)
			TARGET=${VALUE}
			;;
        -c | --compile-only)
			COMPILE_ONLY=1
			;;
        -r | --run-only)
			RUN_ONLY=1
			;;
        -fv | --ftvar)
			FTVAR=${VALUE}
			;;
        -fk | --ftkind)
			FTKIND=${VALUE}
			;;
        -i | --itr)
			ITR=${VALUE}
			;;
        -m | --mode)
			MODE=${VALUE}
			;;
        -RR)
			RR=${VALUE}
			;;
        -RM)
			RMODE=${VALUE}
			;;
        *)  
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;  
    esac
    shift
done

if [ "$openarc" = "" ] || [ ! -d "$openarc" ]; then
    echo "Environment variable, openarc, should be set up correctly to run this script; exit."
    exit
fi

if [ ! -f "$openarc/bin/openarc" ]; then
	echo "The OpenARC compile should be compiled; exit"
	exit
fi

if [ ! -d "${TARGET}" ]; then
	echo "The target benchmark directory does not exist; ${TARGET}"
	echo "exit"
	exit
fi

function getNewName {
    local i=0 
    local tnamebase="indFTTest"
    local tname="${tnamebase}.log"
    while [ -f "$tname" ]; do
        tname="${tnamebase}_${i}.log"
    i=$((i+1))
    done
    echo "$tname"
}

outlog="${TARGET}/$(getNewName)"
date > ${outlog}
echo "" >> ${outlog}

echo "Test Target:  ${TARGET}" | tee -a ${outlog}
echo "Compile Only:  ${COMPILE_ONLY}" | tee -a ${outlog}
echo "Run Only:  ${RUN_ONLY}" | tee -a ${outlog}
echo "Test Mode: ${MODE}" | tee -a ${outlog}
echo "FTVAR: ${FTVAR}" | tee -a ${outlog}
echo "Number of iterations per test: ${ITR}" | tee -a ${outlog}
echo "RR: ${RR}" | tee -a ${outlog}
echo "RMODE: ${RMODE}" | tee -a ${outlog}
echo "FTKIND: ${FTKIND}" | tee -a ${outlog}
echo "" | tee -a ${outlog}

bitVectors=( 1 2 4 8 16 32 64 128 256 512 1024 )

if [ -d "$TARGET" ]; then
	cd "$TARGET"
	if [ -f "batchCompile.bash" ] && [ -f "batchRun.bash" ]; then
		echo "run FT Test" | tee -a ${outlog}
		if [ $RR -le 0 ]; then
			./batchCompile.bash -numRR
			numRR=$?
			echo "Number of RRs: $numRR" | tee -a ${outlog}
				n0=0
				while [ $n0 -lt $numRR ]
				do
					tBit=${bitVectors[$n0]}
					if [ "$RUN_ONLY" = "0" ]; then
						echo "" | tee -a ${outlog}
						echo "./batchCompile.bash ${MODE} ${FTVAR} ${tBit} ${RMODE} ${FTKIND}" | tee -a ${outlog}
						echo "" | tee -a ${outlog}
						./batchCompile.bash ${MODE} ${FTVAR} ${tBit} ${RMODE} ${FTKIND}
					fi
					if [ "$COMPILE_ONLY" = "0" ]; then
						echo "" | tee -a ${outlog}
						echo "./batchRun.bash ${MODE} ${FTVAR} ${ITR} ${tBit} ${RMODE} ${FTKIND}" | tee -a ${outlog}
						echo "" | tee -a ${outlog}
						./batchRun.bash ${MODE} ${FTVAR} ${ITR} ${tBit} ${RMODE} ${FTKIND}
					fi
				n0=$((n0+1))
				done
		else
			echo "Target RR: $RR" | tee -a ${outlog}
			if [ "$RUN_ONLY" = "0" ]; then
				echo "" | tee -a ${outlog}
				echo "./batchCompile.bash ${MODE} ${FTVAR} ${RR} ${RMODE} ${FTKIND}" | tee -a ${outlog}
				echo "" | tee -a ${outlog}
				./batchCompile.bash ${MODE} ${FTVAR} ${RR} ${RMODE} ${FTKIND}
			fi
			if [ "$COMPILE_ONLY" = "0" ]; then
				echo "" | tee -a ${outlog}
				echo "./batchRun.bash ${MODE} ${FTVAR} ${ITR} ${RR} ${RMODE} ${FTKIND}" | tee -a ${outlog}
				echo "" | tee -a ${outlog}
				./batchRun.bash ${MODE} ${FTVAR} ${ITR} ${RR} ${RMODE} ${FTKIND}
			fi
		fi
    fi  
fi

date >> ${outlog}


