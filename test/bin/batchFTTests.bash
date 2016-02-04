#! /bin/bash

function usage()
{
    echo "./batchFTTests.bash"
    echo "List of options:"
    echo -e "\t-h --help"
    echo -e "\t-c --compile-only"
    echo -e "\t-r --run-only"
	echo -e "\t-i=N --itr=N"
	echo -e "\t-fv=variable --ftvar=variable"
	echo -e "\t-m=mode --mode=mode"
    echo -e "\t-RR=N"
    echo -e "\t-RM=N"
    echo -e "\t-fk=N --ftkind=N"
	echo -e "\t-s=program --skip=program"
    echo -e "\tall -target-all"
    echo -e "\t[list of target benchmark suites to test]"
    echo ""
    echo "List of target benchmark suites (default: kernels):"
    echo -e "\tkernels rodinia npb lulesh"
    echo "List of possible modes:"
    echo -e "\tTCPU LLVM ACC"
    echo ""
}

COMPILE_ONLY=0
RUN_ONLY=0
ITR=3
FTVAR=-1
FTKIND=5
RR=-1
RMODE=0
TEST_TARGETS=( )
SKIP_LIST=( )
MODE="TCPU"
while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;  
        -c | --compile-only)
            COMPILE_ONLY=1
            ;;  
        -r | --run-only)
            RUN_ONLY=1
            ;;  
        -i | --itr)
			ITR=${VALUE}
			;;
        -fv | --ftvar)
            FTVAR=${VALUE}
            ;;  
        -fk | --ftkind)
            FTKIND=${VALUE}
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
        -s | --skip)
			SKIP_LIST=( "${SKIP_LIST[@]}" ${VALUE} )
			;;
        all | -target-all)
            TEST_ALL=1
            TEST_TARGETS=( "kernels" "rodinia" "npb" "lulesh" )
            echo "Test All"
            ;;  
        kernels | rodinia | npb | lulesh )
            if [ ! -n "$TEST_ALL" ]; then
                TEST_TARGETS=( "${TEST_TARGETS[@]}" $PARAM )
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

if [ ${#TEST_TARGETS[@]} -eq 0 ]; then
	echo ""
	echo "==> Choose at least one target benchmark suite!"
	echo ""
	usage
	exit
fi

if [ "$openarc" = "" ] || [ ! -d "$openarc" ]; then
    echo "Environment variable, openarc, should be set up correctly to run this script; exit."
    exit
fi

if [ ! -f "$openarc/bin/openarc" ]; then
	echo "The OpenARC compile should be compiled; exit"
	exit
fi

function getNewName {
	local i=0
	local tnamebase="FTbatch"
	local tname="${tnamebase}.log"
	while [ -f "$tname" ]; do
		tname="${tnamebase}_${i}.log"
	i=$((i+1))
	done
	echo "$tname"
}

workDir=`pwd`
alloutlog="$workDir/$(getNewName)"
echo "$alloutlog"

date > ${alloutlog}
echo "" >> ${alloutlog}

echo "Test Target benchmarks:  ${TEST_TARGETS[@]}" | tee -a ${alloutlog}
echo "Skip List: ${SKIP_LIST[@]}" | tee -a ${alloutlog}
echo "Test Mode: ${MODE}" | tee -a ${alloutlog}
echo "Compile Only: ${COMPILE_ONLY}" | tee a ${alloutlog}
echo "Run Only: ${RUN_ONLY}" | tee a ${alloutlog}
echo "Number of iterations per test: ${ITR}" | tee -a ${alloutlog}
echo "" | tee -a ${alloutlog}

if [ $COMPILE_ONLY -eq 1 ]; then
	compileflags="-c"
elif [ $RUN_ONLY -eq 1 ]; then
	compileflags="-r"
else
	compileflags=""
fi

bitVectors=( 1 2 4 8 16 32 64 128 256 512 1024 )

for TARGET in ${TEST_TARGETS[@]}
do
    if [ "$TARGET" = "lulesh" ]; then
		example="lulesh_ftinject"
	else
        if [ -d "$openarc/test/benchmarks/openacc/$TARGET" ]; then
            cd "$openarc/test/benchmarks/openacc/$TARGET"
            benchmarks=( `find . -mindepth 1 -maxdepth 1 -type d` )
            for example in ${benchmarks[@]}
            do  
				echo $example | grep "_ftinject" 2>&1 > /dev/null
				if [ $? -eq 0 ]; then
					skipThisProgram=0
					for skipP in ${SKIP_LIST[@]}
					do
						echo $example | grep "$skipP" 2>&1 > /dev/null
						if [ $? -eq 0 ]; then
							skipThisProgram=1
							break
						fi
					done
						
					if [ $skipThisProgram -eq 0 ]; then
                		echo "==> Target benchmark: ${example}" | tee -a ${alloutlog}
                		cd $openarc/test/benchmarks/openacc/$TARGET/${example}  
                		if [ -f "batchCompile.bash" ] && [ -f "batchRun.bash" ]; then
							echo "FT Test runs for $example" | tee -a ${alloutlog}
							$openarc/test/bin/indFTTest.bash -t=$openarc/test/benchmarks/openacc/$TARGET/${example} -m=${MODE} -i=${ITR} -fv=${FTVAR} -fk=${FTKIND} -RR=${RR} -RM=${RMODE} ${compileflags}
                		fi  
					fi
				fi
            done
        fi  
    fi  
done

date >> ${alloutlog}


