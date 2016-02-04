#! /bin/bash

function usage()
{
    echo -e "./batchTuningTests.bash"
    echo -e "List of options:"
    echo -e "\t-h --help"
    echo -e "\t-c --clean"
    echo -e "\t-p --purge"
    echo -e "\t-cr --checkresults"
    echo -e "\t-t=tLevel --tuninglevel=tLevel"
    echo -e "\t-m=tMode --mode=tMode"
    echo -e "\t-n=N --numitr=N"
	echo -e "\t-s=program --skip=program"
    echo -e "\tall -target-all"
    echo -e "\t[list of target benchmark suites to test]"
    echo -e ""
    echo -e "List of target benchmark suites (default: kernels):"
    echo -e "\tkernels rodinia npb lulesh xsbench"
    echo -e ""
    echo -e "Supported Modes:"
    echo -e "tMode = 0 #perform batch translation and exit"
    echo -e "tMode = 1 #perform batch compilation and exit"
    echo -e "tMode = 2 #perform batch run and exit"
    echo -e "tMode = 3 #perform batch translation, compilation, and run (default)"
    echo -e ""
    echo -e "Supported Tuning Level:"
    echo -e "tLevel = 1 #program-level tuning (default)"
    echo -e "tLevel = 2 #kernel-level tuning"
    echo -e ""
}

###############################
# Tuning Level                #
###############################
# 1 = program-level tuning    #
# 2 = GPU-kernel-level tuning #
###############################
tLevel=1

###########################################
# Tuning Mode                             #
###########################################
# 0 = Batch translation only              #
# 1 = Batch compile only                  #
# 2 = Batch run only                      #
# 3 = Batch translation, compile, and run #
###########################################
tMode=3

# Number of execution iterations
numItr=1

TEST_TARGETS=( )
SKIP_LIST=( )
optionlist=""
while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        -c | --clean)
			optionlist="${optionlist} -c"
            ;;
        -p | --purge)
			optionlist="${optionlist} -p"
            ;;
        -t | --tuninglevel)
            tLevel=${VALUE}
			optionlist="${optionlist} -t=${tLevel}"
            ;;
        -cr | --checkresults)
			optionlist="${optionlist} -cr"
            ;;
        -m | --mode)
            tMode=${VALUE}
			optionlist="${optionlist} -m=${tMode}"
            ;;
        -n | --numitr)
            numItr=${VALUE}
			optionlist="${optionlist} -n=${numItr}"
            ;;
        -s | --skip)
			SKIP_LIST=( "${SKIP_LIST[@]}" ${VALUE} )
			;;
        all | -target-all)
            TEST_ALL=1
            TEST_TARGETS=( "kernels" "rodinia" "npb" "lulesh" "xsbench" )
            echo "Test All"
            ;;  
        kernels | rodinia | npb | lulesh | xsbench )
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

if [ $tLevel -ne 1 ] && [ $tLevel -ne 2 ]; then
    echo "==> [ERROR] incorrect value for tuninglevel option: $tLevel; exit"
    echo "==> Usage:"
    usage
    exit 1
fi

if [ $tLevel -lt 0 ] || [ $tLevel -gt 3 ]; then
    echo "==> [ERROR] incorrect value for mode option: $tMode; exit"
    echo "==> Usage:"
    usage
    exit 1
fi

if [ $numItr -le 0 ]; then
    echo "==> [ERROR] incorrect value for numitr option: $numItr; exit"
    echo "==> Usage:"
    usage
    exit 1
fi

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

workDir=`pwd`
alloutlog="$workDir/batchTuningResults.log"
echo "Tuning output: $alloutlog"

date > ${alloutlog}
echo "" >> ${alloutlog}

echo "Test Target benchmarks:  ${TEST_TARGETS[@]}" | tee -a ${alloutlog}
echo "Skip List: ${SKIP_LIST[@]}" | tee -a ${alloutlog}
echo "" | tee -a ${alloutlog}

for TARGET in ${TEST_TARGETS[@]}
do
    if [ "$TARGET" = "lulesh" ]; then
		example="lulesh_tuning"
    elif [ "$TARGET" = "xsbench" ]; then
		example="xsbench_tuning"
	else
        if [ -d "$openarc/test/benchmarks/openacc/$TARGET" ]; then
            cd "$openarc/test/benchmarks/openacc/$TARGET"
            benchmarks=( `find . -mindepth 1 -maxdepth 1 -type d` )
            for example in ${benchmarks[@]}
            do  
				echo $example | grep "_tuning" 2>&1 > /dev/null
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
                		if [ -f "tBatchTuning.bash" ] && [ -f "checkResults.pl" ]; then
							echo "Tuning Test runs for $example" | tee -a ${alloutlog}
							./tBatchTuning.bash ${optionlist}
							cat *_extracted1 >> ${alloutlog}
                		fi  
					fi
				fi
            done
        fi  
    fi  
done

date >> ${alloutlog}


