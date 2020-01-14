#! /bin/bash

function usage()
{
	echo "./batchTest.bash"
	echo "List of options:"
	echo -e "\t-h --help"
	echo -e "\tall -test-all"
	echo -e "\t[list of targets to test]"
	echo ""
	echo "List of targets:"
	echo -e "\texamples aspen impacc nvl-c openacc openmp4 resilience"
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
        all | -test-all)
            TEST_ALL=1
            TEST_TARGETS=( "examples" "aspen" "impacc" "nvl-c" "openacc" "openmp4" "resilience" )
            echo "Test All"
            ;;  
        examples | aspen | impacc | nvl-c | openacc | openmp4 | resilience )
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
    TEST_TARGETS=( "examples" )
fi


if [ "$openarc" = "" ] || [ ! -d "$openarc" ]; then
    echo "Environment variable, openarc, should be set up correctly to run this script; exit."
    exit
fi

if [ "$rodinia" = "" ] || [ ! -d "$rodinia" ]; then
    echo "Environment variable, rodinia, should be set up correctly to run this script; exit."
    exit
fi

if [ ! -f "$openarc/lib/cetus.jar" ]; then
	echo "Build OpenARC compilter and runtime!"
	echo ""
	cd $openarc
	build.sh clean
	build.sh bin
	cd openarcrt
	batchmake.bash
	cd $openarc
fi

translog="$openarc/test/bin/batchtranslation.log"
compilelog="$openarc/test/bin/batchcompile.log"
runlog="$openarc/test/bin/batchrun.log"
faillog="$openarc/test/bin/failsummary.log"
templog="$openarc/test/bin/temp.log"

date | tee $translog
date | tee $compilelog
date | tee $runlog
date | tee $faillog

for TARGET in ${TEST_TARGETS[@]}
do
	if [ "$TARGET" = "nvl-c" ]; then
		echo "==> Test benchmarks in the $TARGET directory"
	elif [ "$TARGET" = "aspen" ]; then
		echo "==> Test benchmarks in the $TARGET directory"
	else
		echo "==> Test benchmarks in the $TARGET directory"
		if [ "$TARGET" = "examples" ]; then
			targetDir="$openarc/test/examples/openarc"
		else
			targetDir="$openarc/test/benchmarks/$TARGET"
		fi
		if [ -d "$targetDir" ]; then
			cd "$targetDir"
			i=1
			while [ $i -le 3 ]
			do
				cd "$targetDir"
				#benchmarks=( `find . -mindepth $i -maxdepth $i -type d | grep -v bin | grep -v cetus_output | grep -v cetus_input | grep -v Docs | grep -v Spec | grep -v data` )
				searchCMD="find . -mindepth $i -maxdepth $i -type d"
				benchmarks=( `${searchCMD} | grep -v bin | grep -v cetus_output | grep -v cetus_input | grep -v Docs | grep -v Spec | grep -v data` )
				for example in ${benchmarks[@]}
				do
					if [ -f "$targetDir/$example/Makefile" ] && [ -f "$targetDir/$example/O2GBuild.script" ]; then
						echo $example | grep -e "_tuning" -e "_task" -e "_cash" -e "_manualdeepcopy" > /dev/null
						if [ $? -eq 0 ]; then
							echo "" | tee -a $translog
							echo "====> Skip ${targetDir}/${example}!" | tee -a $translog
							echo "" | tee -a $translog
							continue
						fi
						cd ${targetDir}/${example}	
						echo "" | tee -a $translog
						echo "==> Target: ${targetDir}/${example}" | tee -a $translog
						echo "" | tee -a $translog
						rm -f openarcConf.txt options.cetus
						echo "${example}" | grep "nvl-c" > /dev/null
						if [ $? -eq 0 ]; then
							make clean
						else
							make purge
						fi
						./O2GBuild.script 2>&1 | tee $templog
						cat $templog >> $translog
						cat $templog | grep -e "Undeclared symbol" -e "fatal error" -e ERROR -e Error -e "exit on error" > /dev/null
						if [ $? -eq 0 ]; then
							echo "Translation Failed!" | tee -a $translog
							echo "" | tee -a $faillog
							echo "==> Target: ${targetDir}/${example} : failed during translation!" | tee -a $faillog
							echo "" | tee -a $faillog
							continue
						else
							echo "Translation Successful!" | tee -a $translog
						fi
						echo $example | grep -e altera -e "_aspen" -e "_mcl" -e "_cuda" > /dev/null
						if [ $? -eq 0 ]; then
							echo "" | tee -a $translog
							echo "====> Skip compilation of ${targetDir}/${example}!" | tee -a $translog
							echo "" | tee -a $translog
							continue
						fi
						if [ ${OPENARC_ARCH} -ne 0 ]; then
							echo $example | grep -e "_cuda" > /dev/null
							if [ $? -eq 0 ]; then
								echo "" | tee -a $translog
								echo "====> Skip compilation of ${targetDir}/${example}!" | tee -a $translog
								echo "" | tee -a $translog
								continue
							fi
						fi
						makeCMD=""
						runCMD=""
						foundMakeCMD=0
						foundRunCMD=0
						while read LINE
						do
							if [ $foundMakeCMD -eq 1 ];  then
								makeCMD=`echo $LINE | sed "s/^..//g"` 
								foundMakeCMD=0
								#echo "makeCMD : $makeCMD"
							else 
								echo $LINE | grep "To compile the translated output file" > /dev/null
								if [ $? -eq 0 ]; then
									foundMakeCMD=1
								fi
							fi
							if [ $foundRunCMD -eq 1 ];  then
								runCMD=`echo $LINE | sed "s/^..........//g"` 
								foundRunCMD=0
								#echo "runCMD : $runCMD"
							else 
								echo $LINE | grep "To run the compiled binary" > /dev/null
								if [ $? -eq 0 ]; then
									foundRunCMD=1
								fi
							fi
						done < <(cat $templog)

						echo "" | tee -a $compilelog
						echo "==> Target: ${targetDir}/${example}" | tee -a $compilelog
						echo "" | tee -a $compilelog
						echo "makeCMD: $makeCMD" | tee -a $compilelog
						echo "" | tee -a $compilelog
						if [ "$makeCMD" != "" ]; then
							$makeCMD 2>&1 | tee $templog
							cat $templog | grep -v -e "error.cpp" | grep -i error > /dev/null
							if [ $? -eq 0 ]; then
								echo "Compile Failed!" | tee -a $compilelog
								echo "" | tee -a $faillog
								echo "==> Target: ${targetDir}/${example} : failed during compilation!" | tee -a $faillog
								echo "" | tee -a $faillog
								continue
							else
								echo "Compile Successful!" | tee -a $compilelog
							fi
							echo "" | tee -a $compilelog
							cat $templog >> $compilelog
						else
							echo "Compile Failed!" | tee -a $compilelog
							echo "" | tee -a $faillog
							echo "==> Target: ${targetDir}/${example} : cannot find compile-command!" | tee -a $faillog
							echo "" | tee -a $faillog
							continue
						fi

						if [ ${OPENARC_ARCH} -ne 0 ]; then
							echo $example | grep -e "unified" > /dev/null
							if [ $? -eq 0 ]; then
								echo "" | tee -a $compilelog
								echo "====> Skip execution of ${targetDir}/${example}!" | tee -a $compilelog
								echo "" | tee -a $compilelog
								continue
							fi
						fi
						echo "" | tee -a $runlog
						echo "==> Target: ${targetDir}/${example}" | tee -a $runlog
						echo "" | tee -a $runlog
						echo "runCMD: $runCMD" | tee -a $runlog
						echo "" | tee -a $runlog
						if [ "$runCMD" != "" ]; then
							cd bin
							$runCMD 2>&1 | tee $templog
							cat $templog | grep -e NPB -e NAS > /dev/null
							if [ $? -eq 0 ]; then
								cat $templog | grep -i -e "not found" > /dev/null
							else
								echo $example | grep -e vecadd -e arrayreduction > /dev/null
								if [ $? -eq 0 ]; then
									cat $templog | grep -i -e "not found" > /dev/null
								else
									cat $templog | grep -i -e error -e "not found" > /dev/null
								fi
							fi
							if [ $? -eq 0 ]; then
								echo "Run Failed!" | tee -a $runlog
								echo "" | tee -a $faillog
								echo "==> Target: ${targetDir}/${example} : failed during execution!" | tee -a $faillog
								echo "" | tee -a $faillog
								continue
							else
								echo "Run Successful!" | tee -a $runlog
							fi
							echo "" | tee -a $runlog
							cat $templog >> $runlog
						else
							echo "Run Failed!" | tee -a $runlog
							echo "" | tee -a $faillog
							echo "==> Target: ${targetDir}/${example} : cannot find run-command!" | tee -a $faillog
							echo "" | tee -a $faillog
							continue
						fi
					fi
				done
			i=$((i+1))
			done
		fi
	fi
done
