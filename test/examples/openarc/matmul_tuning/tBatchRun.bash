#! /bin/bash

#######################################
# Modify the following conf variables #
#######################################
inputData=( 512 )
numInputs=${#inputData[@]}
startInput=0
benchmark="matmul_tuning"
cmdOptionBase=""
baseExeName="matmul_ACC"

###############################
# Tuning Level                #       
###############################
# 1 = program-level tuning    #
# 2 = GPU-kernel-level tuning #
###############################
if [ $# -ge 1 ]; then
	tLevel=$1
else
	tLevel=1
fi
if [ $tLevel -eq 1 ]; then 
  # number of conf files for program-level tuning
  numExperiments=4 
elif [ $tLevel -eq 2 ]; then 
  # number of conf files for kernel-level tuning
  numExperiments=4 
fi


if [ "$openarc" = "" ] || [ ! -d "$openarc" ]; then
	echo "Environment variable, openarc, should be set up correctly to run this script; exit."
	exit
fi

baseDir="$openarc/test/examples/openarc"
workDir="${baseDir}/${benchmark}"
outputDirBase="${workDir}/bin/"
logFile="${workDir}/${benchmark}_Run.log"

echo "## Batch Run Starts .... " > ${logFile}
date >> ${logFile}

i=${startInput}
while [ $i -lt $numInputs ]
do
	cd ${workDir}
	inputClass=${inputData[$i]}
	echo " " >> ${logFile}
	echo "## Input Data : ${inputClass}" >> ${logFile}
	inputOption=""
	cmdOption="${cmdOptionBase} ${inputOption}"

	k=0
	while [ $k -lt ${numExperiments} ]
	do

		cd ${workDir}
		##########################
		# Run the output binary. #
		##########################
		outputDir="${inputClass}/${baseExeName}_${k}"
		cd "${outputDirBase}/${outputDir}"
		newExeName="${baseExeName}_${k}"
		exeCmd="${newExeName} ${cmdOption}"
		echo " " | tee -a ${logFile}
		echo "## Execution Command: ${exeCmd}" | tee -a ${logFile}
		echo " " | tee -a ${logFile}
		./${exeCmd} 2>&1 | tee -a ${logFile}
		k=$((k+1))
	done

i=$((i+1))
done

cd ${workDir}
rm -f test.out
echo " " >> ${logFile}
echo "## Batch Run Ends .... " >> ${logFile}
date >> ${logFile}
