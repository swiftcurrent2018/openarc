#! /bin/bash

#############################################
# Delete output files generated for tuning. #
#############################################

#######################################
# Modify the following conf variables #
#######################################
inputData=( 512 ) 
numInputs=${#inputData[@]}
startInput=0
benchmark="matmul_tuning"


if [ "$openarc" = "" ] || [ ! -d "$openarc" ]; then
	echo "Environment variable, openarc, should be set up correctly to run this script; exit."
	exit
fi

baseDir="${openarc}/test/examples/openarc"
workDir="${baseDir}/${benchmark}"
outputDirBase="${workDir}/cetus_output"

rm -rf tuning_conf
rm -f TuningOptions.txt
rm -f userDirective*.txt
if [ -d "${outputDirBase}" ]; then
	cd ${outputDirBase}
	rm -f userDirective*.txt
	rm -f confFile.txt

	i=${startInput}
	while [ $i -lt $numInputs ]
	do

		inputClass=${inputData[$i]}
		rm -rf ${inputClass}

	i=$((i+1))
	done
fi


