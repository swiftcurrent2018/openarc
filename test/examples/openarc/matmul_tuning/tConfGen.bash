#! /bin/bash

######################
# Generate confFiles #
######################

#######################################
# Modify the following conf variables #
#######################################
inputData=( 512 ) 
numInputs=${#inputData[@]}
startInput=0
benchmark="matmul_tuning"
inputCFiles=( "matmul.c" )
inputHeaderFiles=( )
inputFiles=( "${inputCFiles[@]}" "${inputHeaderFiles[@]}" )
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

gpuConfs="-assumeNonZeroTripLoops -defaultTuningConfFile=gpuTuning.config"

if [ "$openarc" = "" ] || [ ! -d "$openarc" ]; then
	echo "Environment variable, openarc, should be set up correctly to run this script; exit."
	exit
fi

exeCmd="java -classpath $openarc/lib/cetus.jar:$openarc/lib/antlr.jar openacc.exec.ACC2GPUDriver ${gpuConfs} -genTuningConfFiles -tuningLevel=${tLevel} *.c"

baseDir="${openarc}/test/examples/openarc"
workDir="${baseDir}/${benchmark}"
cetus_output="${workDir}/cetus_output"
logFile="${workDir}/${benchmark}_ConfGen.log"

confFileBase="confFile"
userDirfileBase="userDirective"

echo "## Compile Starts .... " > ${logFile}
date >> ${logFile}
echo " " >> ${logFile}

i=${startInput}
while [ $i -lt $numInputs ]
do
	inputClass=${inputData[$i]}
	echo "## Input Data : ${inputClass}" >> ${logFile}
	echo " " >> ${logFile}

	#################################################################
	# delete old source file and copy new one to the work directory #
	#################################################################
	inputDir="${workDir}/cetus_input/${inputClass}/Src"
	for filename in ${inputFiles[@]}
	do
		rm -f "${workDir}/${filename}"
		inputSrc="${inputDir}/${filename}"
		cp -f ${inputSrc} ${workDir}
	done

	######################
	# Run O2G translator #
	######################
	cd ${workDir}
	numExperiments=`${exeCmd} | grep "Number of created tuning-configuration files:" | cut -d' ' -f10`
	echo "Number of created tuning-configuration files for input class ${inputClass}: ${numExperiments}"
	if [ $? -eq 0 ]; then
		echo "${i} th Compile success : input class => ${inputClass}" >> ${logFile}
	else
		echo "${i} th Compile fail : input class => ${inputClass}" >> ${logFile}
	fi

	###########################################
	# Copy generated confFiles to cetus_input #
	###########################################
	inputDir="${workDir}/tuning_conf"
	targetDir="${workDir}/cetus_input/${inputClass}"
	rm -f ${targetDir}/*txt
	cp -f ${inputDir}/* "${targetDir}/"
	rm -f ${inputDir}/*

	cd "${workDir}/cetus_input"
	if [ ! -d ${inputClass} ]; then
		echo "Input directory, ${inputClass}, does not exist!; skip it."
		continue
	fi  
	cd ${inputClass}
	if [ ! -f "${confFileBase}0.txt" ]; then
		echo "Tuning configuration files (confFile*.txt)  do not exist in directory, ${inputClass}; skip it."
		continue
	fi  
	k=0 
	while [ $k -lt ${numExperiments} ]
	do
		srcDir="Src_$k";
		confFile="${confFileBase}${k}.txt"
		echo "outdir=${cetus_output}/${inputClass}/${srcDir}" >> ${confFile}
	k=$((k+1))
	done

i=$((i+1))
done
cd ${workDir}
rmdir tuning_conf


