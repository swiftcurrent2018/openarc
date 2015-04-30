#! /bin/bash

##################################################
# Compile set of inputs in cetus_input directory #
##################################################

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

exeCmdBase="java -classpath $openarc/lib/cetus.jar:$openarc/lib/antlr.jar openacc.exec.ACC2GPUDriver -gpuConfFile=confFile.txt"


baseDir="$openarc/test/examples/openarc"
workDir="${baseDir}/${benchmark}"
outputDirBase="${workDir}/cetus_output"
logFile="${workDir}/${benchmark}_Translation.log"


confFileBase="confFile"
userDirFileBase="userDirective"

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

	k=0
	while [ $k -lt ${numExperiments} ]
	do

		srcDir="Src_${k}"
		inputDir="${workDir}/cetus_input/${inputClass}"
		#echo "$inputDir"

		####################################################
		# delete old configuration files and copy new ones #
		# to the work directory                            #
		####################################################
		rm -f "${workDir}/confFile.txt"
		inputSrc="${inputDir}/${confFileBase}${k}.txt"
		cp -f ${inputSrc} ${workDir}/confFile.txt
		if [ -f "${inputDir}/${userDirFileBase}${k}.txt" ]; then
			rm -f ${workDir}/${userDirFileBase}*.txt
			inputSrc="${inputDir}/${userDirFileBase}${k}.txt"
			cp -f ${inputSrc} ${workDir}
		fi

		###########################################
		# Create output directory if not existing #
		###########################################
		cd ${outputDirBase}
		if [ ! -d ${inputClass} ]; then
			mkdir ${inputClass}
		fi
		cd ${inputClass}
		if [ ! -d ${srcDir} ]; then
			mkdir ${srcDir}
		fi
		outputDir="${outputDirBase}/${inputClass}/${srcDir}"

		######################
		# Run O2G translator #
		######################
		cd ${workDir}
        exeCmd="${exeCmdBase} -macro=_N_=${inputClass} *.c"
        ${exeCmd} 2>&1 | tee compile.log
        grep -i error compile.log   
        if [ $? -eq 0 ]; then
        	echo "${k} th Compile fail : ${srcDir}" >> ${logFile}
        else
        	echo "${k} th Compile success : ${srcDir}" >> ${logFile}
        fi  


		##################################################
		# Copy other input files to the output directory #
		##################################################
		cp "${workDir}/confFile.txt" ${outputDir}
		if [ -f "${inputDir}/${userDirFileBase}${k}.txt" ]; then
			inputSrc="${inputDir}/${userDirFileBase}${k}.txt"
			cp -f ${inputSrc} ${outputDir}
			#rm -f ${inputSrc}
		fi
		for filename in ${inputHeaderFiles[@]}
		do
			cp "${inputDir}/Src/${filename}" ${outputDir}
		done

		cd ${workDir}

		k=$((k+1))
	done

i=$((i+1))
done

############################
# Clean the work directory #
############################
cd ${workDir}
rm -f compile.log
for filename in ${inputFiles[@]}
do
rm -f "${workDir}/${filename}"
done


