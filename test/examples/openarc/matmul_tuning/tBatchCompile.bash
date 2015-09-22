#! /bin/bash

#######################################
# Modify the following conf variables #
#######################################
inputData=( 512 )
numInputs=${#inputData[@]}
startInput=0
benchmark="matmul_tuning"
if [ "$OPENARC_ARCH" = "0" ] || [ "$OPENARC_ARCH" = "" ]; then
	inputKernelFile="openarc_kernel.cu"
else
	inputKernelFile="openarc_kernel.cl"
fi
cleanCmd="make clean"
exeCmd="make ACC"
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
logFile="${workDir}/${benchmark}_Compile.log"

echo "## Compile Starts .... " > ${logFile}
date >> ${logFile}
echo " " >> ${logFile}

i=${startInput}
while [ $i -lt $numInputs ]
do
	cd ${workDir}
	inputClass=${inputData[$i]}
	echo "## Input Data : ${inputClass}" >> ${logFile}
	echo " " >> ${logFile}
	###########################################
	# Create output directory if not existing #
	###########################################
	if [ ! -d "${outputDirBase}" ]; then
		mkdir -p "$outputDirBase"
	fi
	cd ${outputDirBase}
	if [ ! -d ${inputClass} ]; then
		mkdir ${inputClass}
	fi
	cd ${inputClass}
	rm -rf *

	k=0
	while [ $k -lt ${numExperiments} ]
	do

		cd ${workDir}
		srcDir="Src_${k}"
		inputDir="${workDir}/cetus_output/${inputClass}/${srcDir}"
		echo " "
		echo "$inputDir"
		echo " "

		#######################################
		# Copy new file to the work directory #
		#######################################
		cp -f ${inputDir}/* "${workDir}/cetus_output/"

		###########################
		# Compile input GPU files #
		###########################
		cd ${workDir}
		${cleanCmd} 2> /dev/null
        ${exeCmd} 2>&1 | grep -i error | tee test.out
        grep -i error test.out
    	if [ $? -eq 0 ]; then
      		echo "${k} th Compile fail : ${srcDir}" >> ${logFile}
            cat test.out >> ${logFile}
        else
      		echo "${k} th Compile success : ${srcDir}" >> ${logFile}
    	fi  

		############################################
		# Delete the temporary configuration files #
		############################################
		rm -f "${workDir}/cetus_output/*.txt"


		##############################################
		# Move output binary to the output directory #
		##############################################
		cd ${outputDirBase}
		outputDir="${inputClass}/${baseExeName}_${k}"
		mkdir -p ${outputDir}
		newExeName="${baseExeName}_${k}"
		mv ${baseExeName} ${newExeName}
		mv ${newExeName} "${outputDir}"
		mv ${inputKernelFile} "${outputDir}"
		if [ -f *.aocx ]; then
			mv *.aocx ${outputDir}
		fi
		if [ -f *.ptx ]; then
			mv *.ptx ${outputDir}
		fi

		##############################################
		# Move output log if targeting Altera OpenCL #
		##############################################
		if [ -d "${workDir}/cetus_output/openarc_kernel" ];then
			cp -f "${workDir}/cetus_output/openarc_kernel/*.rpt" "${outputDir}/"
			cp -f "${workDir}/cetus_output/openarc_kernel/sys_description.txt" "${outputDir}/"
			cp -f "${workDir}/cetus_output/openarc_kernel/sys_description.legend.txt" "${outputDir}/"
		fi

		k=$((k+1))
	done

i=$((i+1))
done

cd ${workDir}
rm -f test.out
