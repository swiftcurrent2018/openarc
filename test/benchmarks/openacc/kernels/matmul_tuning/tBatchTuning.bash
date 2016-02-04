#! /bin/bash

function usage()
{
    echo "./tBatchTuning.bash"
    echo "List of options:"
    echo -e "\t-h --help"
    echo -e "\t-c --clean"
    echo -e "\t-p --purge"
    echo -e "\t-cr --checkresults"
    echo -e "\t-t=tLevel --tuninglevel=tLevel"
    echo -e "\t-m=tMode --mode=tMode"
    echo -e "\t-n=N --numitr=N"
    echo ""
	echo -e "Supported Modes:"
	echo -e "tMode = 0 #perform batch translation and exit"
	echo -e "tMode = 1 #perform batch compilation and exit"
	echo -e "tMode = 2 #perform batch run and exit"
	echo -e "tMode = 3 #perform batch translation, compilation, and run (default)"
    echo ""
	echo -e "Supported Tuning Level:"
	echo -e "tLevel = 1 #program-level tuning (default)"
	echo -e "tLevel = 2 #kernel-level tuning"
}

if [ $# -eq 0 ]; then
	echo "==> Commandline inputs are missing!"
	echo "==> Usage:"
	usage
	exit 1
fi

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

# Selected experiment numbers
#experiments=(31 29 38 41 6 4 13 16)
experiments=( )

cleanResults=0
checkResults=0
while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;  
        -c | --clean)
            cleanResults=1
            ;;  
        -p | --purge)
            cleanResults=2
            ;;  
        -t | --tuninglevel)
            tLevel=${VALUE}
            ;;  
        -cr | --checkresults)
            checkResults=1
            ;;  
        -m | --mode)
            tMode=${VALUE}
            ;;  
        -n | --numitr)
            numItr=${VALUE}
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


#######################################
# Modify the following conf variables #
#######################################
inputData=( 512 ) 
numInputs=${#inputData[@]}
startInput=0
benchmark="$(basename `pwd`)"
inputCFiles=( "matmul.c" )
inputHeaderFiles=( )
inputFiles=( "${inputCFiles[@]}" "${inputHeaderFiles[@]}" )

# OpenARC options (common to all inputs) to translate the benchmark.
#openarcOptionBase=( "macro=VERIFICATION=1,HOST_MEM_ALIGNMENT=1" )
openarcOptionBase=( "macro=VERIFICATION=1" )
# commandline options (common to all inputs) to run the benchmark.
cmdOptionBase=""

if [ "$openarc" = "" ] || [ ! -d "$openarc" ]; then
	echo "Environment variable, openarc, should be set up correctly to run this script; exit."
	exit
fi

###########################
# Internal conf variables #
###########################
workDir="$(pwd)"
orgSrcDir="${workDir}/src"
cetus_input="cetus_input"
cetus_output="cetus_output"
outputDirBase="${workDir}/bin"
confFileBase="confFile"
userDirFileBase="userDirective"

if [ "$OPENARC_ARCH" = "0" ] || [ "$OPENARC_ARCH" = "" ]; then
	inputKernelFile="openarc_kernel.cu"
else
	inputKernelFile="openarc_kernel.cl"
fi
cleanCmd="make clean"
makeCmd="make ACC"
baseExeName="matmul_ACC"

if [ $cleanResults -gt 0 ]; then

###############################################################
# Step0:  Delete all tuning output files except for log files #
###############################################################
echo " "
echo "==> Step0:  Delete tuning output files and exit."
echo " "
rm -rf tuning_conf
rm -f TuningOptions.txt
rm -f userDirective*.txt
rm -rf bin cetus_input cetus_output
for filename in ${inputFiles[@]}
do
	rm -f $filename
done
if [ $cleanResults -gt 1 ]; then
	rm -f *.log
	rm -f *.log_extracted1
fi
exit

fi

if [ $tMode -eq 0 ] || [ $tMode -eq 3 ]; then

#####################################################
# Step1:  Delete output files generated for tuning. #
#####################################################
echo " "
echo "==> Step1:  Delete output files generated for tuning."
echo " "
rm -rf tuning_conf
rm -f TuningOptions.txt
rm -f userDirective*.txt
if [ -d "${workDir}/${cetus_output}" ]; then
	cd "${workDir}/${cetus_output}"
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

##########################################
# Step2: Create input/output directories #
##########################################
echo " "
echo "==> Step2: Create input/output directories"
echo " "
################################
# Create cetus_input directory #
################################
cd ${workDir}
if [ ! -d ${cetus_input} ]; then 
  mkdir ${cetus_input}
fi  

cd "${workDir}/${cetus_input}"

for inputDir in ${inputData[@]}
do
  if [ ! -d ${inputDir} ]; then 
    mkdir ${inputDir}
  fi
	cd ${inputDir}
	srcDir="Src"
  if [ ! -d ${srcDir} ]; then 
    mkdir ${srcDir}
  fi
  for filename in ${inputFiles[@]}
  do
    cp "${orgSrcDir}/${filename}" "./${srcDir}"
  done

  cd "${workDir}/${cetus_input}"
done  

#################################
# Create cetus output directory #
#################################
cd ${workDir}
if [ ! -d "${cetus_output}" ]; then 
  mkdir "cetus_output"
fi  

########################
# Create bin directory #
########################
cd ${workDir}
if [ ! -d "${outputDirBase}" ]; then 
  mkdir "bin"
fi  

#############################
# Step3: Generate confFiles #
#############################
cd ${workDir}
echo " "
echo "==> Step3: Generate confFiles "
echo " "
gpuConfs="-assumeNonZeroTripLoops -defaultTuningConfFile=gpuTuning.config"
exeCmd="java -classpath $openarc/lib/cetus.jar:$openarc/lib/antlr.jar openacc.exec.ACC2GPUDriver ${gpuConfs} -genTuningConfFiles -tuningLevel=${tLevel} *.c"
logFile="${workDir}/${benchmark}_ConfGen.log"

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
	inputDir="${workDir}/${cetus_input}/${inputClass}/Src"
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
	echo "${numExperiments}" > numExperiments.log
	if [ $? -eq 0 ]; then
		echo "${i} th Compile success : input class => ${inputClass}" >> ${logFile}
	else
		echo "${i} th Compile fail : input class => ${inputClass}" >> ${logFile}
	fi

	###########################################
	# Copy generated confFiles to cetus_input #
	###########################################
	inputDir="${workDir}/tuning_conf"
	targetDir="${workDir}/${cetus_input}/${inputClass}"
	rm -f ${targetDir}/*txt
	cp -f ${inputDir}/* "${targetDir}/"
	rm -f ${inputDir}/*

	cd "${workDir}/${cetus_input}"
	if [ ! -d ${inputClass} ]; then
		echo "Input directory, ${inputClass}, does not exist!; skip it."
		continue
	fi  
	cd ${inputClass}
	if [ ! -f "${confFileBase}0.txt" ]; then
		echo "Tuning configuration files (confFile*.txt)  do not exist in directory, ${inputClass}; skip it."
		continue
	fi  

	# Input-specific OpenARC options to translate the benchmark.
	inputOpenARCOption="macro=_N_=${inputClass}"
	k=0 
	while [ $k -lt ${numExperiments} ]
	do
        if [[ ${experiments[@]} && " ${experiments[*]} " != *" $k "* ]]; then
            k=$((k+1))
            continue
        fi
		srcDir="Src_$k";
		confFile="${confFileBase}${k}.txt"
		echo "outdir=${workDir}/${cetus_output}/${inputClass}/${srcDir}" >> ${confFile}
		for tORCOption in ${openarcOptionBase[@]}
		do
			echo "${tORCOption}" >> ${confFile}
		done
		echo "${inputOpenARCOption}" >> ${confFile}
	k=$((k+1))
	done

i=$((i+1))
done
cd ${workDir}
rmdir tuning_conf

#########################################################
# Step4: Compile set of inputs in cetus_input directory #
#########################################################
cd ${workDir}
echo " "
echo "==> Step4: Compile set of inputs in cetus_input directory"
echo " "
exeCmdBase="java -classpath $openarc/lib/cetus.jar:$openarc/lib/antlr.jar openacc.exec.ACC2GPUDriver -gpuConfFile=confFile.txt"
logFile="${workDir}/${benchmark}_Translation.log"

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
	inputDir="${workDir}/${cetus_input}/${inputClass}/Src"
	for filename in ${inputFiles[@]}
	do
		rm -f "${workDir}/${filename}"
		inputSrc="${inputDir}/${filename}"
		cp -f ${inputSrc} ${workDir}
	done

	k=0
	while [ $k -lt ${numExperiments} ]
	do
        if [[ ${experiments[@]} && " ${experiments[*]} " != *" $k "* ]]; then
            k=$((k+1))
            continue
        fi

		srcDir="Src_${k}"
		inputDir="${workDir}/${cetus_input}/${inputClass}"
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
		cd "${workDir}/${cetus_output}"
		if [ ! -d ${inputClass} ]; then
			mkdir ${inputClass}
		fi
		cd ${inputClass}
		if [ ! -d ${srcDir} ]; then
			mkdir ${srcDir}
		fi
		outputDir="${workDir}/${cetus_output}/${inputClass}/${srcDir}"

		######################
		# Run O2G translator #
		######################
		cd ${workDir}
        exeCmd="${exeCmdBase} *.c"
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

fi

if [ $tMode -eq 1 ] || [ $tMode -eq 3 ]; then

##############################################
# Step5: Compile the translated output files #
##############################################
echo " "
echo "==> Step5: Compile the translated output files"
echo " "
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
	cd ${outputDirBase}
	if [ ! -d ${inputClass} ]; then
		mkdir ${inputClass}
	fi
	cd ${inputClass}
	rm -rf *

	if [ "${numExperiments}" = "" ]; then
		numExperiments=`cat ${workDir}/numExperiments.log`
	fi
	k=0
	while [ $k -lt ${numExperiments} ]
	do
        if [[ ${experiments[@]} && " ${experiments[*]} " != *" $k "* ]]; then
            k=$((k+1))
            continue
        fi

		cd ${workDir}
		srcDir="Src_${k}"
		inputDir="${workDir}/${cetus_output}/${inputClass}/${srcDir}"
		echo " "
		echo "$inputDir"
		echo " "

		#######################################
		# Copy new file to the work directory #
		#######################################
		cp -f ${inputDir}/* "${workDir}/${cetus_output}/"

		###########################
		# Compile input GPU files #
		###########################
		cd ${workDir}
		${cleanCmd} 2> /dev/null
        ${makeCmd} 2>&1 | grep -i error | tee test.out
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
		rm -f "${workDir}/${cetus_output}/*.txt"

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
		tPTXFiles=( `ls *.ptx` )
		for ptxFile in ${tPTXFiles[@]}
		do	
            mv ${ptxFile} ${outputDir}
		done

        ##############################################
        # Move output log if targeting Altera OpenCL #
        ##############################################
        if [ -d "${workDir}/cetus_output/openarc_kernel" ];then
            cp -f "${workDir}/cetus_output/openarc_kernel/area.rpt" "${outputDir}/"
            cp -f "${workDir}/cetus_output/openarc_kernel/sys_description.txt" "${outputDir}/"
            cp -f "${workDir}/cetus_output/openarc_kernel/sys_description.legend.txt" "${outputDir}/"
        fi  

		k=$((k+1))
	done

i=$((i+1))
done

cd ${workDir}
rm -f test.out

fi

if [ $tMode -eq 2 ] || [ $tMode -eq 3 ]; then

##########################################
# Step6: Batch-run the compiled binaries #
##########################################
echo " "
echo "==> Step6: Batch-run the compiled binaries"
echo " "
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
	# Input-specific options to run the benchmark.
	inputOption=""
	# commandline options to run the benchmark.
	cmdOption="${cmdOptionBase} ${inputOption}"

	if [ "${numExperiments}" = "" ]; then
		numExperiments=`cat ${workDir}/numExperiments.log`
	fi
	k=0
	while [ $k -lt ${numExperiments} ]
	do
        if [[ ${experiments[@]} && " ${experiments[*]} " != *" $k "* ]]; then
            k=$((k+1))
            continue
        fi

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
		if [ -f ${exeCmd} ]; then
			n=0
			while [ $n -lt ${numItr} ]
			do	
				./${exeCmd} 2>&1 | tee -a ${logFile}
				n=$((n+1))
			done
		else
			echo "[ERROR] executable does not exist!" | tee -a ${logFile}
		fi
		which Timer >> /dev/null
		if [ $? -eq 0 ]; then
			Timer 3
		fi
		k=$((k+1))
	done

i=$((i+1))
done

fi

cd ${workDir}
rm -f test.out
for filename in ${inputFiles[@]}
do
	rm -f $filename
done
echo " " >> ${logFile}
echo "## Batch Run Ends .... " >> ${logFile}
date >> ${logFile}

if [ $checkResults -gt 0 ]; then

#########################
# Step7:  Check Results #
#########################
echo " "
echo "==> Step7:  Check Results."
echo " "
cd ${workDir}
if [ -f "${benchmark}_Run.log" ]; then
	./checkResults.pl "${benchmark}_Run.log" 
fi

fi 
