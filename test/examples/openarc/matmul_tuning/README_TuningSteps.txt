[Compilation steps for tuning]
1) Delete output files generated for tuning
	$ tCleanTOutput.bash

2) Create cetus_input directory
- Run the following script
	$ tCreateInitDir.bash

3) Generate tuning configuration files
3-1) Extract applicable tuning parameters
	$ O2GBuild.script 0 #this will invoke the translator 
                        #using "-extractTuningParameters" option,
						#which generates "TuningOptions.txt" file
						#containing applicable options.
3-2) Create default-tuning-configuration-setup file ("gpuTuning.config").
3-3) Generate a set of tuning configuration files.
	$ tConfGen.bash [Tuning Level]
		#Tuning Level = 1 for program-level tuning (default)
		#				2 for GPU-kernel-level tuning

4) Generate output GPU files for each tuning configuration.
- numExperiments in the script should be updated based on the value returned by tConfGen.bash
	$ tBatchTranslation.bash [Tuning Level]
		#Tuning Level = 1 for program-level tuning (default)
		#				2 for GPU-kernel-level tuning

5) Compile output GPU binaries for each tuning configuration
- numExperiments in the script should be updated based on the value returned by tConfGen.bash
	$ tBatchCompile.bash [Tuning Level]
		#Tuning Level = 1 for program-level tuning (default)
		#				2 for GPU-kernel-level tuning

6) Run the compiled output GPU binaries.
- numExperiments in the script should be updated based on the value returned by tConfGen.bash
	$ tBatchRun.bash [Tuning Level]
		#Tuning Level = 1 for program-level tuning (default)
		#				2 for GPU-kernel-level tuning


[Comment]
- If tuning-configuration-setup file ("gpuTuning.config" in Step 3-1) exists or a user wants to use the default configuration set by the compiler, the following script will perform all of the above steps.
	$ tBatchTuning.bash -t=tLevel -m=tMode
		#tLevel = 1 for program-level tuning (default)
		#         2 for GPU-kernel-level tuning
		#
		#tMode = 0 for batch translation and exit
        #tMode = 1 for batch compile and exit
        #tMode = 2 for batch run and exit
        #tMode = 3 for batch translation,compile, and run (default)
