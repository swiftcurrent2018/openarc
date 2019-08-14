[Example configuration procedure to use the OpenACC-to-FPGA translation framework on the Excl Oswald03 system]
1) Download and install OpenARC on Oswald03
	(Assume that $openarc environment variable refers to the OpenARC install directory.)
	- Follow the steps in the $openarc/README.md file
	- Example Bash setup for OpenARC:
		export ACC_DEVICE_TYPE=acc_device_not_host
		export ACC_DEVICE_NUM=0
		export OPENARC_ARCH=3
		export OPENARCRT_UNIFIEDMEM=0
	- In the "make.header" file:
		- Set TARGET_SYSTEM to Oswald
		- In the Oswald section:
			- Set OPENARC_ARCH to 3

2) Set up shell environment variables for Intel OpenCL SDK.
	- Example Bash setup:
		export alteraroot=/opt/altera/17.1/hld
        export INTELFPGAOCLSDKROOT=/opt/altera/17.1/hld
        export LM_LICENSE_FILE=/usr/local/flexlm/licenses/1-ISE6P5_License.dat
        export AOCL_BOARD_PACKAGE_ROOT=${INTELFPGAOCLSDKROOT}/board/nalla_pcie
		export PATH=$PATH:$INTELFPGAOCLSDKROOT/bin
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INTELFPGAOCLSDKROOT/host/linux64/lib:$AOCL_BOARD_PACKAGE_ROOT/linux64/lib
	- License file:  I named them quartus_license_9july2019.dat and put them in /usr/local/flexlm/licenses on oswald03 and pcie. They expire Sept. 7. 

3) Test compile and execution
	- Go to $openarc/test/examples/openarc/altera
	- Update Makefile if necessary
		- AOCL_BOARD should be set to p510t_sch_ax115
		- AOCL_FLAGS should be set to "-v -report"
	- Compile the input OpenACC program
		$ ./O2GBuild.script
	- Synthesize the generated output OpenCL files, which may take several hours
		$ make
	- Run the synthesized FPGA program
		$ cd bin
		$ ./matmul_ACC

	
