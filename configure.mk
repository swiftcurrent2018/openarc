include make.header

COMMON_DEPS = configure.mk make.header

BUILD_CFG_DIR  = class/openacc/exec
BUILD_CFG      = $(BUILD_CFG_DIR)/build.cfg

OPENARC_CC_IN  = src/openarc-cc.in
OPENARC_CC_DIR = bin
OPENARC_CC     = $(OPENARC_CC_DIR)/openarc-cc

.PHONY: base
.PHONY: llvm
base: $(BUILD_CFG)
llvm: $(OPENARC_CC)

$(BUILD_CFG): $(COMMON_DEPS)
	mkdir -p $(BUILD_CFG_DIR)
	echo '# WARNING: This is a generated file. Do not edit.' > $@
	echo 'cpp = $(call cmd2abs, $(CPP))' >> $@
	echo 'cxx = $(call cmd2abs, $(CXX))' >> $@
	echo 'llvmTargetTriple = $(LLVM_TARGET_TRIPLE)' >> $@
	echo 'llvmTargetDataLayout = $(LLVM_TARGET_DATA_LAYOUT)' >> $@
	echo 'mpi_includes = $(MPI_INCLUDES)' >> $@
	echo 'mpi_libdir = $(MPI_LIBDIR)' >> $@
	echo 'mpi_exec = $(MPI_EXEC)' >> $@
	echo 'fc = $(call cmd2abs, $(FC))' >> $@
	echo 'spec_cpu2006 = $(SPEC_CPU2006)' >> $@
	echo 'spec_cfg = $(SPEC_CFG)' >> $@
	echo 'pmem_includes = $(PMEM_INCLUDES)' >> $@
	echo 'pmem_libdir = $(PMEM_LIBDIR)' >> $@
	echo 'nvm_testdir = $(NVM_TESTDIR)' >> $@

$(OPENARC_CC): $(OPENARC_CC_IN) $(COMMON_DEPS)
	mkdir -p $(OPENARC_CC_DIR)
	echo '#!$(call cmd2abs, $(PERL))' > $@
	echo '#' >> $@
	echo '# WARNING: This is a generated file. Do not edit.' >> $@
	echo '#' >> $@
	sed \
	  -e 's|@CC@|$(call cmd2abs, $(CC))|g' \
	  -e 's|@CPP@|$(call cmd2abs, $(CPP))|g' \
	  -e 's|@PMEM_INCLUDES@|$(PMEM_INCLUDES)|g' \
	  -e 's|@PMEM_LIBDIR@|$(PMEM_LIBDIR)|g' \
	$< >> $@
	chmod +x $@
