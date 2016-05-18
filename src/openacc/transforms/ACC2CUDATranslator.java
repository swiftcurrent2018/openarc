/**
 * 
 */
package openacc.transforms;

import cetus.hir.*;
import cetus.exec.*;
import cetus.analysis.LoopTools;

import java.lang.Math;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Collection;
import java.util.Stack;
import java.io.FileWriter;
import java.io.BufferedWriter;

import openacc.analysis.ACCAnalysis;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.*;
import openacc.transforms.CUDATranslationTools;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *
 */
public class ACC2CUDATranslator extends ACC2GPUTranslator {
	protected boolean opt_addErrorCheckingCode = false;
	protected boolean opt_shrdSclrCachingOnSM = false;
	
	protected double CUDACompCapability = 1.1;
	////////////////////////////////////////////////////////
	// Default values are for CUDA Compute Capability 1.x //
	////////////////////////////////////////////////////////
	protected int maxBlockDimXY = 512;
	protected int maxBlockDimZ = 64;
	protected int maxBlockSize = 512;
	protected int maxGridDimensionality = 2;
	protected int maxGridDimSize = 65535; //Max. dim. size of Grid = 65535
	protected int maxSMemSize = 16384;
	protected int maxCMemSize = 65536;
	protected long max1DTextureLMBound = 134217728; //Max 1D linear memory texture bound size in bytes
	protected int warpSize = 32; //default warp size. [CAUTION] if future CUDA has different warp size, this should be set by command-line input.
	
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// Device-dependent values; should be checked using cudaGetDeviceProperties() function at runtime. //
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	protected int maxPitchSize = 262144; //Max pitch size in bytes for cudaMallocPitch()
	protected long defaultGMemSize = 1600000000;
	
	///////////////////////////////////////////
	// Misc. values for translation purpose. //
	///////////////////////////////////////////
	//contains param-symbol to pitch-size symbol map
	protected Map<Symbol, Symbol> pitchedSymMap =  new HashMap<Symbol, Symbol>();
	//contains param-symbol to texture-symbol map
	protected Map<Symbol, Symbol> textureSymMap =  new HashMap<Symbol, Symbol>();
	//contains arg-symbol to texture-access offset map
	protected Map<Symbol, Expression> textureOffsetMap =  new HashMap<Symbol, Expression>();
	//contains param-symbol (= constant symbol) to constant symbol map
	protected Map<Symbol, Symbol> constantSymMap =  new HashMap<Symbol, Symbol>();
	protected Map<TranslationUnit, Map<Procedure, Map<String, Procedure>>> tr2DevProcMap = 
		new HashMap<TranslationUnit, Map<Procedure, Map<String, Procedure>>>();
	protected Map<Procedure, Map<String, Procedure>> devProcMap;
	
	protected String mainEntryFunc = null;
	protected TranslationUnit kernelsTranslationUnit = new TranslationUnit("openarc_kernel.cu");
    protected AnnotationDeclaration accHeaderDecl = null;
	/**
	 * @param prog
	 */
	public ACC2CUDATranslator(Program prog) {
		super(prog);
		pass_name = "[ACC2CUDATranslator]";
		targetModel = 0; //0 for CUDA

		//Add kernel translation unit to the program
        program.addTranslationUnit(kernelsTranslationUnit);

		CUDAInitializer();
	}
	
	protected void setCUDAConfParams(double CUDACompCapability) {
		if( CUDACompCapability >= 2.0 ) {
			maxBlockDimXY = 1024;
			maxBlockSize = 1024;
			maxGridDimensionality = 3;
			maxSMemSize = 49152;
		}
	}
	
	protected void CUDAInitializer() {
		Statement acc_init_stmt, acc_shutdown_stmt;
		Statement optPrintStmt;
		Statement confPrintStmt;
		
		String value = Driver.getOptionValue("addErrorCheckingCode");
		if( value != null ) {
			opt_addErrorCheckingCode = true;
			///////////////////////////////////////////////////////////////////////////
			// If this option is on, forceSyncKernelCall option is suppressed, since //
			// the error checking code contains a built-in synchronization call.     //
			///////////////////////////////////////////////////////////////////////////
			if( opt_forceSyncKernelCall ) {
				opt_forceSyncKernelCall = false;
			} else {
				FunctionCall optSKCPrintCall = new FunctionCall(new NameID("printf"));
				optSKCPrintCall.addArgument(new StringLiteral("====> Explicit synchronization is forced.\\n"));
				optPrintStmts.add( new ExpressionStatement(optSKCPrintCall) );
			}
			FunctionCall optCECPrintCall = new FunctionCall(new NameID("printf"));
			optCECPrintCall.addArgument(new StringLiteral("====> CUDA-error-checking code is added.\\n"));
			optPrintStmts.add( new ExpressionStatement(optCECPrintCall) );
		}
		
		value = Driver.getOptionValue("CUDACompCapability");
		if( value != null ) {
			CUDACompCapability = Double.valueOf(value).doubleValue();
			setCUDAConfParams(CUDACompCapability);
		}
		
		value = Driver.getOptionValue("cudaMaxGridDimSize");
		if( value != null ) {
			maxGridDimSize = Integer.valueOf(value).intValue();
		}

		value = Driver.getOptionValue("cudaGlobalMemSize");
		if( value != null ) {
			defaultGMemSize = Long.valueOf(value).longValue();
		}

		value = Driver.getOptionValue("cudaSharedMemSize");
		if( value != null ) {
			maxSMemSize = Integer.valueOf(value).intValue();
		}
		
		value = Driver.getOptionValue("shrdSclrCachingOnSM");
		if( value != null ) {
			opt_shrdSclrCachingOnSM = true;
			FunctionCall shrdSclrCachingOnSMPrintCall = new FunctionCall(new NameID("printf"));
			shrdSclrCachingOnSMPrintCall.addArgument(
					new StringLiteral("====> Cache shared scalar variables onto GPU shared memory.\\n"));
			optPrintStmts.add( new ExpressionStatement(shrdSclrCachingOnSMPrintCall) );
		}

		value = Driver.getOptionValue("shrdSclrCachingOnReg");
		if( value != null ) {
			FunctionCall shrdSclrCachingOnRegPrintCall = new FunctionCall(new NameID("printf"));
			if( opt_shrdSclrCachingOnSM ) {
				shrdSclrCachingOnRegPrintCall.addArgument(
						new StringLiteral("====> Cache shared scalar variables onto GPU registers.\\n"
								+ "      (Because shrdSclrCachingOnSM is on, R/O shared scalar variables\\n"
								+ "       are cached on shared memory, instead of registers.)\\n"));
			} else {
				shrdSclrCachingOnRegPrintCall.addArgument(
						new StringLiteral("====> Cache shared scalar variables onto GPU registers.\\n"));
			}
			optPrintStmts.add( new ExpressionStatement(shrdSclrCachingOnRegPrintCall) );
		}

		value = Driver.getOptionValue("shrdArryElmtCachingOnReg");
		if( value != null ) {
			FunctionCall shrdArryElmtCachingOnRegPrintCall = new FunctionCall(new NameID("printf"));
			shrdArryElmtCachingOnRegPrintCall.addArgument(
					new StringLiteral("====> Cache shared array elements onto GPU registers.\\n"));
			optPrintStmts.add( new ExpressionStatement(shrdArryElmtCachingOnRegPrintCall) );
		}

		value = Driver.getOptionValue("prvtArryCachingOnSM");
		if( value != null ) {
			FunctionCall prvtArryCachingOnSMPrintCall = new FunctionCall(new NameID("printf"));
			prvtArryCachingOnSMPrintCall.addArgument(
					new StringLiteral("====> Cache private array variables onto GPU shared memory.\\n"));
			optPrintStmts.add( new ExpressionStatement(prvtArryCachingOnSMPrintCall) );
		}

		value = Driver.getOptionValue("shrdArryCachingOnTM");
		if( value != null ) {
			FunctionCall shrdArryCachingOnTMPrintCall = new FunctionCall(new NameID("printf"));
			shrdArryCachingOnTMPrintCall.addArgument(
					new StringLiteral("====> Cache 1-dimensional, R/O shared array variables onto GPU texture memory.\\n"));
			optPrintStmts.add( new ExpressionStatement(shrdArryCachingOnTMPrintCall) );
		}

		value = Driver.getOptionValue("shrdSclrCachingOnConst");
		if( value != null ) {
			FunctionCall shrdSclrCachingOnConstPrintCall = new FunctionCall(new NameID("printf"));
			shrdSclrCachingOnConstPrintCall.addArgument(
					new StringLiteral("====> Cache R/O shared scalar variables onto GPU constant memory.\\n"));
			optPrintStmts.add( new ExpressionStatement(shrdSclrCachingOnConstPrintCall) );
		}
		value = Driver.getOptionValue("shrdArryCachingOnConst");
		if( value != null ) {
			FunctionCall shrdArryCachingOnConstPrintCall = new FunctionCall(new NameID("printf"));
			shrdArryCachingOnConstPrintCall.addArgument(
					new StringLiteral("====> Cache R/O shared array variables onto GPU constant memory.\\n"));
			optPrintStmts.add( new ExpressionStatement(shrdArryCachingOnConstPrintCall) );
		}
		
		value = Driver.getOptionValue("SetAccEntryFunction");
		if( (value != null) && !value.equals("1") ) {
			mainEntryFunc = value;
		}
		
		//Find a list of acc_init() function calls and a list of acc_shutdown() function calls.
		List<FunctionCall> fCallList = IRTools.getFunctionCalls(program);
		for( FunctionCall fCall : fCallList ) {
			String fName = fCall.getName().toString();
			if( fName.compareToIgnoreCase("acc_init") == 0 ) {
				acc_init_list.add(fCall);
			} else if( fName.compareToIgnoreCase("acc_shutdown") == 0 ) {
				acc_shutdown_list.add(fCall);
			}
		}
		boolean found_acc_init_call = true;
		boolean found_acc_shutdown_call = true;
		if( acc_shutdown_list.isEmpty() ) {
			found_acc_shutdown_call = false;
		}
		if( acc_init_list.isEmpty() ) {
			found_acc_init_call = false;
		} else {
			//DEBUG: what if multiple acc_init() calls exist?
			//FunctionCall fCall = acc_init_list.get(0);
			for( FunctionCall fCall : acc_init_list ) {
				accInitStmt = fCall.getStatement();
				Traversable tt = fCall.getParent();
				while( (tt != null) && !(tt instanceof Procedure) ) {
					tt = tt.getParent();
				}
				if( tt == null ) {
					Tools.exit("[ERROR in ACC2CUDATranslator.CUDAInitializer()] no procedure containing acc_init() found; exit!");
				} else if( tt instanceof Procedure ) {
					main = (Procedure)tt;
				}
				while( (tt != null) && !(tt instanceof TranslationUnit) ) {
					tt = tt.getParent();
				}
				if( tt == null ) {
					Tools.exit("[ERROR in ACC2CUDATranslator.CUDAInitializer()] no TranslationUnit containing acc_init() found; exit!");
				} else if( tt instanceof TranslationUnit ) {
					main_TrUnt = (TranslationUnit)tt;
					if( !main_List.contains(main) ) {
						main_List.add(main);
						main_TrUnt_List.add(main_TrUnt);
						//[FIXME] it will work only if one accInitStmt per main procedure.
						accInitStmt_List.add(accInitStmt);
					}
					//main_global_table = (SymbolTable)tt;
				}
			}
		}
		
		boolean found_main = false;
		for ( Traversable tt : program.getChildren() )
		{
			TranslationUnit tu = (TranslationUnit)tt;
			String iFileName = tu.getInputFilename();
			int dot = iFileName.lastIndexOf(".h");
			if( dot >= 0 ) {
				continue;
			}
			if( found_acc_init_call && !found_main ) {
				for(TranslationUnit tmain_TrUnt : main_TrUnt_List) {
					if( tmain_TrUnt.getInputFilename().equals(iFileName) ) {
						found_main = true;
						break;
					}
				}
			} else if( !found_main ) {
				/* find main()procedure */
				DFIterator<Procedure> iter =
					new DFIterator<Procedure>(tu, Procedure.class);
				iter.pruneOn(Procedure.class);
				iter.pruneOn(Statement.class);
				iter.pruneOn(Declaration.class);
				while( iter.hasNext() ) {
					Procedure proc = iter.next();
					String name = proc.getName().toString();

					/* f2c code uses MAIN__ */
					if ( ((mainEntryFunc != null) && name.equals(mainEntryFunc)) || 
						 ((mainEntryFunc == null) && (name.equals("main") || name.equals("MAIN__"))) ) {
						main = proc;
						main_List.add(main);
						main_TrUnt = tu;
						main_TrUnt_List.add(main_TrUnt);
						found_main = true;
						//main_global_table = (SymbolTable) main_TrUnt;
						break;
					}
				}
			}
		}
		if( !found_main ) {
			List<ACCAnnotation> dAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, 
					ACCAnnotation.dataRegions, false);
			Procedure cProc = null;
			if( dAnnots != null ) {
				for( ACCAnnotation dAn : dAnnots ) {
					Annotatable at = dAn.getAnnotatable();
					Procedure tProc = IRTools.getParentProcedure(at);
					if( cProc == null ) {
						cProc = tProc;
					} else {
						if( !cProc.getSymbolName().equals(tProc.getSymbolName())) {
							cProc = null;
							break;
						}
					}
				}
				if( cProc != null ) {
					//All data regions are in the same procedure, which will be a main entry.
					main = cProc;
					main_List.add(main);
					found_main = true;
					main_TrUnt = (TranslationUnit)cProc.getParent();
					main_TrUnt_List.add(main_TrUnt);
					//main_global_table = (SymbolTable)main_TrUnt;
				}
			}
		}
		if( main_List.isEmpty() ) {
			Tools.exit("\n[ERROR in ACC2CUDATranslator.CUDAInitializer()] neither acc_init() call nor main() procedure is found, and thus " +
					"the translator can not find the main entry function; exit. " +
					"To specify the main entry function, either put the acc_init() call explicitly in the main function " +
					"or use \"SetAccEntryFunction\" option.\n");
		}

        // Insert macro for kernel file
            /* Insert CUDA-related header files and macros */
        StringBuilder kernelStr = new StringBuilder(2048);
		kernelStr.append("#ifndef __CUDA_KERNELHEADER__ \n");
		kernelStr.append("#define __CUDA_KERNELHEADER__ \n");
        kernelStr.append("/********************************************/\n");
        kernelStr.append("/* Added codes for OpenACC2CUDA translation */\n");
        kernelStr.append("/********************************************/\n");
		////////////////////////////////////////////////////////////
		//[DEBUG] nvcc calls gcc to compile C++ code.             //
		//There is no "restrict" keyword in C++, but __restrict__ //
		//can be used with GCC when compiling C++.                //
		////////////////////////////////////////////////////////////
		kernelStr.append("#ifdef __cplusplus\n");
		kernelStr.append("#define restrict __restrict__\n");
		kernelStr.append("#endif\n");
        kernelStr.append("#define MAX(a,b) (((a) > (b)) ? (a) : (b))\n");
        kernelStr.append("#define MIN(a,b) (((a) < (b)) ? (a) : (b))\n");
        kernelStr.append("#ifndef DBL_MAX\n");
        kernelStr.append("#define DBL_MAX 1.7976931348623158e+308\n");
		kernelStr.append("#endif\n");
        kernelStr.append("#ifndef DBL_MIN\n");
        kernelStr.append("#define DBL_MIN 2.2250738585072014e-308\n");
		kernelStr.append("#endif\n");
        kernelStr.append("#ifndef FLT_MAX\n");
        kernelStr.append("#define FLT_MAX 3.402823466e+38\n");
		kernelStr.append("#endif\n");
        kernelStr.append("#ifndef FLT_MIN\n");
        kernelStr.append("#define FLT_MIN 1.175494351e-38\n");
		kernelStr.append("#endif\n");
		if( enableFaultInjection ) {
			kernelStr.append("#include \"resilience.cu\"\n");
		}
		kernelStr.append("#endif\n");
        kernelStr.append("\n");
        CodeAnnotation accHeaderAnnot = new CodeAnnotation(kernelStr.toString());
        accHeaderDecl = new AnnotationDeclaration(accHeaderAnnot);

        kernelsTranslationUnit.addDeclarationFirst(accHeaderDecl);
        
        for( int i = 0; i < main_List.size(); i++ ) {
        	Procedure tmain = main_List.get(i);
        	String tmainName = tmain.getSymbolName();
        	boolean real_main = false;
        	if( tmainName.equals("main") || tmainName.equals("MAIN__") ) {
        		real_main = true;
        	}
			if( opt_GenDistOpenACC ) {
        		if( real_main ) {
        			tmain.setName("real_main");
        		}
			}
        	TranslationUnit tu = main_TrUnt_List.get(i);
			String iFileName = tu.getOutputFilename();
        	/* 1) Insert OpenACC initialization call at the beginning of the main() */
        	/*     - acc_init( acc_device_nvidia );                                 */
        	/* 2) Insert OpenACC shutdown call at the end of the main()             */
        	/*     - acc_shutdown( acc_device_nvidia );                             */
        	FunctionCall acc_init_call = new FunctionCall(new NameID("acc_init"));
        	acc_init_call.addArgument(new NameID("acc_device_nvidia"));
        	acc_init_stmt = new ExpressionStatement(acc_init_call);
        	FunctionCall acc_shutdown_call = new FunctionCall(new NameID("acc_shutdown"));
        	acc_shutdown_call.addArgument(new NameID("acc_device_nvidia"));
        	acc_shutdown_stmt = new ExpressionStatement(acc_shutdown_call);
        	FunctionCall optPrintCall = new FunctionCall(new NameID("printf"));
        	optPrintCall.addArgument(new StringLiteral( "/**********************/ \\n" + 
        												"/* Used Optimizations */ \\n" + 
        												"/**********************/ \\n"));
        	FunctionCall confPrintCall = new FunctionCall(new NameID("printf"));
        	confPrintCall.addArgument(new StringLiteral("/***********************/ \\n" + 
        												"/* Input Configuration */ \\n" + 
        												"/***********************/ \\n"));
        	confPrintStmt = new ExpressionStatement(confPrintCall);
        	CompoundStatement mainBody = tmain.getBody();
        	ExpressionStatement flushStmt = null;
        	optPrintStmt = new ExpressionStatement(optPrintCall);
        	Declaration last_decl = IRTools.getLastDeclaration(mainBody);

        	StringBuilder istr = new StringBuilder(256);
        	istr.append("\n////////////////////////////////\n");
        	istr.append("// CUDA Device Initialization //\n");
        	istr.append("////////////////////////////////\n");
        	CodeAnnotation devInitAnnot = new CodeAnnotation(istr.toString());
        	AnnotationStatement devInitStmt = new AnnotationStatement(devInitAnnot);
        	if( last_decl == null ) {
        		firstMainStmt = IRTools.getFirstNonDeclarationStatement(mainBody);
        	} else {
        		firstMainStmt = AnalysisTools.getStatementAfter(mainBody, 
        				(Statement)last_decl.getParent());
        	}
        	mainBody.addStatementBefore(firstMainStmt, devInitStmt);
        	if( !found_acc_init_call ) {
        		mainBody.addStatementBefore(firstMainStmt, acc_init_stmt);
        		accInitStmt = (ExpressionStatement)acc_init_stmt;
        		if( accInitStmt_List.size() > i) {
        			accInitStmt_List.set(i, accInitStmt);
        		} else {
        			accInitStmt_List.add(accInitStmt);
        		}
        		acc_init_list.add(acc_init_call);
        	}
        	if( enableCustomProfiling ) {
        		FunctionCall pfcall = new FunctionCall(new NameID("HI_profile_init"));
        		pfcall.addArgument(new StringLiteral("Program"));
        		int dot = iFileName.lastIndexOf(".");
        		String fNameStem = iFileName.substring(0, dot);
        		pfcall.addArgument(new StringLiteral(fNameStem + ".cprof"));
        		mainBody.addStatementBefore(firstMainStmt, new ExpressionStatement(pfcall));
        	}

        	//PrintTools.println("FirstMainStmt: "+firstMainStmt, 0);
        	/*
        	 * Find return statements in the main function, and add CUDA Exit call
        	 * just before each return statement.
        	 */
        	LinkedList<ReturnStatement> return_list = new LinkedList<ReturnStatement>();
        	BreadthFirstIterator riter = new BreadthFirstIterator(mainBody);
        	riter.pruneOn(Expression.class); /* optimization */
        	for (;;)
        	{
        		ReturnStatement stmt = null;

        		try {
        			stmt = (ReturnStatement)riter.next(ReturnStatement.class);
        		} catch (NoSuchElementException e) {
        			break;
        		}

        		return_list.add(stmt);
        	}
        	if( real_main ) {
        		//Insert shutdown statements only to the real main function.
        		for( Statement rstmt : return_list ) {
        			CompoundStatement rParent = (CompoundStatement)rstmt.getParent();
        			if( (rParent == null) || !(rParent instanceof CompoundStatement) ) {
        				Tools.exit("[ERROR in ACC2CUDATranslator.CUDAInitializer()] can't find " +
        						"a parent statment of a return statement, "  + rstmt);
        			}
        			if( opt_PrintConfigurations ) {
        				rParent.addStatementBefore(rstmt, (Statement)confPrintStmt.clone());
        				for(Statement confStmt : confPrintStmts) {
        					rParent.addStatementBefore(rstmt, (Statement)confStmt.clone());
        				}
        				rParent.addStatementBefore(rstmt, (Statement)optPrintStmt.clone());
        				for(Statement optStmt : optPrintStmts) {
        					rParent.addStatementBefore(rstmt, (Statement)optStmt.clone());
        				}
        			}
        			if( !found_acc_shutdown_call ) {
        				flushStmt = (ExpressionStatement)acc_shutdown_stmt.clone();
        				rParent.addStatementBefore(rstmt, flushStmt);
        				FunctionCall shutdown_call = (FunctionCall)flushStmt.getExpression();
        				acc_shutdown_list.add(shutdown_call);
        			}
        			if( enableCustomProfiling ) {
        				FunctionCall pfcall = new FunctionCall(new NameID("HI_profile_shutdown"));
        				pfcall.addArgument(new StringLiteral("Program"));
        				rParent.addStatementBefore(rstmt, new ExpressionStatement(pfcall));
        			}
        		}
        		////////////////////////////////////////////////////////////
        		// If main() does not have any explicit return statement, //
        		// add OpenACC shutdown call at the end of the main().    //
        		////////////////////////////////////////////////////////////
        		if( return_list.size() == 0 ) {
        			if( opt_PrintConfigurations ) {
        				mainBody.addStatement((Statement)confPrintStmt.clone());
        				for(Statement confStmt : confPrintStmts) {
        					mainBody.addStatement((Statement)confStmt.clone());
        				}
        				mainBody.addStatement((Statement)optPrintStmt.clone());
        				for(Statement optStmt : optPrintStmts) {
        					mainBody.addStatement((Statement)optStmt.clone());
        				}
        			}
        			if( !found_acc_shutdown_call ) {
        				flushStmt = (ExpressionStatement)acc_shutdown_stmt.clone();
        				mainBody.addStatement(flushStmt);
        				FunctionCall shutdown_call = (FunctionCall)flushStmt.getExpression();
        				acc_shutdown_list.add(shutdown_call);
        			}
        			if( enableCustomProfiling ) {
        				FunctionCall pfcall = new FunctionCall(new NameID("HI_profile_shutdown"));
        				pfcall.addArgument(new StringLiteral("Program"));
        				mainBody.addStatement(new ExpressionStatement(pfcall));
        			}
        		}
        	}
        }

		for ( Traversable tt : program.getChildren() )
		{
			TranslationUnit tu = (TranslationUnit)tt;

            //If the translation unit does not contain any declaration, skip
            if(tu.getDeclarations().size() == 0)
                continue;

			//DEBUG: tu.getInputFilename() may include path name, but default 
            //TranslationUnit.output_filename does not have path name.
			String iFileName = tu.getOutputFilename();
			int dot = iFileName.lastIndexOf(".h");
			if( dot >= 0 ) {
				continue;
			}
			PrintTools.println(pass_name + "Input file name = " + iFileName, 5);
			boolean containsACCAnnotations = 
				AnalysisTools.containsPragma(tu, ACCAnnotation.class);
			boolean main_TU = false;
			for( TranslationUnit tmain_TrUnt : main_TrUnt_List ) {
				if( (tmain_TrUnt != null) && tmain_TrUnt.getOutputFilename().equals(iFileName) ) {
					main_TU = true;
					found_main = true;
					break;
				}
			}
			if( main_TU ) {
				//[DEBUG] moved to the previous loop.
			}

			/* Insert CUDA-related header files and macros */
			StringBuilder str = new StringBuilder(256);
			str.append("#ifndef __O2G_INCLUDE__ \n");
			str.append("#define __O2G_INCLUDE__ \n");
			str.append("/********************************************/\n");
			str.append("/* Header files for OpenACC2GPU translation */\n");
			str.append("/********************************************/\n");
			str.append("#include <openacc.h>\n");
			str.append("#include <openaccrt.h>\n");
			str.append("#include <math.h>\n");
			str.append("#include <float.h>\n");
			str.append("#include <limits.h>\n");
			if( opt_GenDistOpenACC ) {	
            	str.append("#include <impacc.h>\n");
            	if( main_TU ) str.append("#include <impacc_app.h>\n");
			}
			if( enableCustomProfiling ) {
				str.append("#include <profile.h>\n");
			}
			str.append("#endif \n/* End of __O2G_INCLUDE__ */");
			CodeAnnotation headerAnnot = new CodeAnnotation(str.toString());
			AnnotationDeclaration headerDecl = new AnnotationDeclaration(headerAnnot);
			tu.addDeclarationFirst(headerDecl);
			str = new StringBuilder(2048);
			str.append("\n#ifndef __O2G_HEADER__ \n");
			str.append("#define __O2G_HEADER__ \n");
			str.append("/*******************************************/\n");
			str.append("/* Codes added for OpenACC2GPU translation */\n");
			str.append("/*******************************************/\n");
			str.append("#define MAX(a,b) (((a) > (b)) ? (a) : (b))\n");
			str.append("#define MIN(a,b) (((a) < (b)) ? (a) : (b))\n");
			////////////////////////////////////////////////////////////
			//[DEBUG] nvcc calls gcc to compile C++ code.             //
			//There is no "restrict" keyword in C++, but __restrict__ //
			//can be used with GCC when compiling C++.                //
			////////////////////////////////////////////////////////////
			str.append("#ifdef __cplusplus\n");
			str.append("#define restrict __restrict__\n");
			str.append("#endif\n");
			str.append("\n");
			str.append("/**********************************************************/\n");
			str.append("/* Maximum width of linear memory bound to texture memory */\n");
			str.append("/**********************************************************/\n");
			str.append("/* width in bytes */\n");
			str.append("#define LMAX_WIDTH    " + max1DTextureLMBound + "\n");
			str.append("/**********************************/\n");
			str.append("/* Maximum memory pitch (in bytes)*/\n");
			str.append("/**********************************/\n");
			str.append("#define MAX_PITCH   " + maxPitchSize + "\n");  
			str.append("/****************************************/\n");
			str.append("/* Maximum allowed GPU global memory    */\n");    
			str.append("/* (should be less than actual size ) */\n");
			str.append("/****************************************/\n");
			str.append("#define MAX_GMSIZE  " + defaultGMemSize + "\n");
			str.append("/****************************************/\n");
			str.append("/* Maximum allowed GPU shared memory    */\n");    
			str.append("/****************************************/\n");
			str.append("#define MAX_SMSIZE  " + maxSMemSize + "\n");
			str.append("/********************************************/\n");
			str.append("/* Maximum size of each dimension of a grid */\n");  
			str.append("/********************************************/\n");
			str.append("#define MAX_GDIMENSION  " + maxGridDimSize + "\n");
			str.append("\n");
			if( maxNumGangs > 0 ) {
				str.append("#define MAX_NUMGANGS  " + maxNumGangs + "\n");
			}
			str.append("#define NUM_WORKERS  " + defaultNumWorkers +"\n");
			//tu.setHeader(str.toString());
			headerAnnot = new CodeAnnotation(str.toString());
			headerDecl = new AnnotationDeclaration(headerAnnot);

			Declaration firstDecl = tu.getFirstDeclaration();
			Declaration lastCudaDecl = null;
			tu.addDeclarationBefore(firstDecl, headerDecl);

			/*                                                     */
			/* Insert variables used for GPU-Kernel Initialization */
			/*                                                     */
			List<Specifier> specs = null;
			Declaration totalNumThreads_decl = null;

			VariableDeclarator numThreads_declarator = new VariableDeclarator(new NameID("gpuNumThreads"));
			numThreads_declarator.setInitializer(new Initializer(new NameID("NUM_WORKERS")));
			specs = new LinkedList<Specifier>();
			specs.add(Specifier.STATIC);
			specs.add(Specifier.INT);
			Declaration numThreads_decl = new VariableDeclaration(specs, numThreads_declarator);
			//tu.addDeclarationAfter(annot, numThreads_decl);
			if( containsACCAnnotations ) {
				tu.addDeclarationAfter(headerDecl, numThreads_decl);
			}
			
/*			VariableDeclarator numThreads_declaratorX = new VariableDeclarator(new NameID("gpuNumThreadsX"));
			numThreads_declaratorX.setInitializer(new Initializer(new IntegerLiteral(1)));
			specs = new LinkedList<Specifier>();
			specs.add(Specifier.STATIC);
			specs.add(Specifier.INT);
			Declaration numThreads_declX = new VariableDeclaration(specs, numThreads_declaratorX);
			tu.addDeclarationAfter(numThreads_decl, numThreads_declX);
			
			VariableDeclarator numThreads_declaratorY = new VariableDeclarator(new NameID("gpuNumThreadsY"));
			numThreads_declaratorY.setInitializer(new Initializer(new IntegerLiteral(1)));
			specs = new LinkedList<Specifier>();
			specs.add(Specifier.STATIC);
			specs.add(Specifier.INT);
			Declaration numThreads_declY = new VariableDeclaration(specs, numThreads_declaratorY);
			tu.addDeclarationAfter(numThreads_declX, numThreads_declY);
			
			VariableDeclarator numThreads_declaratorZ = new VariableDeclarator(new NameID("gpuNumThreadsZ"));
			numThreads_declaratorZ.setInitializer(new Initializer(new IntegerLiteral(1)));
			specs = new LinkedList<Specifier>();
			specs.add(Specifier.STATIC);
			specs.add(Specifier.INT);
			Declaration numThreads_declZ = new VariableDeclaration(specs, numThreads_declaratorZ);
			tu.addDeclarationAfter(numThreads_declY, numThreads_declZ);
*/
			VariableDeclarator totalNumThreads_declarator = new VariableDeclarator(new NameID("totalGpuNumThreads"));
			specs = new LinkedList<Specifier>();
			specs.add(Specifier.STATIC);
			specs.add(Specifier.INT);
			totalNumThreads_decl = new VariableDeclaration(specs, 
					totalNumThreads_declarator);
//			tu.addDeclarationAfter(numThreads_declZ, totalNumThreads_decl);
			if( containsACCAnnotations ) {
				tu.addDeclarationAfter(numThreads_decl, totalNumThreads_decl);
			}
			
			VariableDeclarator numBlocks_declarator = new VariableDeclarator(new NameID("gpuNumBlocks"));
			specs = new LinkedList<Specifier>();
			specs.add(Specifier.STATIC);
			specs.add(Specifier.INT);
			Declaration numBlocks_decl = new VariableDeclaration(specs, numBlocks_declarator);
			if( containsACCAnnotations ) {
				tu.addDeclarationAfter(totalNumThreads_decl, numBlocks_decl);
			}

/*			VariableDeclarator numBlocks_declaratorX = new VariableDeclarator(new NameID("gpuNumBlocksX"));
			specs = new LinkedList<Specifier>();
			specs.add(Specifier.STATIC);
			specs.add(Specifier.INT);
			Declaration numBlocks_declX = new VariableDeclaration(specs, numBlocks_declaratorX);
			tu.addDeclarationAfter(numBlocks_decl, numBlocks_declX);

			VariableDeclarator numBlocks_declaratorY = new VariableDeclarator(new NameID("gpuNumBlocksY"));
			specs = new LinkedList<Specifier>();
			specs.add(Specifier.STATIC);
			specs.add(Specifier.INT);
			Declaration numBlocks_declY = new VariableDeclaration(specs, numBlocks_declaratorY);
			tu.addDeclarationAfter(numBlocks_declX, numBlocks_declY);
			
			if( CUDACompCapability >= 2.0 ) {
				VariableDeclarator numBlocks_declaratorZ = new VariableDeclarator(new NameID("gpuNumBlocksZ"));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.STATIC);
				specs.add(Specifier.INT);
				Declaration numBlocks_declZ = new VariableDeclaration(specs, numBlocks_declaratorZ);
				tu.addDeclarationAfter(numBlocks_declY, numBlocks_declZ);
				//tu.addDeclarationAfter(numBlocks_declZ, totalNumThreads_decl);
			} else {
				//tu.addDeclarationAfter(numBlocks_declY, totalNumThreads_decl);
			}
*/
			VariableDeclarator gpuMemSize_declarator = null;
			Declaration gpuMemSize_decl = null;
			VariableDeclarator smemSize_declarator = null;
			Declaration smemSize_decl = null;
			if( main_TU ) {
				gpuMemSize_declarator = new VariableDeclarator(new NameID("gpuGmemSize"));
				gpuMemSize_declarator.setInitializer(new Initializer(new IntegerLiteral(0)));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.UNSIGNED);
				specs.add(Specifier.INT);
				gpuMemSize_decl = new VariableDeclaration(specs, gpuMemSize_declarator);
				if( opt_addSafetyCheckingCode ) {
					if( containsACCAnnotations ) {
						tu.addDeclarationAfter(numBlocks_decl, gpuMemSize_decl);
					} else {
						tu.addDeclarationAfter(headerDecl, gpuMemSize_decl);
					}
				}
				smemSize_declarator = new VariableDeclarator(new NameID("gpuSmemSize"));
				smemSize_declarator.setInitializer(new Initializer(new IntegerLiteral(0)));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.UNSIGNED);
				specs.add(Specifier.INT);
				smemSize_decl = new VariableDeclaration(specs, smemSize_declarator);
				if( opt_addSafetyCheckingCode ) {
					if( containsACCAnnotations ) {
						tu.addDeclarationAfter(gpuMemSize_decl, smemSize_decl);
					} else {
						tu.addDeclarationAfter(headerDecl, smemSize_decl);
					}
				}
			} else {
				gpuMemSize_declarator = new VariableDeclarator(new NameID("gpuGmemSize"));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.EXTERN);
				specs.add(Specifier.UNSIGNED);
				specs.add(Specifier.INT);
				gpuMemSize_decl = new VariableDeclaration(specs, gpuMemSize_declarator);
				if( opt_addSafetyCheckingCode && containsACCAnnotations ) {
					tu.addDeclarationAfter(numBlocks_decl, gpuMemSize_decl);
				}
				smemSize_declarator = new VariableDeclarator(new NameID("gpuSmemSize"));
				specs = new LinkedList<Specifier>();
				specs.add(Specifier.EXTERN);
				specs.add(Specifier.UNSIGNED);
				specs.add(Specifier.INT);
				smemSize_decl = new VariableDeclaration(specs, smemSize_declarator);
				if( opt_addSafetyCheckingCode && containsACCAnnotations ) {
					tu.addDeclarationAfter(gpuMemSize_decl, smemSize_decl);
				}
			}
			
			VariableDeclarator bytes_declarator = new VariableDeclarator(new NameID("gpuBytes"));
			bytes_declarator.setInitializer(new Initializer(new IntegerLiteral(0)));
			specs = new LinkedList<Specifier>();
			specs.add(Specifier.STATIC);
			specs.add(Specifier.UNSIGNED);
			specs.add(Specifier.INT);
			Declaration bytes_decl = new VariableDeclaration(specs, bytes_declarator);
			if( containsACCAnnotations ) {
				if( opt_addSafetyCheckingCode ) {
					tu.addDeclarationAfter(smemSize_decl, bytes_decl);
				} else {
					tu.addDeclarationAfter(numBlocks_decl, bytes_decl);
				}
			}
			str = new StringBuilder(256);
            if( containsACCAnnotations ) {
            	if( !opt_GenDistOpenACC ) {
            		str.append("\n#ifdef _OPENMP\n");
                	str.append("#pragma omp threadprivate(gpuNumThreads, totalGpuNumThreads, gpuNumBlocks, gpuBytes)\n");
            		str.append("#endif\n");
            	}
			}
			str.append("\n#endif \n/* End of __O2G_HEADER__ */\n");
			CodeAnnotation tailAnnot = new CodeAnnotation(str.toString());
			AnnotationDeclaration tailDecl = new AnnotationDeclaration(tailAnnot);
			if( containsACCAnnotations ) {
				tu.addDeclarationAfter(bytes_decl, tailDecl);
			} else {
				if( main_TU ) {
					if( opt_addSafetyCheckingCode ) {
						tu.addDeclarationAfter(gpuMemSize_decl, tailDecl);
					} else {
						tu.addDeclarationAfter(headerDecl, tailDecl);
					}
				} else {
					tu.addDeclarationAfter(headerDecl, tailDecl);
				}
			}
			CommentAnnotation endComment = new CommentAnnotation("endOfCUDADecls");
			endComment.setSkipPrint(true);
			AnnotationDeclaration endCommentDecl = new AnnotationDeclaration(endComment);
			tu.addDeclarationAfter(tailDecl, endCommentDecl);
			lastCudaDecl = endCommentDecl;
			OpenACCHeaderEndMap.put(tu, lastCudaDecl);
		}
		if( !found_main && !found_acc_init_call ) {
		}
		
		if( opt_LoopCollapse ) {
			loopCollapseHandler = new CUDALoopCollapse(program);
		}
	}
	
	/**
	 * Generate CUDA codes for each OpenACC data clause; if deviceptr clause exists, no CUDA code is generated, but internal annotations
	 * are updated as following:
	 *     - add accdeviceptr internal clause that contains symbols for the variables in the deviceptr clause.
	 *     - remove symbols in the accdeviceptr clause from accshared clause.
	 * 
	 */
	protected void handleDataClauses(ACCAnnotation dAnnot, List<Statement> inStmts, List<Statement> outStmts, 
			DataRegionType dRegionType, boolean IRSymbolOnly) {
		Annotatable at = dAnnot.getAnnotatable();
		Procedure pProc = IRTools.getParentProcedure(at);
		Expression loopIndex = null;
		//Find enclosing ForLoop if existing.
		Traversable tt = at.getParent();
		while( (tt != null) && !(tt instanceof Procedure) ) {
			if( tt instanceof ForLoop ) {
				loopIndex = LoopTools.getIndexVariable((ForLoop)tt);
				break;
			} else {
				tt = tt.getParent();
			}
		}
		//If kernelVerification is on, result-compare codes will be inserted after the following
		//statement. Otherwise, this is not used.
		Statement asyncW1Stmt = null;
		if( kernelVerification && (dRegionType == DataRegionType.ComputeRegion) ) {
			ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "refname");
			String refname = null;
			if( iAnnot != null ) {
				refname = iAnnot.get("refname");
			}
			CompoundStatement cStmt = (CompoundStatement)at.getParent();
			FunctionCall printfCall = new FunctionCall(new NameID("printf"));
			if( refname == null ) {
				printfCall.addArgument(new StringLiteral("[DEBUG-INFO] Start Kernel Verification Test " +
						"for the following compute region; \\n" +
						"OpenACC Annotation: " + dAnnot + "\\nEnclosing Procedure: " + pProc.getSymbolName() + "\\n" ));
			} else {
				printfCall.addArgument(new StringLiteral("[DEBUG-INFO] Start Kernel Verification " +
						"Test for the GPU kernel, " + refname + "\\n" ));
			}
			cStmt.addStatementBefore((Statement)at, new ExpressionStatement(printfCall));
			//Check repeat clause in resilience region.
			boolean containsRepeatClause = false;
			ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "resilience");
			if( enableFaultInjection && (tAnnot != null) && tAnnot.containsKey("repeat") ) {
				Expression ftcond = tAnnot.get("ftcond");
				if( (ftcond ==null) || !(ftcond instanceof IntegerLiteral) 
						|| (((IntegerLiteral)ftcond).getValue() != 0) ) {
					containsRepeatClause = true;
				}
			}
			if( !containsRepeatClause ) {
				printfCall = new FunctionCall(new NameID("printf"));
				printfCall.addArgument(new StringLiteral("[DEBUG-INFO] Kernel Verification Test Passed!\\n"));
				cStmt.addStatementAfter((Statement)at, new ExpressionStatement(printfCall));
			}
			FunctionCall asyncW2Call = new FunctionCall(new NameID("HI_waitS2"), new IntegerLiteral(1));
			ExpressionStatement asyncW2Stmt = new ExpressionStatement(asyncW2Call);
			cStmt.addStatementAfter((Statement)at, asyncW2Stmt);
			FunctionCall asyncW1Call = new FunctionCall(new NameID("HI_waitS1"), new IntegerLiteral(1));
			asyncW1Stmt = new ExpressionStatement(asyncW1Call);
			ACCAnnotation wAnnot = new ACCAnnotation("wait", new IntegerLiteral(1));
			asyncW1Stmt.annotate(wAnnot);
			cStmt.addStatementAfter((Statement)at, asyncW1Stmt);
		}
		//DEBUG: This implementation assumes that a CUDA clause exists at most once per an annotatable object.
		Set<Symbol> constantSet = new HashSet<Symbol>();
		Set<Symbol> textureSet = new HashSet<Symbol>();
		Set<Symbol> sharedROSet = new HashSet<Symbol>();
		Set<Symbol> psharedROSet = new HashSet<Symbol>();
		Set<Symbol> ROSymSet = new HashSet<Symbol>();
		Set<Symbol> PROSymSet = new HashSet<Symbol>();
		Set<Symbol> expSharedSymSet = new HashSet<Symbol>();
		ARCAnnotation tCAnnot = at.getAnnotation(ARCAnnotation.class, "constant");
		Set<SubArray> dataSet;
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("constant");
			constantSet.addAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "noconstant");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("noconstant");
			constantSet.removeAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "texture");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("texture");
			textureSet.addAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "notexture");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("notexture");
			textureSet.removeAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "sharedRO");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("sharedRO");
			sharedROSet.addAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "noshared");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("noshared");
			sharedROSet.removeAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "psharedRO");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("psharedRO");
			psharedROSet.addAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		ACCAnnotation tACCAnnot = at.getAnnotation(ACCAnnotation.class, "accreadonly");
		if( tACCAnnot != null ) {
			ROSymSet.addAll((Set<Symbol>)tACCAnnot.get("accreadonly"));
		}
		tACCAnnot = at.getAnnotation(ACCAnnotation.class, "accpreadonly");
		if( tACCAnnot != null ) {
			PROSymSet.addAll((Set<Symbol>)tACCAnnot.get("accpreadonly"));
		}
		tACCAnnot = at.getAnnotation(ACCAnnotation.class, "accexplicitshared");
		if( tACCAnnot != null ) {
			expSharedSymSet.addAll((Set<Symbol>)tACCAnnot.get("accexplicitshared"));
		}
		//Check if condition
		Expression ifCond = null;
		ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "if");
		if( tAnnot != null ) {
			ifCond = (Expression)tAnnot.get("if");
			ifCond = Symbolic.simplify(ifCond);
			if( ifCond instanceof IntegerLiteral ) {
				if( ((IntegerLiteral)ifCond).getValue() != 0 ) {
					ifCond = null; //Compiler knows that this region will be executed; ignore the if-condition.
				} else { //compiler knows that this region will not be outlined as a GPU kernel; skip conversion.
					return;
				}
			}
		}
		
		//Remove OpenMP annotations if existing.
		//[FIXME] if ifCond is not null and OpenMP annotation exists, this region should be copied for the
		//case where ifCond is false.
		if( ifCond == null ) {
			at.removeAnnotations(OmpAnnotation.class);
		} else {
			List tList = at.getAnnotations(OmpAnnotation.class);
			if( (tList != null) && !tList.isEmpty() && (dRegionType != DataRegionType.ComputeRegion) ) {
				Tools.exit("[ERROR in ACC2CUDATranslator.handleDataClauses()] cloning of data region should be implemented; exit!");
			}
		}
		
		//Check async condition
		Expression asyncID = null;
		tAnnot = at.getAnnotation(ACCAnnotation.class, "async");
		if( tAnnot != null ) {
			Object obj = tAnnot.get("async");
			if( obj instanceof String ) { //async ID is not specified by a user; use minimum int value.
				//asyncID = new NameID("INT_MAX");
				asyncID = new NameID("acc_async_noval");
			} else if( obj instanceof Expression ) {
				asyncID = (Expression)obj;
			}
		}
		
		Statement profileRegion = null;
		if( enableCustomProfiling ) {
			ACCAnnotation profAnnot = at.getAnnotation(ACCAnnotation.class, "profile");
			if( profAnnot != null ) {
				Set<String> modeSet = profAnnot.get("mode");
				if( (modeSet != null) && (modeSet.contains(CustomProfilingTransformation.mode_all) || 
						modeSet.contains(CustomProfilingTransformation.mode_memorytransfer)) ) {
					//Find a parent profile region containing track directive 
					//If no parent profile region is found, use the current profile region.
					tt = at.getParent();
					while ((tt != null) && (tt instanceof Annotatable)) {
						Annotatable atObj = (Annotatable)tt;
						tAnnot = atObj.getAnnotation(ACCAnnotation.class, "track");
						if( tAnnot != null ) {
							profileRegion = (Statement)atObj;
							break;
						}
						tt = tt.getParent();
					}
					if( profileRegion == null ) {
						profileRegion = (Statement)at;
					}
				}
			}
		}
		
		for( String key : dAnnot.keySet() ) {
			MallocType mallocT = MallocType.NormalMalloc;
			MemTrType memtrT = MemTrType.NoCopy;
			DataClauseType dataClauseT = DataClauseType.Malloc;
			boolean genCodeForDataClause = false;
			if( key.equals("copy") ) {
				genCodeForDataClause = true;
				memtrT = MemTrType.CopyInOut;
			} else if( key.equals("copyin") ) {
				genCodeForDataClause = true;
				memtrT = MemTrType.CopyIn;
			} else if( key.equals("copyout") ) {
				genCodeForDataClause = true;
				memtrT = MemTrType.CopyOut;
			} else if( key.equals("create") ) {
				genCodeForDataClause = true;
				memtrT = MemTrType.NoCopy;
			} else if( key.equals("present") ) {
				genCodeForDataClause = true;
				memtrT = MemTrType.NoCopy;
				dataClauseT = DataClauseType.CheckOnly;
			} else if( key.equals("pcopy") ) {
				genCodeForDataClause = true;
				memtrT = MemTrType.CopyInOut;
				dataClauseT = DataClauseType.CheckNMalloc;
			} else if( key.equals("pcopyin") ) {
				genCodeForDataClause = true;
				memtrT = MemTrType.CopyIn;
				dataClauseT = DataClauseType.CheckNMalloc;
			} else if( key.equals("pcopyout") ) {
				genCodeForDataClause = true;
				memtrT = MemTrType.CopyOut;
				dataClauseT = DataClauseType.CheckNMalloc;
			} else if( key.equals("pcreate") ) {
				genCodeForDataClause = true;
				memtrT = MemTrType.NoCopy;
				dataClauseT = DataClauseType.CheckNMalloc;
			} else if( key.equals("device_resident") ) {
				genCodeForDataClause = true;
				memtrT = MemTrType.NoCopy;
			} else if( key.equals("deviceptr") ) {
				genCodeForDataClause = false;
				memtrT = MemTrType.NoCopy;
				dataClauseT = DataClauseType.CheckOnly;
				tAnnot = at.getAnnotation(ACCAnnotation.class, "accdeviceptr");
				if( tAnnot == null ) {
					ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
					dataSet = (Set<SubArray>)dAnnot.get(key);
					Set<Symbol> devicePtrSet = new HashSet<Symbol>();
					devicePtrSet.addAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
					iAnnot.put("accdeviceptr", devicePtrSet);
					Set<Symbol> accSharedSet = iAnnot.get("accshared");
					if( accSharedSet != null ) {
						accSharedSet.removeAll(devicePtrSet);
					}
				}
			}
			if( genCodeForDataClause ) {
				Object value = dAnnot.get(key);
				boolean isFirstData = true;
				if( value instanceof Set ) {
					Collection vSet = AnalysisTools.getSortedCollection((Set)value);
					for( Object elm : vSet ) {
						if( elm instanceof SubArray ) {
							//DEBUG: In current implementation, all data clauses contains Set<SubArray> as values, even for data clauses
							//in declare directives. (see ACCParser.parse_acc_declareddataclause())
							SubArray sArray = (SubArray)elm;
							Expression varName = sArray.getArrayName();
							Symbol sym = SymbolTools.getSymbolOf(varName);
							List<Expression> startList = new LinkedList<Expression>();
							List<Expression> lengthList = new LinkedList<Expression>();
							boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, at);
							if( !key.equals("deviceptr") && !key.equals("present") ) {
								if( !foundDimensions ) {
									Tools.exit("[ERROR in ACC2CUDATranslator.handleDataClauses()] Dimension information " +
											"of the following variable is unknown; exit.\n" + 
											"Variable: " + varName + "\n" +
											"ACCAnnotation: " + dAnnot + "\n" +
											"Enclosing Procedure: " + pProc.getSymbolName() + "\n");
								}
							}
							//PrintTools.println("sArray: " + sArray + ", sArray.getArrayDimension(): " + sArray.getArrayDimension() + "\n", 0);
							List<Specifier> typeSpecs = new ArrayList<Specifier>();
							Symbol IRSym = sym;
							if( sym instanceof PseudoSymbol ) {
								IRSym = ((PseudoSymbol)sym).getIRSymbol();
							}
							if( IRSymbolOnly ) {
								sym = IRSym;
								varName = new Identifier(sym);
								typeSpecs.addAll(((VariableDeclaration)sym.getDeclaration()).getSpecifiers());
							} else {
								Symbol tSym = sym;
								while( tSym instanceof AccessSymbol ) {
									tSym = ((AccessSymbol)tSym).getMemberSymbol();
								}
								typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
							}
							int dimension = lengthList.size();
							mallocT = MallocType.NormalMalloc;
							if( constantSet.contains(sym) ) {
								//CUDA does not allow dynamically allocated constant memory, and thus data should have
								//constant size.
								if( dimension == 0 ) {
									mallocT = MallocType.ConstantMalloc;
								} else {
									boolean constantDimension = true;
									for( Expression tDim : lengthList ) {
										if( (tDim == null) || !(tDim instanceof IntegerLiteral) ) {
											constantDimension = false;
											break;
										}
									}
									if( constantDimension ) {
										mallocT = MallocType.ConstantMalloc;
									}
								}
							} else if( textureSet.contains(sym) ) {
								if( dimension == 1 ) {
									if( sym.getTypeSpecifiers().contains(Specifier.DOUBLE) ) {
										PrintTools.println("\n[WARNING] CUDA does not support double textures; " +
											"textureing of the following double variable will be skipped!\n" + 
											"Variable: " + varName + "\n" +
											"ACCAnnotation: " + dAnnot + "\n" +
											"Enclosing Procedure: " + pProc.getSymbolName() + "\n", 0);
									} else {
										mallocT = MallocType.TextureMalloc;
									}
								} else {
									PrintTools.println("\n[WARNING] current implementation supports only 1D texture reference " +
											"bound to linear memory; textureing of the following variable will be skipped!\n" + 
											"Variable: " + varName + "\n" +
											"ACCAnnotation: " + dAnnot + "\n" +
											"Enclosing Procedure: " + pProc.getSymbolName() + "\n", 0);
								}
							} else if( opt_MallocPitch ) {
								//FIXME: for now, MallocPitch is allowed only to 2D array.
								//if( (dimension == 2) || (dimension == 3) ) 
								if( dimension == 2 ) {
									mallocT = MallocType.PitchedMalloc;
								}
							}
							if( key.equals("present") && mallocT.equals(MallocType.ConstantMalloc) ) {
								if( !foundDimensions ) {
									Tools.exit("[ERROR in ACC2CUDATranslator.handleDataClauses()] Dimension information " +
											"of the following variable is needed to be cached on the Constant memory; " +
											"either provide missing dimensions or disable constant array caching; exit.\n" +
											"Variable: " + sArray.getArrayName() + "\n" + 
											"ACCAnnotation: " + dAnnot + "\n" +
											"Enclosing Procedure: " + pProc.getSymbolName() + "\n");
								}
							}
							
							if( (dimension == 0) && 
									(((memtrT == MemTrType.CopyIn) && ((sharedROSet.contains(sym) &&
											(at.containsAnnotation(ACCAnnotation.class, "kernels") || at.containsAnnotation(ACCAnnotation.class, "parallel"))) 
													|| (sharedROSet.contains(sym) && at.containsAnnotation(ACCAnnotation.class, "data"))))
									|| ((memtrT == MemTrType.NoCopy) && (dataClauseT == DataClauseType.CheckOnly) && psharedROSet.contains(sym)) 
									) ) {
								//We don't have to allocate memory/copy data using HI_memcpy() for R/O shared scalar
								//variable that is in both copyin clause and either 1) sharedRO clause of a compute region 
								//or 2) sharedRO clause of a data region.
								//We also don't have to check its presence for R/O shared scalar variables in the present clause 
								//if they are in the psharedRO clause too.
								//[FIXME] If memtrVerification is on, set_status() function should be added here.
								//==> Better solution is not to insert check_read() function for this variable.
/*								if( memtrVerification ) {
									StringLiteral refName = null;
									ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "refname");
									Procedure cProc = IRTools.getParentProcedure(at);
									if( iAnnot == null ) {
										StringBuilder str = new StringBuilder("[ERROR in ACC2CUDATranslator.handleDataClauses()] can not find referenc name " +
												"used for memory transfer verification; please turn off the verification option " +
												"(programVerification != 1).\n" +
												"OpenACC Annotation: " + dAnnot + "\n");
										if( cProc != null ) {
											str.append("Enclosing Procedure: " + cProc.getSymbolName() + "\n");
										}
										Tools.exit(str.toString());
									} else {
										refName = new StringLiteral((String)iAnnot.get("refname"));
									}
									FunctionCall setStatusCall = new FunctionCall(new NameID("HI_set_status"));
									setStatusCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
											varName.clone()));
									setStatusCall.addArgument(new NameID("acc_device_nvidia"));
									setStatusCall.addArgument(new NameID("HI_notstale"));
									setStatusCall.addArgument(new StringLiteral(varName.toString()));
									setStatusCall.addArgument(refName.clone());
									if( loopIndex != null ) {
										checkCall.addArgument(loopIndex.clone());
									} else {
										checkCall.addArgument(new NameID("INT_MIN"));
									}
									CompoundStatement cStmt = (CompoundStatement)at.getParent();
									cStmt.addStatementBefore((Statement)at, new ExpressionStatement(setStatusCall));
								}*/
								continue;
							} else if( (dRegionType == DataRegionType.ComputeRegion) && (memtrT == MemTrType.NoCopy) 
									&& (dataClauseT == DataClauseType.CheckOnly) && (!expSharedSymSet.contains(sym)) ) { 
								//If a present clause in a compute region is not what the user explicitly inserted,
								//it is safe to skip the present table lookup code generation if the program is correct.
								Traversable t = at.getParent();
								boolean foundEnclosedDataRegion = false;
								while( t != null ) {
									if( t instanceof Annotatable ) {
										Annotatable tAn = (Annotatable)t;
										if( tAn.containsAnnotation(ACCAnnotation.class, "data") ) {
											foundEnclosedDataRegion = true;
											break;
										}
									}
									t = t.getParent();
								}
								if( foundEnclosedDataRegion ) {
									continue;
								}
							} else {
								boolean ROSymbol = false;
								if( ROSymSet.contains(IRSym) ) {
									ROSymbol = true;
								}
								genCUDACodesForDataClause(dAnnot, IRSym, varName, startList, lengthList, typeSpecs, ifCond, asyncID, dataClauseT, 
										mallocT, memtrT, dRegionType, inStmts, outStmts, asyncW1Stmt, profileRegion, isFirstData, ROSymbol);
								isFirstData = false;
							}
						} else {
							break;
						}
					}
				}
			}
		}
		StringLiteral refName = null;
		if( memtrVerification && dRegionType == DataRegionType.ExplicitDataRegion ) {
			List<FunctionCall> fCallList = IRTools.getFunctionCalls(program);
			//Get refname to be used for memory-transfer verification.
			ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "refname");
			Procedure cProc = IRTools.getParentProcedure(at);
			if( iAnnot == null ) {
				StringBuilder str = new StringBuilder("[ERROR in ACC2CUDATranslator.handleDataClauses()] can not find referenc name " +
						"used for memory transfer verification; please turn off the verification option " +
						"(programVerification != 1).\n" +
						"OpenACC Annotation: " + dAnnot + "\n");
				if( cProc != null ) {
					str.append("Enclosing Procedure: " + cProc.getSymbolName() + "\n");
				}
				Tools.exit(str.toString());
			} else {
				refName = new StringLiteral((String)iAnnot.get("refname"));
			}
			tAnnot = at.getAnnotation(ACCAnnotation.class, "tempinternal");
			if( tAnnot != null ) {
				List<ACCAnnotation> pragmas = at.getAnnotations(ACCAnnotation.class);
				Map<Symbol, Symbol> g2lSymMap = new HashMap<Symbol, Symbol>();
				//Remove tempinternal annotation.
				at.removeAnnotations(ACCAnnotation.class);
				for( ACCAnnotation nAnnot : pragmas ) {
					if( !nAnnot.containsKey("tempinternal") ) {
						at.annotate(nAnnot);
					}
				}
				CompoundStatement cStmt = (CompoundStatement)at.getParent();
				Set<Symbol> accessedSyms = null;
				Set<Symbol> firstWriteSet = tAnnot.get("gfirstwriteSet");
				//Set<Symbol> firstReadSet = tAnnot.get("firstreadSet");
				//Set<Symbol> mayKilledSet = tAnnot.get("maykilled");
				//Set<Symbol> deadSet = tAnnot.get("dead");
				Set<Symbol> checkSet = new HashSet<Symbol>();
				if( firstWriteSet != null ) {
					checkSet.addAll(firstWriteSet);
					//System.err.println("Found gfirstwriteSet :" + firstWriteSet +"\nCurrent region: " + at +"\n");
				}
/*				if( firstReadSet != null ) {
					checkSet.addAll(firstReadSet);
				}
				if( mayKilledSet != null ) {
					checkSet.addAll(mayKilledSet);
				}
				if( deadSet != null ) {
					checkSet.addAll(deadSet);
				}*/
				if( !checkSet.isEmpty() ) {
					//Find local symbol visible in the current procedure scope.
					ACCAnnotation sharedAnnot = at.getAnnotation(ACCAnnotation.class, "accshared");
					if( sharedAnnot != null ) {
						accessedSyms = sharedAnnot.get("accshared");
					}
					if( accessedSyms != null ) {
						for( Symbol lSym : accessedSyms ) {
							List symbolInfo = new ArrayList(2);
							if( AnalysisTools.SymbolStatus.OrgSymbolFound(
									AnalysisTools.findOrgSymbol(lSym, at, true, null, symbolInfo, fCallList)) ) {
								Symbol gSym = (Symbol)symbolInfo.get(0);
								if( checkSet.contains(gSym) ) {
									g2lSymMap.put(gSym, lSym);
								}
							}
						}
					}
				}
				if( firstWriteSet != null ) {
					for( Symbol gsym : firstWriteSet ) {
						FunctionCall checkCall = new FunctionCall(new NameID("HI_check_write"));
						Expression hostVar = null;
						Symbol lsym = g2lSymMap.get(gsym);
						if( lsym == null ) {
							Tools.exit("[ERROR in ACC2CUDATranslator.handleDataClauses()] can't find locally visible symbol " +
									"for the first-write symbol: " + gsym + "\nEnclosing procedure: " + 
									cProc.getSymbolName() + "\n");
						}
						if( lsym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
						} else {
							hostVar = new Identifier(lsym);
						}
						if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
							checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
									hostVar.clone()));
						} else {
							checkCall.addArgument(hostVar.clone());
						}
						checkCall.addArgument(new NameID("acc_device_nvidia"));
						checkCall.addArgument(new StringLiteral(hostVar.toString()));
						checkCall.addArgument(refName.clone());
						if( loopIndex != null ) {
							checkCall.addArgument(loopIndex.clone());
						} else {
							checkCall.addArgument(new NameID("INT_MIN"));
						}
						cStmt.addStatementBefore((Statement)at, new ExpressionStatement(checkCall));
					}
				}
/*				if( firstReadSet != null ) {
					for( Symbol gsym : firstReadSet ) {
						FunctionCall checkCall = new FunctionCall(new NameID("HI_check_read"));
						Expression hostVar = null;
						Symbol lsym = g2lSymMap.get(gsym);
						if( lsym == null ) {
							Tools.exit("[ERROR in ACC2CUDATranslator.handleDataClauses()] can't find locally visible symbol " +
									"for the first-read symbol: " + gsym + "\nEnclosing procedure: " + 
									cProc.getSymbolName() + "\n");
						}
						if( lsym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
						} else {
							hostVar = new Identifier(lsym);
						}
						if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
							checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
									hostVar.clone()));
						} else {
							checkCall.addArgument(hostVar.clone());
						}
						checkCall.addArgument(new NameID("acc_device_nvidia"));
						checkCall.addArgument(new StringLiteral(hostVar.toString()));
						checkCall.addArgument(refName.clone());
						if( loopIndex != null ) {
							checkCall.addArgument(loopIndex.clone());
						} else {
							checkCall.addArgument(new NameID("INT_MIN"));
						}
						cStmt.addStatementBefore((Statement)at, new ExpressionStatement(checkCall));
					}
				}
				if( mayKilledSet != null ) {
					for( Symbol gsym : mayKilledSet ) {
						FunctionCall checkCall = new FunctionCall(new NameID("HI_reset_status"));
						Expression hostVar = null;
						Symbol lsym = g2lSymMap.get(gsym);
						if( lsym == null ) {
							Tools.exit("[ERROR in ACC2CUDATranslator.handleDataClauses()] can't find locally visible symbol " +
									"for the may-killed symbol: " + gsym + "\nEnclosing procedure: " + 
									cProc.getSymbolName() + "\n");
						}
						if( lsym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
						} else {
							hostVar = new Identifier(lsym);
						}
						if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
							checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
									hostVar.clone()));
						} else {
							checkCall.addArgument(hostVar.clone());
						}
						checkCall.addArgument(new NameID("acc_device_host"));
						checkCall.addArgument(new NameID("HI_maystale"));
						//checkCall.addArgument(new NameID("INT_MIN"));
						checkCall.addArgument(new NameID("DEFAULT_QUEUE"));
						cStmt.addStatementAfter((Statement)at, new ExpressionStatement(checkCall));
					}
				}
				if( deadSet != null ) {
					for( Symbol gsym : deadSet ) {
						FunctionCall checkCall = new FunctionCall(new NameID("HI_reset_status"));
						Expression hostVar = null;
						Symbol lsym = g2lSymMap.get(gsym);
						if( lsym == null ) {
							System.err.println("g2lSymMap" + g2lSymMap);
							Tools.exit("[ERROR in ACC2CUDATranslator.handleDataClauses()] can't find locally visible symbol " +
									"for the dead symbol: " + gsym + "\nEnclosing procedure: " + 
									cProc.getSymbolName() + "\n");
						}
						if( lsym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
						} else {
							hostVar = new Identifier(lsym);
						}
						if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
							checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
									hostVar.clone()));
						} else {
							checkCall.addArgument(hostVar.clone());
						}
						checkCall.addArgument(new NameID("acc_device_host"));
						checkCall.addArgument(new NameID("HI_notstale"));
						//checkCall.addArgument(new NameID("INT_MIN"));
						checkCall.addArgument(new NameID("DEFAULT_QUEUE"));
						cStmt.addStatementAfter((Statement)at, new ExpressionStatement(checkCall));
					}
				}*/
			}
		}
	}
	
	protected void handleUseDevicesClauses(ACCAnnotation uAnnot, boolean IRSymbolOnly) {
		Annotatable at = uAnnot.getAnnotatable();
		SymbolTable targetSymbolTable = null;
		Traversable tt = at;
		Procedure parentProc = null;
		TranslationUnit parentTrUnt = null;
		while( (tt != null) && !(tt instanceof Procedure) && !(tt instanceof TranslationUnit) ) {
			tt = tt.getParent();
		}
		if( tt instanceof Procedure ) {
			parentProc = (Procedure)tt;
		} else if( tt instanceof TranslationUnit ) {
			parentTrUnt = (TranslationUnit)tt;
		}
		if( parentTrUnt == null ) {
			parentTrUnt = (TranslationUnit)parentProc.getParent();
		}
		for( String key: uAnnot.keySet() ) {
				Object value = uAnnot.get(key);
				if( (key.equals("use_device")) && (value instanceof Set) ) {
					Collection vSet = AnalysisTools.getSortedCollection((Set)value);
					for( Object elm : vSet ) {
						if( elm instanceof SubArray ) {
							SubArray sArray = (SubArray)elm;
							Expression hostVar = sArray.getArrayName();
							Symbol sym = SymbolTools.getSymbolOf(hostVar);
							Boolean isArray = SymbolTools.isArray(sym);
							Boolean isPointer = SymbolTools.isPointer(sym);
							if( sym instanceof NestedDeclarator ) {
								isPointer = true;
							}
							Boolean isScalar = !isArray && !isPointer;
/*							List<Expression> startList = new LinkedList<Expression>();
							List<Expression> lengthList = new LinkedList<Expression>();
							boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly);
							int dimension = lengthList.size();*/
							//PrintTools.println("sArray: " + sArray + ", sArray.getArrayDimension(): " + sArray.getArrayDimension() + "\n", 0);
							List<Specifier> typeSpecs = new ArrayList<Specifier>();
							Symbol IRSym = sym;
							if( sym instanceof PseudoSymbol ) {
								IRSym = ((PseudoSymbol)sym).getIRSymbol();
							}
							if( IRSymbolOnly ) {
								sym = IRSym;
								hostVar = new Identifier(sym);
								typeSpecs.addAll(((VariableDeclaration)sym.getDeclaration()).getSpecifiers());
							} else {
								Symbol tSym = sym;
								while( tSym instanceof AccessSymbol ) {
									tSym = ((AccessSymbol)tSym).getMemberSymbol();
								}
								typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
							}
							targetSymbolTable = AnalysisTools.getIRSymbolScope(IRSym, at.getParent());
							if( targetSymbolTable == null ) {
								Tools.exit("[ERROR in ACC2CUDATranslator.handleUseDevices()] a symbol(" +
										IRSym + ") in a host_data directive is not visible; exit!");
							} else if( targetSymbolTable instanceof Procedure ) {
								targetSymbolTable = ((Procedure)targetSymbolTable).getBody();
							}
							if( targetSymbolTable instanceof CompoundStatement ) {
								if( AnalysisTools.ipFindFirstPragmaInParent(at, OmpAnnotation.class, new HashSet(Arrays.asList("parallel", "task")), false, null, null) != null ) { 
									targetSymbolTable = (CompoundStatement)at.getParent();
								}
							}
							List<Specifier> clonedspecs = new ChainedList<Specifier>();
							clonedspecs.addAll(typeSpecs);
							clonedspecs.remove(Specifier.STATIC);
							///////////////////////////////////////////
							// GPU variables should not be constant. //
							///////////////////////////////////////////
							clonedspecs.remove(Specifier.CONST);
							//////////////////////////////
							// Remove extern specifier. //
							//////////////////////////////
							clonedspecs.remove(Specifier.EXTERN);
							
							///////////////////////////////////////////////////////////////////////////////
							// Create a GPU device variable corresponding to shared_var if not existing. //
							// Ex: float * gpu__b;                                                       //
							///////////////////////////////////////////////////////////////////////////////
							// Give a new name for the device variable 
							Identifier gpuVar = null;
							StringBuilder str = new StringBuilder(80);
							str.append("gpu__");
							if( hostVar instanceof AccessExpression ) {
								str.append(TransformTools.buildAccessExpressionName((AccessExpression)hostVar));
							} else {
								str.append(hostVar.toString());
							}
							Set<Symbol> symSet = targetSymbolTable.getSymbols();
							Symbol gpu_sym = AnalysisTools.findsSymbol(symSet, str.toString());
							boolean addNewGpuSymbol = false;
							if( gpu_sym != null ) {
								gpuVar = new Identifier(gpu_sym);
							} else {
								// Create a new GPU device variable.
								// The type of the device symbol should be a pointer type 
								VariableDeclarator gpu_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
										new NameID(str.toString()));
								VariableDeclaration gpu_decl = new VariableDeclaration(clonedspecs, 
										gpu_declarator);
								gpuVar = new Identifier(gpu_declarator);
								StringBuilder str2 = new StringBuilder(256);
								str2.append("#ifdef _OPENMP\n");
								str2.append("#pragma omp threadprivate(");
								str2.append(str.toString());
								str2.append(")\n");
								str2.append("#endif");
								CodeAnnotation tOmpAnnot = new CodeAnnotation(str2.toString());
								AnnotationDeclaration tAnnotDecl = new AnnotationDeclaration(tOmpAnnot);
								//AnnotationStatement tAnnotStmt = new AnnotationStatement(tOmpAnnot);
								if( targetSymbolTable instanceof TranslationUnit ) {
									symSet = main_TrUnt.getSymbols();
									gpu_sym = AnalysisTools.findsSymbol(symSet, str.toString());
									if( gpu_sym == null ) {
										addNewGpuSymbol = true;
										//FIXME: how to handle this case?
										Declaration tLastCudaDecl = OpenACCHeaderEndMap.get(main_TrUnt);
										main_TrUnt.addDeclarationAfter(tLastCudaDecl, gpu_decl);
										OpenACCHeaderEndMap.put(main_TrUnt, gpu_decl);
										if( parentTrUnt != main_TrUnt ) {
											gpu_declarator = gpu_declarator.clone();
											List<Specifier> extended_clonedspecs = new ChainedList<Specifier>();
											if( !clonedspecs.contains(Specifier.EXTERN) ) {
												extended_clonedspecs.add(Specifier.EXTERN);
											}
											extended_clonedspecs.addAll(clonedspecs);
											gpu_decl = new VariableDeclaration(extended_clonedspecs, 
													gpu_declarator);
											tLastCudaDecl = OpenACCHeaderEndMap.get(parentTrUnt);
											parentTrUnt.addDeclarationAfter(tLastCudaDecl, tAnnotDecl);
											parentTrUnt.addDeclarationAfter(tLastCudaDecl, gpu_decl);
											OpenACCHeaderEndMap.put(parentTrUnt, gpu_decl);
											gpuVar = new Identifier(gpu_declarator);
										}
									} else { //gpuVar exists in the main translation unit, but not in the current translation unit.
										gpu_declarator = gpu_declarator.clone();
										List<Specifier> extended_clonedspecs = new ChainedList<Specifier>();
										if( !clonedspecs.contains(Specifier.EXTERN) ) {
											extended_clonedspecs.add(Specifier.EXTERN);
										}
										extended_clonedspecs.addAll(clonedspecs);
										gpu_decl = new VariableDeclaration(extended_clonedspecs, 
												gpu_declarator);
										Declaration tLastCudaDecl = OpenACCHeaderEndMap.get(parentTrUnt);
										parentTrUnt.addDeclarationAfter(tLastCudaDecl, tAnnotDecl);
										parentTrUnt.addDeclarationAfter(tLastCudaDecl, gpu_decl);
										OpenACCHeaderEndMap.put(parentTrUnt, gpu_decl);
										gpuVar = new Identifier(gpu_declarator);
									}
								} else {
									addNewGpuSymbol = true;
									targetSymbolTable.addDeclaration(gpu_decl);
									//Check whether the current symbol declaration is within an OpenMP parallel/task region.
									//If not, add #pragma omp threadprivate directive to it.
									//[DEBUG] automatic variable can not be threadprivate.
						/*			if( (AnalysisTools.ipFindFirstPragmaInParent(targetSymbolTable, OmpAnnotation.class, new HashSet(Arrays.asList("parallel", "task")), false, null, null) == null) && 
											!((Annotatable)targetSymbolTable).containsAnnotation(OmpAnnotation.class, "parallel")  &&
											!((Annotatable)targetSymbolTable).containsAnnotation(OmpAnnotation.class, "task") )  {
										if( targetSymbolTable instanceof CompoundStatement ) {
											((CompoundStatement)targetSymbolTable).addStatementAfter((Statement)gpu_decl.getParent(), tAnnotStmt);
										}
									}*/
								}
							}
							// Replace all instances of the shared variable to the parameter variable
							Expression newExp = gpuVar;
							if( isScalar ) {
								newExp = new UnaryExpression(UnaryOperator.DEREFERENCE, (Identifier)gpuVar.clone());
							}
							if( sym instanceof AccessSymbol ) {
								TransformTools.replaceAccessExpressions(at, (AccessSymbol)sym, newExp);
							} else {
								TransformTools.replaceAll(at, hostVar, newExp);
							}
						} else {
							break;
						}
					}
				}
		}
	}
	

	/**
	 * Generate CUDA codes for all data clauses except for deviceptr clause.
	 * 
	 * @param dAnnot annotation including current data clause
	 * @param IRSym IR symbol of hostVar 
	 * @param hostVar host variable to be allocated on the GPU memory
	 * @param startList a list of start index of hostVar subarray
	 * @param lengthList a list of length of hostVar subarray.
	 * @param typeSpecs a list of types in the Declaration containing the hostVar.
	 * @param ifCondExp condition expression for explicit data region or compute region.
	 * @param asyncExp argument of async clause.
	 * @param dataClauseT data clause type
	 * @param mallocT GPU memory allocation type 
	 * @param memtrT GPU memory transfer type
	 * @param dRegionType data region type (implicit program/procedure data region, explicit data region, compute region)
	 * @param inStmts a list of reference statements where malloc-related statements are inserted.
	 * @param outStmts a list of reference statements where free-related statements are inserted.
	 */
	protected void genCUDACodesForDataClause(ACCAnnotation dAnnot, Symbol IRSym, Expression hostVar, List<Expression> startList, 
			List<Expression> lengthList, List<Specifier> typeSpecs, Expression ifCondExp, Expression asyncExp, DataClauseType dataClauseT, 
			MallocType mallocT, MemTrType memtrT, DataRegionType dRegionType, List<Statement> inStmts, List<Statement>outStmts,
			Statement asyncW1Stmt, Statement profileRegion, boolean isFirstData, boolean ROSymbol) {
		Annotatable at = dAnnot.getAnnotatable();
		SymbolTable targetSymbolTable = null;
		Set<SymbolTable> targetSymbolTables = new HashSet<SymbolTable>();
		Traversable tt = at;
		Procedure parentProc = null;
		TranslationUnit parentTrUnt = null;
		while( (tt != null) && !(tt instanceof Procedure) && !(tt instanceof TranslationUnit) ) {
			tt = tt.getParent();
		}
		if( tt instanceof Procedure ) {
			parentProc = (Procedure)tt;
		} else if( tt instanceof TranslationUnit ) {
			parentTrUnt = (TranslationUnit)tt;
		}
		if( parentTrUnt == null ) {
			parentTrUnt = (TranslationUnit)parentProc.getParent();
		}
		
		Expression loopIndex = null;
		//Find enclosing ForLoop if existing.
		tt = at.getParent();
		while( (tt != null) && !(tt instanceof Procedure) ) {
			if( tt instanceof ForLoop ) {
				loopIndex = LoopTools.getIndexVariable((ForLoop)tt);
				break;
			} else {
				tt = tt.getParent();
			}
		}
		
		boolean checkPresent = false;
		if( (dataClauseT == DataClauseType.CheckOnly) ||
				(dataClauseT == DataClauseType.CheckNMalloc) ) {
			checkPresent = true;
		}
		
		boolean genMallocCode = true;
		if( (dataClauseT == DataClauseType.CheckOnly) ||
				(dataClauseT == DataClauseType.UpdateOnly) ) {
			genMallocCode = false;
		}
		
		//Check partial array passing, which is allowed only for 1-dim arrays in update directives in the current implementation.
		//CAVEAT: To allow partial array passing, runtime should be changed too, since current runtime
		//checks only the base address of each variable.
		//CAVEAT: If partial array passing occurs, memory-transfer verification may not work correctly.
		boolean partialArrayPassing = false;
		boolean isContinuous = true;
		boolean isFirstDimension = true;
		for( Expression startIndex : startList ) {
			if( startIndex instanceof IntegerLiteral ) {
				if( ((IntegerLiteral)startIndex).getValue() != 0 ) {
					partialArrayPassing = true;
					if( !isFirstDimension ) {
						isContinuous = false;
					}
				}
			} else {
				partialArrayPassing = true;
				if( !isFirstDimension ) {
					isContinuous = false;
				}
			}
			if( partialArrayPassing && (
					(dataClauseT != DataClauseType.UpdateOnly) || !isContinuous) ) {
				Tools.exit("[ERROR in ACC2CUDATranslator.genCUDACodesForDataClause()] current implementation allows partial " +
						"array passing only for continuous subarray in update directives," +
						"but a variable (" + hostVar + ") in the following non-update directive has non-zero start index; exit!\n" +
								"Enclosing annotation: " + dAnnot + "\nEnclosing file: " + parentTrUnt.getInputFilename() + "\n");
			}
			isFirstDimension = false;
		}
		if( partialArrayPassing && (startList.size() > 1) ) {
			PrintTools.println("\n[WARNING] The current implementation assumes that the subarray in the following directive is continuous in the memory layout."
					+ "If not, the generated code will transfer incorrect data.\n"
					+ "OpenACC annotation: " + dAnnot + AnalysisTools.getEnclosingAnnotationContext(dAnnot),0);
		}
		
		if( dRegionType == DataRegionType.ImplicitProgramRegion ) {
			tt = inStmts.get(0).getParent();
			while( (tt != null) && !(tt instanceof TranslationUnit) ) {
				tt = tt.getParent();
			}
			if( tt instanceof TranslationUnit ) {
				targetSymbolTable = (SymbolTable)tt;
			}
			if( inStmts.size() > 1 ) {
				for( int i=1; i<inStmts.size(); i++ ) {
					tt = inStmts.get(i);
					while( (tt != null) && !(tt instanceof TranslationUnit) ) {
						tt = tt.getParent();
					}
					if( tt instanceof TranslationUnit ) {
						SymbolTable sTable = (SymbolTable)tt;
						if( targetSymbolTable != sTable ) {
							targetSymbolTables.add(sTable);
						}
					}
				}
			}
		} else {// either implicit data region or explicit data region/compute region
			targetSymbolTable = AnalysisTools.getIRSymbolScope(IRSym, at.getParent());
			if( targetSymbolTable instanceof Procedure ) {
				targetSymbolTable = ((Procedure)targetSymbolTable).getBody();
			}
		}
		if( targetSymbolTable == null ) {
			String scope;
			if( (dRegionType == DataRegionType.ImplicitProgramRegion) || 
					(dRegionType == DataRegionType.ImplicitProcedureRegion) ) {
				scope = "declare directive";
			} else if( dRegionType == DataRegionType.ExplicitDataRegion) {
				scope = "data region";
			} else {
				scope = "compute region";
			}
			Tools.exit("[ERROR in ACC2CUDATranslator.genCUDACodesForDataClause()] a symbol(" +
					IRSym + ") in a " + scope + " is not visible; exit!");
		}
		if( targetSymbolTable instanceof CompoundStatement ) {
			if( AnalysisTools.ipFindFirstPragmaInParent(at, OmpAnnotation.class, new HashSet(Arrays.asList("parallel", "task")), false, null, null) != null ) { 
				targetSymbolTable = (CompoundStatement)at.getParent();
			}
		}
		
		if(asyncExp != null)
		{
         	FunctionCall setAsyncCall = new FunctionCall(new NameID("HI_set_async"));
        	setAsyncCall.addArgument(asyncExp.clone());
        	if( isFirstData ) {
        		TransformTools.addStatementBefore((CompoundStatement)((Statement)at).getParent(), (Statement)at, new ExpressionStatement(setAsyncCall.clone()));
        	}
		}

		List<Specifier> clonedspecs = new ChainedList<Specifier>();
		clonedspecs.addAll(typeSpecs);
		clonedspecs.remove(Specifier.STATIC);
		///////////////////////////////////////////
		// GPU variables should not be constant. //
		///////////////////////////////////////////
		clonedspecs.remove(Specifier.CONST);
		//////////////////////////////
		// Remove extern specifier. //
		//////////////////////////////
		clonedspecs.remove(Specifier.EXTERN);
		
		VariableDeclaration bytes_decl = (VariableDeclaration)SymbolTools.findSymbol(parentTrUnt, "gpuBytes");
		Identifier cloned_bytes = new Identifier((VariableDeclarator)bytes_decl.getDeclarator(0));			
		VariableDeclaration gmem_decl = null;
		Identifier gmemsize = null;
		VariableDeclaration smem_decl = null;
		Identifier smemsize = null;
		ExpressionStatement gMemAdd_stmt = null;
		ExpressionStatement gMemSub_stmt = null;
		if( opt_addSafetyCheckingCode ) {
			gmem_decl = (VariableDeclaration)SymbolTools.findSymbol(parentTrUnt, "gpuGmemSize");
			gmemsize = new Identifier((VariableDeclarator)gmem_decl.getDeclarator(0));					
			smem_decl = (VariableDeclaration)SymbolTools.findSymbol(parentTrUnt, "gpuSmemSize");
			smemsize = new Identifier((VariableDeclarator)smem_decl.getDeclarator(0));					
			gMemAdd_stmt = new ExpressionStatement( new AssignmentExpression((Identifier)gmemsize.clone(),
					AssignmentOperator.ADD, (Identifier)cloned_bytes.clone()) );
			gMemSub_stmt = new ExpressionStatement( new AssignmentExpression((Identifier)gmemsize.clone(),
					AssignmentOperator.SUBTRACT, (Identifier)cloned_bytes.clone()) );
		}
		
		SizeofExpression sizeof_expr = new SizeofExpression(clonedspecs);
		
		///////////////////////////////////////////////////////////////////////////////
		// Create a GPU device variable corresponding to shared_var if not existing. //
		// Ex: float * gpu__b;                                                       //
		///////////////////////////////////////////////////////////////////////////////
		// Give a new name for the device variable 
		Identifier gpuVar = null;
		StringBuilder str = new StringBuilder(80);
		str.append("gpu__");
		if( hostVar instanceof AccessExpression ) {
			str.append(TransformTools.buildAccessExpressionName((AccessExpression)hostVar));
		} else {
			str.append(hostVar.toString());
		}
		Set<Symbol> symSet = targetSymbolTable.getSymbols();
		Symbol gpu_sym = AnalysisTools.findsSymbol(symSet, str.toString());
		boolean addNewGpuSymbol = false;
		if( gpu_sym != null ) {
			gpuVar = new Identifier(gpu_sym);
		} else {
			// Create a new GPU device variable.
			// The type of the device symbol should be a pointer type 
			VariableDeclarator gpu_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
					new NameID(str.toString()));
			VariableDeclaration gpu_decl = new VariableDeclaration(clonedspecs, 
					gpu_declarator);
			gpuVar = new Identifier(gpu_declarator);
			StringBuilder str2 = new StringBuilder(256);
			str2.append("#ifdef _OPENMP\n");
			str2.append("#pragma omp threadprivate(");
			str2.append(str.toString());
			str2.append(")\n");
			str2.append("#endif");
			CodeAnnotation tOmpAnnot = new CodeAnnotation(str2.toString());
			AnnotationDeclaration tAnnotDecl = new AnnotationDeclaration(tOmpAnnot);
			//AnnotationStatement tAnnotStmt = new AnnotationStatement(tOmpAnnot);
			if( targetSymbolTable instanceof TranslationUnit ) {
				symSet = main_TrUnt.getSymbols();
				gpu_sym = AnalysisTools.findsSymbol(symSet, str.toString());
				if( gpu_sym == null ) {
					addNewGpuSymbol = true;
					Declaration tLastCudaDecl = OpenACCHeaderEndMap.get(main_TrUnt);
					main_TrUnt.addDeclarationAfter(tLastCudaDecl, tAnnotDecl);
					main_TrUnt.addDeclarationAfter(tLastCudaDecl, gpu_decl);
					OpenACCHeaderEndMap.put(main_TrUnt, gpu_decl);
					if( dRegionType == DataRegionType.ImplicitProgramRegion ) {
						for( SymbolTable tTbl : targetSymbolTables ) {
							TranslationUnit tTrUnt = (TranslationUnit)tTbl;
							if( tTrUnt != main_TrUnt ) {
								gpu_declarator = gpu_declarator.clone();
								List<Specifier> extended_clonedspecs = new ChainedList<Specifier>();
								if( !clonedspecs.contains(Specifier.EXTERN) ) {
									extended_clonedspecs.add(Specifier.EXTERN);
								}
								extended_clonedspecs.addAll(clonedspecs);
								gpu_decl = new VariableDeclaration(extended_clonedspecs, 
										gpu_declarator);
								tLastCudaDecl = OpenACCHeaderEndMap.get(tTrUnt);
								tTrUnt.addDeclarationAfter(tLastCudaDecl, tAnnotDecl.clone());
								tTrUnt.addDeclarationAfter(tLastCudaDecl, gpu_decl);
								OpenACCHeaderEndMap.put(tTrUnt, gpu_decl);
								//gpuVar = new Identifier(gpu_declarator);
							}
						}
					} else {
						if( parentTrUnt != main_TrUnt ) {
							gpu_declarator = gpu_declarator.clone();
							List<Specifier> extended_clonedspecs = new ChainedList<Specifier>();
							if( !clonedspecs.contains(Specifier.EXTERN) ) {
								extended_clonedspecs.add(Specifier.EXTERN);
							}
							extended_clonedspecs.addAll(clonedspecs);
							gpu_decl = new VariableDeclaration(extended_clonedspecs, 
									gpu_declarator);
							tLastCudaDecl = OpenACCHeaderEndMap.get(parentTrUnt);
							parentTrUnt.addDeclarationAfter(tLastCudaDecl, tAnnotDecl.clone());
							parentTrUnt.addDeclarationAfter(tLastCudaDecl, gpu_decl);
							OpenACCHeaderEndMap.put(parentTrUnt, gpu_decl);
							gpuVar = new Identifier(gpu_declarator);
						}
					}
				} else { //gpuVar exists in the main translation unit, but not in the current translation unit.
					gpu_declarator = gpu_declarator.clone();
					List<Specifier> extended_clonedspecs = new ChainedList<Specifier>();
					if( !clonedspecs.contains(Specifier.EXTERN) ) {
						extended_clonedspecs.add(Specifier.EXTERN);
					}
					extended_clonedspecs.addAll(clonedspecs);
					gpu_decl = new VariableDeclaration(extended_clonedspecs, 
							gpu_declarator);
					Declaration tLastCudaDecl = OpenACCHeaderEndMap.get(parentTrUnt);
					parentTrUnt.addDeclarationAfter(tLastCudaDecl, tAnnotDecl.clone());
					parentTrUnt.addDeclarationAfter(tLastCudaDecl, gpu_decl);
					OpenACCHeaderEndMap.put(parentTrUnt, gpu_decl);
					gpuVar = new Identifier(gpu_declarator);
				}
			} else {
				addNewGpuSymbol = true;
				targetSymbolTable.addDeclaration(gpu_decl);
				//Check whether the current symbol declaration is within an OpenMP parallel/task region.
				//If not, add #pragma omp threadprivate directive to it.
				//[DEBUG] automatic variable can not be threadprivate.
	/*			if( (AnalysisTools.ipFindFirstPragmaInParent(targetSymbolTable, OmpAnnotation.class, new HashSet(Arrays.asList("parallel", "task")), false, null, null) == null) && 
						!((Annotatable)targetSymbolTable).containsAnnotation(OmpAnnotation.class, "parallel")  &&
						!((Annotatable)targetSymbolTable).containsAnnotation(OmpAnnotation.class, "task") )  {
					if( targetSymbolTable instanceof CompoundStatement ) {
						((CompoundStatement)targetSymbolTable).addStatementAfter((Statement)gpu_decl.getParent(), tAnnotStmt);
					}
				}*/
			}
		}
		

		///////////////////////////////////////////////////////////////////////////////
		// Create a pinned-host variable corresponding to shared_var if not existing. //
		// Ex: float * phost__b;                                                       //
		///////////////////////////////////////////////////////////////////////////////
		Identifier phostVar = null;
		if( kernelVerification && (dRegionType == DataRegionType.ComputeRegion) ) {
			// Give a new name for the pinned-host variable 
			str = new StringBuilder(80);
			str.append("phost__");
			if( hostVar instanceof AccessExpression ) {
				str.append(TransformTools.buildAccessExpressionName((AccessExpression)hostVar));
			} else {
				str.append(hostVar.toString());
			}
			symSet = targetSymbolTable.getSymbols();
			Symbol phost_sym = AnalysisTools.findsSymbol(symSet, str.toString());
			boolean addNewPhostSymbol = false;
			if( phost_sym != null ) {
				phostVar = new Identifier(phost_sym);
			} else {
				// Create a new pinned-host variable.
				// The type of the pinned-host symbol should be a pointer type 
				VariableDeclarator phost_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
						new NameID(str.toString()));
				VariableDeclaration phost_decl = new VariableDeclaration(clonedspecs, 
						phost_declarator);
				phostVar = new Identifier(phost_declarator);
				if( targetSymbolTable instanceof TranslationUnit ) {
					symSet = main_TrUnt.getSymbols();
					phost_sym = AnalysisTools.findsSymbol(symSet, str.toString());
					if( phost_sym == null ) {
						addNewPhostSymbol = true;
						Declaration tLastCudaDecl = OpenACCHeaderEndMap.get(main_TrUnt);
						main_TrUnt.addDeclarationAfter(tLastCudaDecl, phost_decl);
						OpenACCHeaderEndMap.put(main_TrUnt, phost_decl);
						if( dRegionType == DataRegionType.ImplicitProgramRegion ) {
							for( SymbolTable tTbl : targetSymbolTables ) {
								TranslationUnit tTrUnt = (TranslationUnit)tTbl;
								if( tTrUnt != main_TrUnt ) {
									phost_declarator = phost_declarator.clone();
									List<Specifier> extended_clonedspecs = new ChainedList<Specifier>();
									if( !clonedspecs.contains(Specifier.EXTERN) ) {
										extended_clonedspecs.add(Specifier.EXTERN);
									}
									extended_clonedspecs.addAll(clonedspecs);
									phost_decl = new VariableDeclaration(extended_clonedspecs, 
											phost_declarator);
									tLastCudaDecl = OpenACCHeaderEndMap.get(tTrUnt);
									tTrUnt.addDeclarationAfter(tLastCudaDecl, phost_decl);
									OpenACCHeaderEndMap.put(tTrUnt, phost_decl);
									//phostVar = new Identifier(phost_declarator);
								}
							}
						} else {
							if( parentTrUnt != main_TrUnt ) {
								phost_declarator = phost_declarator.clone();
								List<Specifier> extended_clonedspecs = new ChainedList<Specifier>();
								if( !clonedspecs.contains(Specifier.EXTERN) ) {
									extended_clonedspecs.add(Specifier.EXTERN);
								}
								extended_clonedspecs.addAll(clonedspecs);
								phost_decl = new VariableDeclaration(extended_clonedspecs, 
										phost_declarator);
								tLastCudaDecl = OpenACCHeaderEndMap.get(parentTrUnt);
								parentTrUnt.addDeclarationAfter(tLastCudaDecl, phost_decl);
								OpenACCHeaderEndMap.put(parentTrUnt, phost_decl);
								phostVar = new Identifier(phost_declarator);
							}
						}
					} else { //phostVar exists in the main translation unit, but not in the current translation unit.
						phost_declarator = phost_declarator.clone();
						List<Specifier> extended_clonedspecs = new ChainedList<Specifier>();
						if( !clonedspecs.contains(Specifier.EXTERN) ) {
							extended_clonedspecs.add(Specifier.EXTERN);
						}
						extended_clonedspecs.addAll(clonedspecs);
						phost_decl = new VariableDeclaration(extended_clonedspecs, 
								phost_declarator);
						Declaration tLastCudaDecl = OpenACCHeaderEndMap.get(parentTrUnt);
						parentTrUnt.addDeclarationAfter(tLastCudaDecl, phost_decl);
						OpenACCHeaderEndMap.put(parentTrUnt, phost_decl);
						phostVar = new Identifier(phost_declarator);
					}
				} else {
					addNewPhostSymbol = true;
					targetSymbolTable.addDeclaration(phost_decl);
				}
			}
		}
		
		//Get refname to be used for memory-transfer verification.
		StringLiteral refName = null;
		if( memtrVerification ) {
			ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "refname");
			if( iAnnot == null ) {
				str = new StringBuilder("[ERROR in ACC2CUDATranslator.genCUDACodesForDataClause()] can not find referenc name " +
						"used for memory transfer verification; please turn off the verification option " +
						"(programVerification != 1).\n" +
						"OpenACC Annotation: " + dAnnot + "\n");
				if( parentProc != null ) {
					str.append("Enclosing Procedure: " + parentProc.getSymbolName() + "\n");
				} else if( parentTrUnt != null ) {
					str.append("Enclosing File: " + parentTrUnt.getInputFilename() + "\n");
				}
				Tools.exit(str.toString());
			} else {
				refName = new StringLiteral((String)iAnnot.get("refname"));
			}
		}
		
		ExpressionStatement malloc_stmt = null;
		ExpressionStatement free_stmt = null;
		ExpressionStatement gpuBytes_stmt = null;
		ExpressionStatement gpuBytes_stmt2 = null;
		ExpressionStatement textureBind_stmt = null;
		ExpressionStatement textureUnbind_stmt = null;
		ExpressionStatement copyin_stmt = null;
		ExpressionStatement copyout_stmt = null;
		CompoundStatement presentErrorCode = null;
		Identifier textureRefID = null;
		Identifier pitchID = null;
		Identifier constantID = null;
		boolean addCopyInStmt = false;
		boolean addCopyOutStmt = false;
		if( memtrT == MemTrType.CopyIn ) {
			addCopyInStmt = true;
		} else if( memtrT == MemTrType.CopyOut ) {
			addCopyOutStmt = true;
		} else if( memtrT == MemTrType.CopyInOut ) {
			addCopyInStmt = true;
			addCopyOutStmt = true;
		}
		List<Statement> preambleList = new LinkedList<Statement>();
		List<Statement> postscriptList = new LinkedList<Statement>();
		//List<Statement> kernelVerifyCodes = new LinkedList<Statement>();
		
		String profRegionLabel = null;
		Expression induction = null;
		Statement measureCopyInStmt = null;
		Statement measureCopyOutStmt = null;
		if( enableCustomProfiling && (profileRegion != null) && 
				(addCopyInStmt || addCopyOutStmt) ) {
			if( profileRegion != at ) {
				ACCAnnotation profileAnnot = profileRegion.getAnnotation(ACCAnnotation.class, "track");
				if( profileAnnot != null ) {
					profRegionLabel = profileAnnot.get("label");
					induction = profileAnnot.get("induction");
				}
			}
			CompoundStatement gpRegion = (CompoundStatement)profileRegion.getParent();
			FunctionCall fCall = new FunctionCall(new NameID("HI_profile_track"));
			StringBuilder str1 = new StringBuilder(80);
			if( profRegionLabel != null ) {
				str1.append(profRegionLabel);
			}
			str1.append(": ");
			if( addCopyInStmt && addCopyOutStmt ) {
				str1.append("HICOPY");
			} else if( addCopyInStmt ) {
				str1.append("HICOPYIN");
			} else if( addCopyOutStmt ) {
				str1.append("HICOPYOUT");
			}
			fCall.addArgument(new StringLiteral(str1.toString()));
			fCall.addArgument(new StringLiteral(hostVar.toString()));
			if( induction == null ) {
				fCall.addArgument(new StringLiteral(CustomProfilingTransformation.HINoInduction));
			} else {
				fCall.addArgument(new StringLiteral(induction.toString()));
			}
			fCall.addArgument(CustomProfilingTransformation.booleanFalse.clone());
			Statement fCallStmt = new ExpressionStatement(fCall);
			gpRegion.addStatementBefore(profileRegion, fCallStmt);
			if( addCopyInStmt ) {
				fCall = new FunctionCall(new NameID("HI_profile_measure_userevent"));
				str1 = new StringBuilder(80);
				str1.append("HICOPYIN: ");
				str1.append(hostVar.toString());
				fCall.addArgument(new StringLiteral(str1.toString()));
				List specs = new ArrayList(1);
				specs.add(Specifier.DOUBLE);
				fCall.addArgument(new Typecast(specs, cloned_bytes.clone()));
				measureCopyInStmt = new ExpressionStatement(fCall);
			}
			if( addCopyOutStmt ) {
				fCall = new FunctionCall(new NameID("HI_profile_measure_userevent"));
				str1 = new StringBuilder(80);
				str1.append("HICOPYOUT: ");
				str1.append(hostVar.toString());
				fCall.addArgument(new StringLiteral(str1.toString()));
				List specs = new ArrayList(1);
				specs.add(Specifier.DOUBLE);
				fCall.addArgument(new Typecast(specs, cloned_bytes.clone()));
				measureCopyOutStmt = new ExpressionStatement(fCall);
			}
		}
		
					
		if( mallocT == MallocType.TextureMalloc ) {
			/////////////////////////////////
			// Create a texture reference. //
			/////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////
			// CUDA Toolkit API 4.0 Ch3.2.10.1.1: A texture reference can only be declared as a //
			// static global variable and cannot be passed as an argument to a function.        //
			//////////////////////////////////////////////////////////////////////////////////////
			//  texture<float, cudaTextureType1D, cudaReadModeElementType> texture__var;  //
			////////////////////////////////////////////////////////////////////////////////
			str = new StringBuilder(80);
			str.append("texture__");
			if( hostVar instanceof AccessExpression ) {
				str.append(TransformTools.buildAccessExpressionName((AccessExpression)hostVar));
			} else {
				str.append(hostVar.toString());
			}
			if( !SymbolTools.isGlobal(IRSym) ) {
				str.append("__" + parentProc.getSymbolName());
			}

			Set<Symbol> symbolSet = kernelsTranslationUnit.getSymbols();
			//Find the symbol from kernels translation unit instead
			/*
			Set<Symbol> symbolSet = null;
			if( dRegionType == DataRegionType.ImplicitProgramRegion ) {
				symbolSet = targetSymbolTable.getSymbols();
			} else {
				symbolSet = parentTrUnt.getSymbols();
			}*/
			Symbol textureSym = AnalysisTools.findsSymbol(symbolSet, str.toString());
			boolean genNewTexRef = false;
			if( textureSym != null ) {
				textureRefID = new Identifier((VariableDeclarator)textureSym);
				//DEBUG: For implicit program-level data region, multiple textureRefID may be needed.
			} else {
				genNewTexRef = true;
				//[DEBUG] below constructor generates code compatible with CUDA driver API v4.0 or higher.
				//For backward compatibility, use old constructor.
				//TextureSpecifier texturespec = new TextureSpecifier(clonedspecs);
				TextureSpecifier texturespec = new TextureSpecifier(clonedspecs, 1, "cudaReadModeElementType");
				VariableDeclarator textureRef_declarator = new VariableDeclarator(new NameID(str.toString()));
				Declaration textureRef_decl = new VariableDeclaration(texturespec, textureRef_declarator);
				textureRefID = new Identifier(textureRef_declarator); 
				//Insert texture reference declaration.
				if( dRegionType == DataRegionType.ImplicitProgramRegion ) {
					TranslationUnit tUnt = (TranslationUnit)targetSymbolTable;
					Declaration tLastCUDADecl = OpenACCHeaderEndMap.get(tUnt);
					tUnt.addDeclarationAfter(tLastCUDADecl, textureRef_decl);
					OpenACCHeaderEndMap.put(tUnt, textureRef_decl);
					if( !targetSymbolTables.isEmpty() ) {
						for( SymbolTable sTable : targetSymbolTables ) {
							tUnt = (TranslationUnit)sTable;
							tLastCUDADecl = OpenACCHeaderEndMap.get(tUnt);
							Declaration clonedTextureRefDecl = textureRef_decl.clone();
							//Move texture declaration to kernel file
							//tUnt.addDeclarationAfter(tLastCUDADecl, clonedTextureRefDecl);
							Procedure ttProc = AnalysisTools.findFirstProcedure(kernelsTranslationUnit);
							if( ttProc == null ) {
								kernelsTranslationUnit.addDeclaration(clonedTextureRefDecl);
							} else {
								kernelsTranslationUnit.addDeclarationBefore(ttProc, clonedTextureRefDecl);
							}
							//OpenACCHeaderEndMap.put(tUnt, clonedTextureRefDecl);
							//DEBUG: multiple textureRefID should be needed depending on the tUnt.
						}
					}
				} else {
					Declaration tLastCUDADecl = OpenACCHeaderEndMap.get(parentTrUnt);
					//Move texture declaration to kernel file
					//parentTrUnt.addDeclarationAfter(tLastCUDADecl, textureRef_decl);
					Procedure ttProc = AnalysisTools.findFirstProcedure(kernelsTranslationUnit);
					if( ttProc == null ) {
						kernelsTranslationUnit.addDeclaration(textureRef_decl);
					} else {
						kernelsTranslationUnit.addDeclarationBefore(ttProc, textureRef_decl);
					}
					//OpenACCHeaderEndMap.put(parentTrUnt, textureRef_decl);
				}
			}
			
			///////////////////////////////////////////////////////
			// Allocate GPU global memory for the host variable. //
			///////////////////////////////////////////////////////////////////
			//  gpuBytes=((193536*4)*sizeof (int));                          //
			//  HI_malloc1D( hostPtr, ((void**)(& devPtr), gpuBytes ); //
			//  HI_malloc1D( &hostVar, ((void**)(& devPtr), gpuBytes );//
			///////////////////////////////////////////////////////////////////
			// CAVEAT: the above will work only if hostPtr pointer points to //
			// continuous memory. ( **hostPtr may not work. )                //
			///////////////////////////////////////////////////////////////////
			// Add malloc size (gpuBytes) statement
			// Ex: gpuBytes=sizeof(float)*((2048+2)*(2048+2));
			Expression biexp = sizeof_expr.clone();
			for( int i=0; i<lengthList.size(); i++ )
			{
				biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
			}
			AssignmentExpression assignex = new AssignmentExpression(cloned_bytes.clone(),AssignmentOperator.NORMAL, 
					biexp);
			gpuBytes_stmt = new ExpressionStatement(assignex);
			boolean gpuBytesStmtAdded = false;
			if( genMallocCode ) {
				preambleList.add(gpuBytes_stmt);
				gpuBytesStmtAdded = true;
				// Add malloc statement
				// Ex: HI_malloc1D( hostPtr, ((void**)(& devPtr), gpuBytes );  //
				FunctionCall malloc_call = new FunctionCall(new NameID("HI_malloc1D"));
				List<Expression> arg_list = new ArrayList<Expression>();
				if( lengthList.size() == 0 ) { //hostVar is scalar.
					arg_list.add( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone()));
				} else {
					arg_list.add(hostVar.clone());
				}

				//Check async condition
                ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "async");
                Expression asyncID = getAsyncExpression(tAnnot);

				List<Specifier> specs = new ArrayList<Specifier>(4);
				specs.add(Specifier.VOID);
				specs.add(PointerSpecifier.UNQUALIFIED);
				specs.add(PointerSpecifier.UNQUALIFIED);
				arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
						gpuVar.clone())));
				arg_list.add(cloned_bytes.clone());
				arg_list.add(asyncID.clone());
				if( ROSymbol ) {
					arg_list.add(new NameID("HI_MEM_READ_ONLY"));
				} else {
					arg_list.add(new NameID("HI_MEM_READ_WRITE"));
				}
				malloc_call.setArguments(arg_list);
				malloc_stmt = new ExpressionStatement(malloc_call);
				preambleList.add(malloc_stmt);
				if( opt_addSafetyCheckingCode ) {
					preambleList.add(gMemAdd_stmt.clone());
				}
			}
			
			if( genMallocCode || ((dataClauseT == DataClauseType.CheckOnly) && genNewTexRef) ) {
				if( !gpuBytesStmtAdded ) {
					preambleList.add(gpuBytes_stmt);
					gpuBytesStmtAdded = true;
				}
				////////////////////////////////////////////////////////////
				// Bind a texture reference to a GPU global memory.       //
				////////////////////////////////////////////////////////////
				//  cudaBindTexture(0, texture__var, gpuPtr, gpuBytes);   //
				////////////////////////////////////////////////////////////
				/*
				FunctionCall textureBindCall = new FunctionCall(new NameID("cudaBindTexture"));
				textureBindCall.addArgument(new IntegerLiteral(0));
				textureBindCall.addArgument(textureRefID.clone());
				textureBindCall.addArgument(gpuVar.clone());
				textureBindCall.addArgument(cloned_bytes.clone());
				textureBind_stmt = new ExpressionStatement(textureBindCall);
				preambleList.add(textureBind_stmt);
				*/
				//New OpenARC runtime use HI_bind_tex(name, type, gpu_ptr, size) instead
                FunctionCall textureBindCall = new FunctionCall(new NameID("HI_bind_tex"));
                textureBindCall.addArgument(new StringLiteral(textureRefID.getName()));
                if(clonedspecs.size() > 1)
                {
                    Tools.exit("[ERROR] Texture specifier list size is larger than 1");
                }
                Specifier texSpec = clonedspecs.get(0);
                NameID typeArg = null;
                if(texSpec ==  Specifier.INT) {typeArg = new NameID("HI_int");}
                else if(texSpec ==  Specifier.FLOAT) {typeArg = new NameID("HI_float");}
                else if(texSpec ==  Specifier.DOUBLE) {typeArg = new NameID("HI_double");}
                else
                {
                    Tools.exit("[ERROR] Texture type " + texSpec.toString() + " is not supported");
                }
                textureBindCall.addArgument(typeArg);
                textureBindCall.addArgument(gpuVar.clone());
                textureBindCall.addArgument(cloned_bytes.clone());
                textureBind_stmt = new ExpressionStatement(textureBindCall);
                preambleList.add(textureBind_stmt);
			}
			
			///////////////////////////////////////////////
			// Copyin the host data into the GPU memory. //
			/////////////////////////////////////////////////////////////////////////////////////////
			// HI_memcpy(gpuPtr, hostPtr, gpuBytes, HI_MemcpyHostToDevice, 0);               //
			// HI_memcpy_async(gpuPtr, hostPtr, gpuBytes, HI_MemcpyHostToDevice, 0, asyncID);//
			/////////////////////////////////////////////////////////////////////////////////////////
			if( addCopyInStmt ) {
				if( !gpuBytesStmtAdded ) {
					preambleList.add(gpuBytes_stmt);
					gpuBytesStmtAdded = true;
				}
				FunctionCall copyinCall = null;
				if( asyncExp == null ) {
					copyinCall = new FunctionCall(new NameID("HI_memcpy"));
				} else {
					copyinCall = new FunctionCall(new NameID("HI_memcpy_async"));
				}
				if( partialArrayPassing ) {
					copyinCall.addArgument(new BinaryExpression(gpuVar.clone(), BinaryOperator.ADD,
							startList.get(0).clone()));
				} else {
					copyinCall.addArgument(gpuVar.clone());
				}
				if( lengthList.size() == 0 ) { //hostVar is scalar.
					copyinCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone()));
				} else {
					if( partialArrayPassing ) {
						copyinCall.addArgument(new BinaryExpression(hostVar.clone(), BinaryOperator.ADD,
								startList.get(0).clone()));
					} else {
						copyinCall.addArgument(hostVar.clone());
					}
				}
				copyinCall.addArgument(cloned_bytes.clone());
				copyinCall.addArgument(new NameID("HI_MemcpyHostToDevice"));
				copyinCall.addArgument(new IntegerLiteral(0));
				if( asyncExp != null ) {
					copyinCall.addArgument(asyncExp.clone());
				}
				copyin_stmt = new ExpressionStatement(copyinCall);
				preambleList.add(copyin_stmt);
				
				//Insert HI_profile_measure_userevent("HICOPYIN: var", (double)copy_size)
				//call for this variable if enableCustomPrfiling option is on.
				if( enableCustomProfiling && (measureCopyInStmt != null) ) {
					preambleList.add(measureCopyInStmt.clone());
				}
				
				//Insert set_status() call for this variable if memtrVerification is used.
				if( memtrVerification ) {
					FunctionCall setStatusCall = new FunctionCall(new NameID("HI_set_status"));
					if( lengthList.size() == 0 ) { //hostVar is scalar.
						setStatusCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								hostVar.clone()));
					} else {
						setStatusCall.addArgument(hostVar.clone());
					}
					setStatusCall.addArgument(new NameID("acc_device_nvidia"));
					setStatusCall.addArgument(new NameID("HI_notstale"));
					setStatusCall.addArgument(new StringLiteral(hostVar.toString()));
					setStatusCall.addArgument(refName.clone());
					if( loopIndex != null ) {
						setStatusCall.addArgument(loopIndex.clone());
					} else {
						setStatusCall.addArgument(new NameID("INT_MIN"));
					}
					preambleList.add(new ExpressionStatement(setStatusCall));
				}
			}
			
			//////////////////////////////////////////////////
			// Copyout the GPU data back to the CPU memory. //
			/////////////////////////////////////////////////////////////////////////////////////////
			// gpuBytes=sizeof(float)*((2048+2)*(2048+2));                                         //
			// HI_memcpy(hostPtr, gpuPtr, gpuBytes, HI_MemcpyDeviceToHost, 0);               //
			// HI_memcpy_async(hostPtr, gpuPtr, gpuBytes, HI_MemcpyDeviceToHost, 0, asyncID);//
			/////////////////////////////////////////////////////////////////////////////////////////
			if( addCopyOutStmt ) {
				gpuBytes_stmt2 = gpuBytes_stmt.clone();
				postscriptList.add(gpuBytes_stmt2);
				FunctionCall copyoutCall = null;
				if( asyncExp == null ) {
					copyoutCall = new FunctionCall(new NameID("HI_memcpy"));
				} else {
					if( kernelVerification ) {
						copyoutCall = new FunctionCall(new NameID("HI_memcpy_asyncS"));
					} else {
						copyoutCall = new FunctionCall(new NameID("HI_memcpy_async"));
					}
				}
				if( lengthList.size() == 0 ) { //hostVar is scalar.
					copyoutCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone()));
				} else {
					if( partialArrayPassing ) {
						copyoutCall.addArgument(new BinaryExpression(hostVar.clone(), BinaryOperator.ADD,
								startList.get(0).clone()));
					} else {
						copyoutCall.addArgument(hostVar.clone());
					}
				}
				if( partialArrayPassing ) {
					copyoutCall.addArgument(new BinaryExpression(gpuVar.clone(), BinaryOperator.ADD,
							startList.get(0).clone()));
				} else {
					copyoutCall.addArgument(gpuVar.clone());
				}
				copyoutCall.addArgument(cloned_bytes.clone());
				copyoutCall.addArgument(new NameID("HI_MemcpyDeviceToHost"));
				copyoutCall.addArgument(new IntegerLiteral(0));
				if( asyncExp != null ) {
					copyoutCall.addArgument(asyncExp.clone());
				}
				copyout_stmt = new ExpressionStatement(copyoutCall);
				postscriptList.add(copyout_stmt);
				
				//Insert HI_profile_measure_userevent("HICOPYOUT: var", (double)copy_size)
				//call for this variable if enableCustomPrfiling option is on.
				if( enableCustomProfiling && (measureCopyOutStmt != null) ) {
					postscriptList.add(measureCopyOutStmt.clone());
				}
				
				//Insert set_status() call for this variable if memtrVerification is used.
				if( memtrVerification ) {
					FunctionCall setStatusCall = new FunctionCall(new NameID("HI_set_status"));
					if( lengthList.size() == 0 ) { //hostVar is scalar.
						setStatusCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								hostVar.clone()));
					} else {
						setStatusCall.addArgument(hostVar.clone());
					}
					setStatusCall.addArgument(new NameID("acc_device_host"));
					setStatusCall.addArgument(new NameID("HI_notstale"));
					setStatusCall.addArgument(new StringLiteral(hostVar.toString()));
					setStatusCall.addArgument(refName.clone());
					if( loopIndex != null ) {
						setStatusCall.addArgument(loopIndex.clone());
					} else {
						setStatusCall.addArgument(new NameID("INT_MIN"));
					}
					postscriptList.add(new ExpressionStatement(setStatusCall));
				}
				//Generate result-compare codes for this variable.
				if( asyncW1Stmt != null ) {
					TransformTools.genResultCompareCodes(parentProc, asyncW1Stmt, hostVar, phostVar, pitchID, lengthList, clonedspecs,
							dAnnot, marginOfError, true, minCheckValue);
				}
			}
			
			//if( genMallocCode || ((dataClauseT == DataClauseType.CheckOnly) && genNewTexRef) )
			if( genMallocCode ) {
				// New OpenARC runtime does not require this fuction
				//////////////////////////////////////////////////////////////
				// Unbind the texture reference from the GPU global memory. //
				//////////////////////////////////////////////////////////////
				// cudaUnbindTexture(texture__var);                         //
				//////////////////////////////////////////////////////////////
				/*FunctionCall textureUnbindCall = new FunctionCall(new NameID("cudaUnbindTexture"));
				textureUnbindCall.addArgument((Identifier)textureRefID.clone());
				textureUnbind_stmt = new ExpressionStatement(textureUnbindCall);
				postscriptList.add(textureUnbind_stmt);
				*/
			}
			
			if( genMallocCode ) {
				////////////////////////////
				// Deallocate GPU memory. //
				////////////////////////////
				// HI_free(hostPtr);   //
				////////////////////////////
				FunctionCall free_call = null;

				if( asyncExp == null ) {
					free_call = new FunctionCall(new NameID("HI_free"));
				} else {
					free_call = new FunctionCall(new NameID("HI_free_async"));
				}

				if( lengthList.size() == 0 ) { //hostVar is scalar.
					free_call.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone()));
				} else {
					free_call.addArgument(hostVar.clone());
				}

				//Check async condition
                ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "async");
                Expression asyncID = getAsyncExpression(tAnnot);

				free_call.addArgument(asyncID.clone());

				/*if( asyncExp != null ) {
					free_call.addArgument(asyncExp.clone());
				}*/
				free_stmt = new ExpressionStatement(free_call);
				postscriptList.add(free_stmt);
				if( opt_addSafetyCheckingCode ) {
					postscriptList.add(gpuBytes_stmt.clone());
					postscriptList.add(gMemSub_stmt.clone());
				}
				//Insert reset_status(GPU, stale) call for this variable if memtrVerification is used.
				if( memtrVerification ) {
					FunctionCall setStatusCall = new FunctionCall(new NameID("HI_reset_status"));
					if( lengthList.size() == 0 ) { //hostVar is scalar.
						setStatusCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								hostVar.clone()));
					} else {
						setStatusCall.addArgument(hostVar.clone());
					}
					setStatusCall.addArgument(new NameID("acc_device_nvidia"));
					setStatusCall.addArgument(new NameID("HI_stale"));
					if( asyncExp == null ) {
						//setStatusCall.addArgument(new NameID("INT_MIN"));
						setStatusCall.addArgument(new NameID("DEFAULT_QUEUE"));
					} else {
						setStatusCall.addArgument(asyncExp.clone());
					}
					postscriptList.add(new ExpressionStatement(setStatusCall));
				}
			}
			
		} else if( mallocT == MallocType.ConstantMalloc ) {
			////////////////////////////////////////////////////////////////////////////
			//CUDA Toolkit API V4.2 says that __constant__ variables are only allowed //
			//at a file scope. __constant__ variable can not be declared as external  //
			//using the extern keyword.                                               //
			////////////////////////////////////////////////////////////////////////////
			// FIXME: if a compute region is in a file different from that of the //
			// enclosing data region, data may need to be copied again.           //
			////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////
			// Create a __constant__ variable. //
			/////////////////////////////////////
			//  __constant__ float a[SIZE1][SIZE2];  //
			/////////////////////////////
			str = new StringBuilder(80);
			str.append("const__");
			if( hostVar instanceof AccessExpression ) {
				str.append(TransformTools.buildAccessExpressionName((AccessExpression)hostVar));
			} else {
				str.append(hostVar.toString());
			}
			if( !SymbolTools.isGlobal(IRSym) ) {
				str.append("__" + parentProc.getSymbolName());
			}
            Set<Symbol> symbolSet = kernelsTranslationUnit.getSymbols();
            /*
			Set<Symbol> symbolSet = null;
			if( dRegionType == DataRegionType.ImplicitProgramRegion ) {
				symbolSet = targetSymbolTable.getSymbols();
			} else {
				symbolSet = parentTrUnt.getSymbols();
			}
			*/
			Symbol constantSym = AnalysisTools.findsSymbol(symbolSet, str.toString());
			boolean addNewConstSymbol = false;
			if( constantSym != null ) {
				constantID = new Identifier((VariableDeclarator)constantSym);
				//DEBUG: For implicit program-level data region, multiple constantID may be needed.
			} else {
				addNewConstSymbol = true;
				List<Expression> arryDimList = new ArrayList<Expression>();
				for( int i=0; i<lengthList.size(); i++ ) {
					arryDimList.add(lengthList.get(i).clone());
				}
				ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
				VariableDeclarator constantRef_declarator = new VariableDeclarator(new NameID(str.toString()), arraySpecs);
				List<Specifier> constspecs = new ChainedList<Specifier>();
				constspecs.add(CUDASpecifier.CUDA_CONSTANT);
				constspecs.addAll(clonedspecs);
				Declaration constantRef_decl = new VariableDeclaration(constspecs, constantRef_declarator);
				constantID = new Identifier(constantRef_declarator); 
				//Insert __constant__ variable declaration.
				if( dRegionType == DataRegionType.ImplicitProgramRegion ) {
                    //kernelsTranslationUnit.addDeclarationAfter(accHeaderDecl, constantRef_decl);
					Procedure ttProc = AnalysisTools.findFirstProcedure(kernelsTranslationUnit);
					if( ttProc == null ) {
						kernelsTranslationUnit.addDeclaration(constantRef_decl);
					} else {
						kernelsTranslationUnit.addDeclarationBefore(ttProc, constantRef_decl);
					}
                    /*
                    TranslationUnit tUnt = (TranslationUnit)targetSymbolTable;
					Declaration tLastCUDADecl = OpenACCHeaderEndMap.get(tUnt);
					tUnt.addDeclarationAfter(tLastCUDADecl, constantRef_decl);
					OpenACCHeaderEndMap.put(tUnt, constantRef_decl);
					if( !targetSymbolTables.isEmpty() ) {
						for( SymbolTable sTable : targetSymbolTables ) {
							tUnt = (TranslationUnit)sTable;
							tLastCUDADecl = OpenACCHeaderEndMap.get(tUnt);
							Declaration clonedConstantRefDecl = constantRef_decl.clone();
							tUnt.addDeclarationAfter(tLastCUDADecl, clonedConstantRefDecl);
							OpenACCHeaderEndMap.put(tUnt, clonedConstantRefDecl);
							//DEBUG: multiple constantID should be needed depending on the tUnt.
						}
					}
					*/
				} else {
					//Declaration tLastCUDADecl = OpenACCHeaderEndMap.get(parentTrUnt);
					//parentTrUnt.addDeclarationAfter(tLastCUDADecl, constantRef_decl);
					//OpenACCHeaderEndMap.put(parentTrUnt, constantRef_decl);
                    //kernelsTranslationUnit.addDeclarationAfter(accHeaderDecl, constantRef_decl);
					Procedure ttProc = AnalysisTools.findFirstProcedure(kernelsTranslationUnit);
					if( ttProc == null ) {
						kernelsTranslationUnit.addDeclaration(constantRef_decl);
					} else {
						kernelsTranslationUnit.addDeclarationBefore(ttProc, constantRef_decl);
					}
				}
				//PrintTools.println(kernelsTranslationUnit.toString(), 0);
			}
			
			///////////////////////////////////////////////
			// Copyin the host data into the GPU memory. //
			/////////////////////////////////////////////////////////////////////////////////////////////////////
			// gpuBytes=sizeof(float)*((2048+2)*(2048+2));                                                     //
			// HI_memcpy_const((void *)hostPtr, "constant__x", HI_MemcpyHostToDevice, gpuBytes);               //
			// HI_memcpy_const_async((void *)hostPtr, "constant__x", HI_MemcpyHostToDevice, gpuBytes, asyncID);//
			/////////////////////////////////////////////////////////////////////////////////////////////////////
			if( addCopyInStmt || ((dataClauseT == DataClauseType.CheckOnly) && addNewConstSymbol) ) {
				Expression biexp = sizeof_expr.clone();
				for( int i=0; i<lengthList.size(); i++ )
				{
					biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
				}
				AssignmentExpression assignex = new AssignmentExpression(cloned_bytes.clone(),AssignmentOperator.NORMAL, 
						biexp);
				gpuBytes_stmt = new ExpressionStatement(assignex);
				preambleList.add(gpuBytes_stmt);
				FunctionCall copyinCall = null;
				if( asyncExp == null ) {
					copyinCall = new FunctionCall(new NameID("HI_memcpy_const"));
				} else {
					//[DEBUG] OpenARC CUDA driver uses cuModuleGetGlobal() driver API, but it does not offer
					//asynchronous version.
					copyinCall = new FunctionCall(new NameID("HI_memcpy_const_async"));
				}
				List<Specifier> castspecs = new LinkedList<Specifier>();
				castspecs.add(Specifier.VOID);
				castspecs.add(PointerSpecifier.UNQUALIFIED);
				if( lengthList.size() == 0 ) { //hostVar is scalar.
					//copyinCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
					//		constantID.clone()));
					copyinCall.addArgument(new Typecast(castspecs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone())));
				} else {
					//if( partialArrayPassing ) {
					//	copyinCall.addArgument(new BinaryExpression(constantID.clone(), BinaryOperator.ADD,
					//			startList.get(0).clone()));
					//} else {
					//	copyinCall.addArgument(constantID.clone());
					//}
					if( partialArrayPassing ) {
						copyinCall.addArgument(new Typecast(castspecs, new BinaryExpression(hostVar.clone(), BinaryOperator.ADD,
								startList.get(0).clone())));
					} else {
						copyinCall.addArgument(new Typecast(castspecs, hostVar.clone()));
					}
				}
                /*
				copyinCall.addArgument(cloned_bytes.clone());
				copyinCall.addArgument(new NameID("HI_MemcpyHostToDevice"));
				copyinCall.addArgument(new IntegerLiteral(1));
				*/
                copyinCall.addArgument(new StringLiteral(constantID.getName()));
                copyinCall.addArgument(new NameID("HI_MemcpyHostToDevice"));
                copyinCall.addArgument(cloned_bytes.clone());

                if( asyncExp != null ) {
					copyinCall.addArgument(asyncExp.clone());
				}
				copyin_stmt = new ExpressionStatement(copyinCall);
				preambleList.add(copyin_stmt);
				
				//Insert HI_profile_measure_userevent("HICOPYIN: var", (double)copy_size)
				//call for this variable if enableCustomPrfiling option is on.
				if( enableCustomProfiling && (measureCopyInStmt != null) ) {
					preambleList.add(measureCopyInStmt.clone());
				}
				
				//Insert set_status() call for this variable if memtrVerification is used.
				if( memtrVerification ) {
					FunctionCall setStatusCall = new FunctionCall(new NameID("HI_set_status"));
					if( lengthList.size() == 0 ) { //hostVar is scalar.
						setStatusCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								hostVar.clone()));
					} else {
						setStatusCall.addArgument(hostVar.clone());
					}
					setStatusCall.addArgument(new NameID("acc_device_nvidia"));
					setStatusCall.addArgument(new NameID("HI_notstale"));
					setStatusCall.addArgument(new StringLiteral(hostVar.toString()));
					setStatusCall.addArgument(refName.clone());
					if( loopIndex != null ) {
						setStatusCall.addArgument(loopIndex.clone());
					} else {
						setStatusCall.addArgument(new NameID("INT_MIN"));
					}
					preambleList.add(new ExpressionStatement(setStatusCall));
				}
			}
			
		} else if( mallocT == MallocType.PitchedMalloc ) {
			///////////////////////////
			// Add pitch variable.   //
			///////////////////////////
			// size_t pitch_hostVar; //
			///////////////////////////
			str = new StringBuilder(80);
			str.append("pitch__");
			if( hostVar instanceof AccessExpression ) {
				str.append(TransformTools.buildAccessExpressionName((AccessExpression)hostVar));
			} else {
				str.append(hostVar.toString());
			}
/*			if( !SymbolTools.isGlobal(IRSym) ) {
				str.append("__" + parentProc.getSymbolName());
			}
*/			Set<Symbol> symbolSet = targetSymbolTable.getSymbols();
			Symbol pitchSym = AnalysisTools.findsSymbol(symbolSet, str.toString());
			boolean addNewPitchSymbol = false;
			if( pitchSym != null ) {
				pitchID = new Identifier(pitchSym);
			} else {
				VariableDeclarator pitch_declarator = new VariableDeclarator(new NameID(str.toString()));
				VariableDeclaration pitch_decl = new VariableDeclaration(OpenACCSpecifier.SIZE_T, 
						pitch_declarator);
				pitchID = new Identifier(pitch_declarator);
				if( targetSymbolTable instanceof TranslationUnit ) {
					symbolSet = main_TrUnt.getSymbols();
					pitchSym = AnalysisTools.findsSymbol(symbolSet, str.toString());
					if( pitchSym == null ) {
						addNewPitchSymbol = true;
						Declaration tLastCudaDecl = OpenACCHeaderEndMap.get(main_TrUnt);
						main_TrUnt.addDeclarationAfter(tLastCudaDecl, pitch_decl);
						OpenACCHeaderEndMap.put(main_TrUnt, pitch_decl);
						if( dRegionType == DataRegionType.ImplicitProgramRegion ) {
							for( SymbolTable tTbl : targetSymbolTables ) {
								TranslationUnit tTrUnt = (TranslationUnit)tTbl;
								if( tTrUnt != main_TrUnt ) {
									pitch_declarator = pitch_declarator.clone();
									List<Specifier> extended_specs = new ChainedList<Specifier>();
									extended_specs.add(Specifier.EXTERN);
									extended_specs.add(OpenACCSpecifier.SIZE_T);
									pitch_decl = new VariableDeclaration(extended_specs, 
											pitch_declarator);
									tLastCudaDecl = OpenACCHeaderEndMap.get(tTrUnt);
									tTrUnt.addDeclarationAfter(tLastCudaDecl, pitch_decl);
									OpenACCHeaderEndMap.put(tTrUnt, pitch_decl);
									//pitchID = new Identifier(pitch_declarator);
								}
							}
						} else {
							if( parentTrUnt != main_TrUnt ) {
								pitch_declarator = pitch_declarator.clone();
								List<Specifier> extended_specs = new ChainedList<Specifier>();
								extended_specs.add(Specifier.EXTERN);
								extended_specs.add(OpenACCSpecifier.SIZE_T);
								pitch_decl = new VariableDeclaration(extended_specs, 
										pitch_declarator);
								tLastCudaDecl = OpenACCHeaderEndMap.get(parentTrUnt);
								parentTrUnt.addDeclarationAfter(tLastCudaDecl, pitch_decl);
								OpenACCHeaderEndMap.put(parentTrUnt, pitch_decl);
								pitchID = new Identifier(pitch_declarator);
							}
						}
					} else {
						pitch_declarator = pitch_declarator.clone();
						List<Specifier> extended_specs = new ChainedList<Specifier>();
						extended_specs.add(Specifier.EXTERN);
						extended_specs.add(OpenACCSpecifier.SIZE_T);
						pitch_decl = new VariableDeclaration(extended_specs, 
								pitch_declarator);
						Declaration tLastCudaDecl = OpenACCHeaderEndMap.get(parentTrUnt);
						parentTrUnt.addDeclarationAfter(tLastCudaDecl, pitch_decl);
						OpenACCHeaderEndMap.put(parentTrUnt, pitch_decl);
						pitchID = new Identifier(pitch_declarator);
					}
				} else {
					addNewPitchSymbol = true;
					targetSymbolTable.addDeclaration(pitch_decl);
				}
			}
			
			if( genMallocCode ) {
				///////////////////////////////////////////////////////
				// Allocate GPU global memory for the host variable. //
				///////////////////////////////////////////////////////////////////////////////////////////
				//  HI_malloc2D( hostPtr, ((void**)(& devPtr), &pitch, widthInBytes, height );        //
				//  HI_malloc3D( hostPtr, ((void**)(& devPtr), &pitch, widthInBytes, height, depth ); //
				//  gpuBytes=(pitch * height);                                                           //
				//  gpuBytes=(pitch * height * depth);                                                   //
				///////////////////////////////////////////////////////////////////////////////////////////
				// CAVEAT: the above will work only if hostPtr pointer points to //
				// continuous memory. ( **hostPtr may not work. )                //
				///////////////////////////////////////////////////////////////////
				// Add malloc statement
				// Ex: HI_malloc2D( hostPtr, ((void**)(& devPtr), &pitch, widthInBytes, height );
				FunctionCall malloc_call = null;
				if( lengthList.size() == 2 ) { //hostVar is 2D array.
					malloc_call = new FunctionCall(new NameID("HI_malloc2D"));
				} else { //hostVar is 3D array.
					malloc_call = new FunctionCall(new NameID("HI_malloc3D"));
				}

				//Check async condition
                ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "async");
                Expression asyncID = getAsyncExpression(tAnnot);

				List<Expression> arg_list = new ArrayList<Expression>();
				arg_list.add(hostVar.clone());
				List<Specifier> specs = new ArrayList<Specifier>(4);
				specs.add(Specifier.VOID);
				specs.add(PointerSpecifier.UNQUALIFIED);
				specs.add(PointerSpecifier.UNQUALIFIED);
				arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
						(Identifier)gpuVar.clone())));
				arg_list.add(new UnaryExpression(UnaryOperator.ADDRESS_OF,pitchID.clone()));
				arg_list.add(
						new BinaryExpression(sizeof_expr.clone(), BinaryOperator.MULTIPLY, lengthList.get(lengthList.size()-1).clone()));
				for( int i=lengthList.size()-2; i>=0; i-- ) {
					arg_list.add(lengthList.get(i).clone());
				}
				arg_list.add(asyncID.clone());
				if( ROSymbol ) {
					arg_list.add(new NameID("HI_MEM_READ_ONLY"));
				} else {
					arg_list.add(new NameID("HI_MEM_READ_WRITE"));
				}
				malloc_call.setArguments(arg_list);
				malloc_stmt = new ExpressionStatement(malloc_call);
				preambleList.add(malloc_stmt);
				// Add malloc size (gpuBytes) statement
				// Ex: gpuBytes=pitch * height * depth;
				Expression biexp = pitchID.clone();
				for( int i=1; i<lengthList.size(); i++ )
				{
					biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
				}
				AssignmentExpression assignex = new AssignmentExpression(cloned_bytes.clone(),AssignmentOperator.NORMAL, 
						biexp);
				gpuBytes_stmt = new ExpressionStatement(assignex);
				preambleList.add(gpuBytes_stmt);
				if( opt_addSafetyCheckingCode ) {
					preambleList.add(gMemAdd_stmt.clone());
				}
			} else if( addNewPitchSymbol ) {
				////////////////////////////////////////////////////////////////////////////////////
				// Pitch symbol is not visible in the current scope; assign a correct pitch value //
				// to the newly added pitch symbol.                                               //
				////////////////////////////////////////////////////////////////////////////////////
				// Ex: pitch = HI_get_pitch(hostVar); //
				///////////////////////////////////////////
				FunctionCall getPitchCall = new FunctionCall(new NameID("HI_get_pitch"));
				getPitchCall.addArgument(hostVar.clone());
				Statement getPitchStmt = new ExpressionStatement(new AssignmentExpression(pitchID.clone(), 
						AssignmentOperator.NORMAL, getPitchCall));
				preambleList.add(getPitchStmt);
			}
			
			///////////////////////////////////////////////
			// Copyin the host data into the GPU memory. //
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// HI_memcpy2D(gpuPtr, dpitch, hostPtr, spitch, widthInBytes, height, HI_MemcpyHostToDevice);                     //
			// HI_memcpy2D_async(gpuPtr, dpitch, hostPtr, spitch, widthInBytes, height, HI_MemcpyHostToDevice, asyncID);      //
			// HI_memcpy3D(gpuPtr, dpitch, hostPtr, spitch, widthInBytes, height, depth HI_MemcpyHostToDevice);               //
			// HI_memcpy3D_async(gpuPtr, dpitch, hostPtr, spitch, widthInBytes, height, depth HI_MemcpyHostToDevic, asyncIDe);//
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if( addCopyInStmt ) {
				FunctionCall copyinCall = null;
				if( asyncExp == null ) {
					if( lengthList.size() == 2 ) { //hostVar is 2D array.
						copyinCall = new FunctionCall(new NameID("HI_memcpy2D"));
					} else {
						copyinCall = new FunctionCall(new NameID("HI_memcpy3D"));
					}
				} else {
					if( lengthList.size() == 2 ) { //hostVar is 2D array.
						copyinCall = new FunctionCall(new NameID("HI_memcpy2D_async"));
					} else {
						copyinCall = new FunctionCall(new NameID("HI_memcpy3D_async"));
					}
				}
				copyinCall.addArgument(gpuVar.clone());
				copyinCall.addArgument(pitchID.clone());
				copyinCall.addArgument(hostVar.clone());
				Expression widthInBytes = new BinaryExpression(sizeof_expr.clone(), BinaryOperator.MULTIPLY, 
						lengthList.get(lengthList.size()-1).clone());
				copyinCall.addArgument(widthInBytes);
				copyinCall.addArgument(widthInBytes.clone());
				for( int i=lengthList.size()-2; i>=0; i-- ) {
					copyinCall.addArgument(lengthList.get(i).clone());
				}
				copyinCall.addArgument(new NameID("HI_MemcpyHostToDevice"));
				if( asyncExp != null ) {
					copyinCall.addArgument(asyncExp.clone());
				}
				copyin_stmt = new ExpressionStatement(copyinCall);
				preambleList.add(copyin_stmt);
				
				//Insert HI_profile_measure_userevent("HICOPYIN: var", (double)copy_size)
				//call for this variable if enableCustomPrfiling option is on.
				if( enableCustomProfiling && (measureCopyInStmt != null) ) {
					preambleList.add(measureCopyInStmt.clone());
				}
				
				//Insert set_status() call for this variable if memtrVerification is used.
				if( memtrVerification ) {
					FunctionCall setStatusCall = new FunctionCall(new NameID("HI_set_status"));
					setStatusCall.addArgument(hostVar.clone());
					setStatusCall.addArgument(new NameID("acc_device_nvidia"));
					setStatusCall.addArgument(new NameID("HI_notstale"));
					setStatusCall.addArgument(new StringLiteral(hostVar.toString()));
					setStatusCall.addArgument(refName.clone());
					if( loopIndex != null ) {
						setStatusCall.addArgument(loopIndex.clone());
					} else {
						setStatusCall.addArgument(new NameID("INT_MIN"));
					}
					preambleList.add(new ExpressionStatement(setStatusCall));
				}
			}
			
			//////////////////////////////////////////////////
			// Copyout the GPU data back to the CPU memory. //
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// HI_memcpy2D(hostPtr, dpitch, gpuPtr, spitch, widthInBytes, height, HI_MemcpyDeviceToHost);                      //
			// HI_memcpy2D_async(hostPtr, dpitch, gpuPtr, dpitch, widthInBytes, height, HI_MemcpyDeviceToHost, asyncID);       //
			// HI_memcpy3D(hostPtr, dpitch, gpuPtr, spitch, widthInBytes, height, depth HI_MemcpyDeviceToHost);                //
			// HI_memcpy3D_async(hostPtr, dpitch, gpuPtr, spitch, widthInBytes, height, depth HI_MemcpyDeviceToHost, asyncIDe);//
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if( addCopyOutStmt ) {
				FunctionCall copyoutCall = null;
				if( asyncExp == null ) {
					if( lengthList.size() == 2 ) { //hostVar is 2D array.
						copyoutCall = new FunctionCall(new NameID("HI_memcpy2D"));
					} else {
						copyoutCall = new FunctionCall(new NameID("HI_memcpy3D"));
					}
				} else {
					if( lengthList.size() == 2 ) { //hostVar is 2D array.
						copyoutCall = new FunctionCall(new NameID("HI_memcpy2D_async"));
					} else {
						copyoutCall = new FunctionCall(new NameID("HI_memcpy3D_async"));
					}
				}
				copyoutCall.addArgument(hostVar.clone());
				Expression widthInBytes = new BinaryExpression(sizeof_expr.clone(), BinaryOperator.MULTIPLY, lengthList.get(lengthList.size()-1).clone());
				copyoutCall.addArgument(widthInBytes);
				copyoutCall.addArgument(gpuVar.clone());
				copyoutCall.addArgument(pitchID.clone());
				copyoutCall.addArgument(widthInBytes.clone());
				for( int i=lengthList.size()-2; i>=0; i-- ) {
					copyoutCall.addArgument(lengthList.get(i).clone());
				}
				copyoutCall.addArgument(new NameID("HI_MemcpyDeviceToHost"));
				if( asyncExp != null ) {
					copyoutCall.addArgument(asyncExp.clone());
				}
				copyout_stmt = new ExpressionStatement(copyoutCall);
				postscriptList.add(copyout_stmt);
				
				//Insert HI_profile_measure_userevent("HICOPYOUT: var", (double)copy_size)
				//call for this variable if enableCustomPrfiling option is on.
				if( enableCustomProfiling && (measureCopyOutStmt != null) ) {
					postscriptList.add(measureCopyOutStmt.clone());
				}
				
				//Insert set_status() call for this variable if memtrVerification is used.
				if( memtrVerification ) {
					FunctionCall setStatusCall = new FunctionCall(new NameID("HI_set_status"));
					setStatusCall.addArgument(hostVar.clone());
					setStatusCall.addArgument(new NameID("acc_device_host"));
					setStatusCall.addArgument(new NameID("HI_notstale"));
					setStatusCall.addArgument(new StringLiteral(hostVar.toString()));
					setStatusCall.addArgument(refName.clone());
					if( loopIndex != null ) {
						setStatusCall.addArgument(loopIndex.clone());
					} else {
						setStatusCall.addArgument(new NameID("INT_MIN"));
					}
					postscriptList.add(new ExpressionStatement(setStatusCall));
				}
				//Generate result-compare codes for this variable.
				if( asyncW1Stmt != null ) {
					TransformTools.genResultCompareCodes(parentProc, asyncW1Stmt, hostVar, phostVar, pitchID, lengthList, clonedspecs,
							dAnnot, marginOfError, true, minCheckValue);
				}
			}
			
			if( genMallocCode ) {
				////////////////////////////
				// Deallocate GPU memory. //
				////////////////////////////
				// HI_free(hostPtr);   //
				////////////////////////////
				FunctionCall free_call = null;
				if( asyncExp == null ) {
					free_call = new FunctionCall(new NameID("HI_free"));
				} else {
					free_call = new FunctionCall(new NameID("HI_free_async"));
				}
				free_call.addArgument(hostVar.clone());

				//Check async condition
                ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "async");
                Expression asyncID = getAsyncExpression(tAnnot);

				free_call.addArgument(asyncID.clone());
				/*if( asyncExp != null ) {
					free_call.addArgument(asyncExp.clone());
				}*/
				free_stmt = new ExpressionStatement(free_call);
				postscriptList.add(free_stmt);
				if( opt_addSafetyCheckingCode ) {
					postscriptList.add(gpuBytes_stmt.clone());
					postscriptList.add(gMemSub_stmt.clone());
				}
				//Insert reset_status(GPU, stale) call for this variable if memtrVerification is used.
				if( memtrVerification ) {
					FunctionCall setStatusCall = new FunctionCall(new NameID("HI_reset_status"));
					setStatusCall.addArgument(hostVar.clone());
					setStatusCall.addArgument(new NameID("acc_device_nvidia"));
					setStatusCall.addArgument(new NameID("HI_stale"));
					if( asyncExp == null ) {
						//setStatusCall.addArgument(new NameID("INT_MIN"));
						setStatusCall.addArgument(new NameID("DEFAULT_QUEUE"));
					} else {
						setStatusCall.addArgument(asyncExp.clone());
					}
					postscriptList.add(new ExpressionStatement(setStatusCall));
				}
			}

		} else if( mallocT == MallocType.NormalMalloc) { //Normal malloc.
			///////////////////////////////////////////////////////
			// Allocate GPU global memory for the host variable. //
			////////////////////////////////////////////////////////////////
			//  gpuBytes=((193536*4)*sizeof (int));                       //
			//  HI_malloc1D( hostPtr, ((void**)(& devPtr), gpuBytes ); //
			//  HI_malloc1D( &hostVar, ((void**)(& devPtr), gpuBytes );//
			///////////////////////////////////////////////////////////////////
			// CAVEAT: the above will work only if hostPtr pointer points to //
			// continuous memory. ( **hostPtr may not work. )                //
			///////////////////////////////////////////////////////////////////
			// Add malloc size (gpuBytes) statement
			// Ex: gpuBytes=sizeof(float)*((2048+2)*(2048+2));
			if( genMallocCode || addCopyInStmt || addCopyOutStmt ) {
				Expression biexp = sizeof_expr.clone();
				for( int i=0; i<lengthList.size(); i++ )
				{
					biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
				}
				AssignmentExpression assignex = new AssignmentExpression(cloned_bytes.clone(),AssignmentOperator.NORMAL, 
						biexp);
				gpuBytes_stmt = new ExpressionStatement(assignex);
			}
			boolean gpuBytesStmtAdded = false;
			if( genMallocCode ) {
				preambleList.add(gpuBytes_stmt);
				gpuBytesStmtAdded = true;
				// Add malloc statement
				// Ex: HI_malloc1D( hostPtr, ((void**)(& devPtr), gpuBytes );  //
				FunctionCall malloc_call = new FunctionCall(new NameID("HI_malloc1D"));
				List<Expression> arg_list = new ArrayList<Expression>();
				if( lengthList.size() == 0 ) { //hostVar is scalar.
					arg_list.add( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone()));
				} else {
					arg_list.add(hostVar.clone());
				}

				//Check async condition
                ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "async");
                Expression asyncID = getAsyncExpression(tAnnot);

				List<Specifier> specs = new ArrayList<Specifier>(5);
				specs.add(Specifier.VOID);
				specs.add(PointerSpecifier.UNQUALIFIED);
				specs.add(PointerSpecifier.UNQUALIFIED);
				arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
						(Identifier)gpuVar.clone())));
				arg_list.add(cloned_bytes.clone());
				arg_list.add(asyncID.clone());
				if( ROSymbol ) {
					arg_list.add(new NameID("HI_MEM_READ_ONLY"));
				} else {
					arg_list.add(new NameID("HI_MEM_READ_WRITE"));
				}
				malloc_call.setArguments(arg_list);
				malloc_stmt = new ExpressionStatement(malloc_call);
				preambleList.add(malloc_stmt);
				if( opt_addSafetyCheckingCode ) {
					preambleList.add(gMemAdd_stmt.clone());
				}
			}
			
			///////////////////////////////////////////////
			// Copyin the host data into the GPU memory. //
			/////////////////////////////////////////////////////////////////////////////////////////
			// HI_memcpy(gpuPtr, hostPtr, gpuBytes, HI_MemcpyHostToDevice, 0);               //
			// HI_memcpy_async(gpuPtr, hostPtr, gpuBytes, HI_MemcpyHostToDevice, 0, asyncID);//
			/////////////////////////////////////////////////////////////////////////////////////////
			if( addCopyInStmt ) {
				if( !gpuBytesStmtAdded ) {
					preambleList.add(gpuBytes_stmt);
					gpuBytesStmtAdded = true;
				}
				FunctionCall copyinCall = null;
				if( asyncExp == null ) {
					copyinCall = new FunctionCall(new NameID("HI_memcpy"));
				} else {
					copyinCall = new FunctionCall(new NameID("HI_memcpy_async"));
				}
				if( partialArrayPassing ) {
					copyinCall.addArgument(new BinaryExpression(gpuVar.clone(), BinaryOperator.ADD,
							startList.get(0).clone()));
				} else {
					copyinCall.addArgument(gpuVar.clone());
				}
				if( lengthList.size() == 0 ) { //hostVar is scalar.
					copyinCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone()));
				} else {
					if( partialArrayPassing ) {
						copyinCall.addArgument(new BinaryExpression(hostVar.clone(), BinaryOperator.ADD,
								startList.get(0).clone()));
					} else {
						copyinCall.addArgument(hostVar.clone());
					}
				}
				copyinCall.addArgument(cloned_bytes.clone());
				copyinCall.addArgument(new NameID("HI_MemcpyHostToDevice"));
				copyinCall.addArgument(new IntegerLiteral(0));
				if( asyncExp != null ) {
					copyinCall.addArgument(asyncExp.clone());
				}
				copyin_stmt = new ExpressionStatement(copyinCall);
				preambleList.add(copyin_stmt);
				
				//Insert HI_profile_measure_userevent("HICOPYIN: var", (double)copy_size)
				//call for this variable if enableCustomPrfiling option is on.
				if( enableCustomProfiling && (measureCopyInStmt != null) ) {
					preambleList.add(measureCopyInStmt.clone());
				}
				
				//Insert set_status() call for this variable if memtrVerification is used.
				if( memtrVerification ) {
					FunctionCall setStatusCall = new FunctionCall(new NameID("HI_set_status"));
					if( lengthList.size() == 0 ) { //hostVar is scalar.
						setStatusCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								hostVar.clone()));
					} else {
						setStatusCall.addArgument(hostVar.clone());
					}
					setStatusCall.addArgument(new NameID("acc_device_nvidia"));
					setStatusCall.addArgument(new NameID("HI_notstale"));
					setStatusCall.addArgument(new StringLiteral(hostVar.toString()));
					setStatusCall.addArgument(refName.clone());
					if( loopIndex != null ) {
						setStatusCall.addArgument(loopIndex.clone());
					} else {
						setStatusCall.addArgument(new NameID("INT_MIN"));
					}
					preambleList.add(new ExpressionStatement(setStatusCall));
				}
			}
			
			//////////////////////////////////////////////////
			// Copyout the GPU data back to the CPU memory. //
			/////////////////////////////////////////////////////////////////////////////////////////
			// gpuBytes=sizeof(float)*((2048+2)*(2048+2));                                         //
			// HI_memcpy(hostPtr, gpuPtr, gpuBytes, HI_MemcpyDeviceToHost, 0);               //
			// HI_memcpy_async(hostPtr, gpuPtr, gpuBytes, HI_MemcpyDeviceToHost, 0, asyncID);//
			/////////////////////////////////////////////////////////////////////////////////////////
			if( addCopyOutStmt ) {
				gpuBytes_stmt2 = gpuBytes_stmt.clone();
				postscriptList.add(gpuBytes_stmt2);
				FunctionCall copyoutCall = null;
				if( asyncExp == null ) {
					copyoutCall = new FunctionCall(new NameID("HI_memcpy"));
				} else {
					if( kernelVerification ) {
						copyoutCall = new FunctionCall(new NameID("HI_memcpy_asyncS"));
					} else {
						copyoutCall = new FunctionCall(new NameID("HI_memcpy_async"));
					}
				}
				if( lengthList.size() == 0 ) { //hostVar is scalar.
					copyoutCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone()));
				} else {
					if( partialArrayPassing ) {
						copyoutCall.addArgument(new BinaryExpression(hostVar.clone(), BinaryOperator.ADD,
								startList.get(0).clone()));
					} else {
						copyoutCall.addArgument(hostVar.clone());
					}
				}
				if( partialArrayPassing ) {
					copyoutCall.addArgument(new BinaryExpression(gpuVar.clone(), BinaryOperator.ADD,
							startList.get(0).clone()));
				} else {
					copyoutCall.addArgument(gpuVar.clone());
				}
				copyoutCall.addArgument(cloned_bytes.clone());
				copyoutCall.addArgument(new NameID("HI_MemcpyDeviceToHost"));
				copyoutCall.addArgument(new IntegerLiteral(0));
				if( asyncExp != null ) {
					copyoutCall.addArgument(asyncExp.clone());
				}
				copyout_stmt = new ExpressionStatement(copyoutCall);
				postscriptList.add(copyout_stmt);
				
				//Insert HI_profile_measure_userevent("HICOPYOUT: var", (double)copy_size)
				//call for this variable if enableCustomPrfiling option is on.
				if( enableCustomProfiling && (measureCopyOutStmt != null) ) {
					postscriptList.add(measureCopyOutStmt.clone());
				}
				
				//Insert set_status() call for this variable if memtrVerification is used.
				if( memtrVerification ) {
					FunctionCall setStatusCall = new FunctionCall(new NameID("HI_set_status"));
					if( lengthList.size() == 0 ) { //hostVar is scalar.
						setStatusCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								hostVar.clone()));
					} else {
						setStatusCall.addArgument(hostVar.clone());
					}
					setStatusCall.addArgument(new NameID("acc_device_host"));
					setStatusCall.addArgument(new NameID("HI_notstale"));
					setStatusCall.addArgument(new StringLiteral(hostVar.toString()));
					setStatusCall.addArgument(refName.clone());
					if( loopIndex != null ) {
						setStatusCall.addArgument(loopIndex.clone());
					} else {
						setStatusCall.addArgument(new NameID("INT_MIN"));
					}
					postscriptList.add(new ExpressionStatement(setStatusCall));
				}
				//Generate result-compare codes for this variable.
				if( asyncW1Stmt != null ) {
					TransformTools.genResultCompareCodes(parentProc, asyncW1Stmt, hostVar, phostVar, pitchID, lengthList, clonedspecs,
							dAnnot, marginOfError, true, minCheckValue);
				}
			}
			
			if( genMallocCode ) {
				////////////////////////////
				// Deallocate GPU memory. //
				////////////////////////////
				// HI_free(hostPtr);   //
				////////////////////////////
				FunctionCall free_call = null;
				if( asyncExp == null ) {
					free_call = new FunctionCall(new NameID("HI_free"));
				} else {
					free_call = new FunctionCall(new NameID("HI_free_async"));
				}
				if( lengthList.size() == 0 ) { //hostVar is scalar.
					free_call.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone()));
				} else {
					free_call.addArgument(hostVar.clone());
				}

				//Check async condition
                ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "async");
                Expression asyncID = getAsyncExpression(tAnnot);

				free_call.addArgument(asyncID.clone());

				/*if( asyncExp != null ) {
					free_call.addArgument(asyncExp.clone());
				}*/
				free_stmt = new ExpressionStatement(free_call);
				postscriptList.add(free_stmt);
				if( opt_addSafetyCheckingCode ) {
					postscriptList.add(gpuBytes_stmt.clone());
					postscriptList.add(gMemSub_stmt.clone());
				}
				//Insert reset_status(GPU, stale) call for this variable if memtrVerification is used.
				if( memtrVerification ) {
					FunctionCall setStatusCall = new FunctionCall(new NameID("HI_reset_status"));
					if( lengthList.size() == 0 ) { //hostVar is scalar.
						setStatusCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								hostVar.clone()));
					} else {
						setStatusCall.addArgument(hostVar.clone());
					}
					setStatusCall.addArgument(new NameID("acc_device_nvidia"));
					setStatusCall.addArgument(new NameID("HI_stale"));
					if( asyncExp == null ) {
						//setStatusCall.addArgument(new NameID("INT_MIN"));
						setStatusCall.addArgument(new NameID("DEFAULT_QUEUE"));
					} else {
						setStatusCall.addArgument(asyncExp.clone());
					}
					postscriptList.add(new ExpressionStatement(setStatusCall));
				}
			}
		}
		
		if( !genMallocCode ) {
			/////////////////////////////////////////////////////////////////////////////////
			//  Add error-exit code for "present" data clause or "update" directive        //
			/////////////////////////////////////////////////////////////////////////////////
			//     printf("GPU memory for the host variable, hostVar, does not exit. \n"); //
			//     printf("Enclosing Translation Unit: filename\n");                       // 
			//     printf("Enclosing annotation:\n ACCAnnotation \n");                     //
			//     exit(1);                                                                //
			/////////////////////////////////////////////////////////////////////////////////
			presentErrorCode = new CompoundStatement();
			FunctionCall printfCall = new FunctionCall(new NameID("printf"));
			printfCall.addArgument(new StringLiteral("[ERROR] GPU memory for the host variable, "+hostVar.toString()+
					", does not exist. \\n"));
			presentErrorCode.addStatement(new ExpressionStatement(printfCall));
			printfCall = new FunctionCall(new NameID("printf"));
			printfCall.addArgument(new StringLiteral("Enclosing Translation Unit: " + parentTrUnt.getInputFilename() + "\\n"));
			printfCall = new FunctionCall(new NameID("printf"));
			printfCall.addArgument(new StringLiteral("Enclosing annotation: \\n" + dAnnot.toString() + " \\n"));
			presentErrorCode.addStatement(new ExpressionStatement(printfCall));
			FunctionCall exitCall = new FunctionCall(new NameID("exit"));
			exitCall.addArgument(new IntegerLiteral(1));
			presentErrorCode.addStatement(new ExpressionStatement(exitCall));
		}
		
		////////////////////////////////////////////////////////////////////////////////////////////////
		// Code to check whether GPU memory is present or not.                                        //
		// if( HI_get_device_address(hostPtr, ((void **)&devicePtr))  != HI_success ) { ..... } //
		// if( HI_getninc_prtcounter(hostPtr, ((void **)&devicePtr))  == 0 ) { ..... }             //
		// if( HI_decnget_prtcounter(hostPtr, ((void **)&devicePtr))  == 0 ) { ..... }             //
		////////////////////////////////////////////////////////////////////////////////////////////////
		Expression presentCheck_exp = null;
		Expression presentCheckS_exp = null;
		Expression presentCheckE_exp = null;
		FunctionCall presentCheck_call = new FunctionCall(new NameID("HI_get_device_address"));
		FunctionCall presentCheckS_call = new FunctionCall(new NameID("HI_getninc_prtcounter"));
		FunctionCall presentCheckE_call = new FunctionCall(new NameID("HI_decnget_prtcounter"));
		if( lengthList.size() == 0 ) { //hostVar is scalar.
			presentCheck_call.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
					hostVar.clone()));
			presentCheckS_call.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
					hostVar.clone()));
			presentCheckE_call.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
					hostVar.clone()));
		} else {
			presentCheck_call.addArgument(hostVar.clone());
			presentCheckS_call.addArgument(hostVar.clone());
			presentCheckE_call.addArgument(hostVar.clone());
		}
		List<Specifier> specs = new ArrayList<Specifier>(4);
		specs.add(Specifier.VOID);
		specs.add(PointerSpecifier.UNQUALIFIED);
		specs.add(PointerSpecifier.UNQUALIFIED);
		presentCheck_call.addArgument(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
				(Identifier)gpuVar.clone())));
		presentCheck_exp = new BinaryExpression(presentCheck_call, BinaryOperator.COMPARE_NE, new NameID("HI_success"));
		specs = new ArrayList<Specifier>(4);
		specs.add(Specifier.VOID);
		specs.add(PointerSpecifier.UNQUALIFIED);
		specs.add(PointerSpecifier.UNQUALIFIED);
		presentCheckS_call.addArgument(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
				(Identifier)gpuVar.clone())));
		presentCheckS_exp = new BinaryExpression(presentCheckS_call, BinaryOperator.COMPARE_EQ, new IntegerLiteral(0));
		specs = new ArrayList<Specifier>(4);
		specs.add(Specifier.VOID);
		specs.add(PointerSpecifier.UNQUALIFIED);
		specs.add(PointerSpecifier.UNQUALIFIED);
		presentCheckE_call.addArgument(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
				(Identifier)gpuVar.clone())));
		presentCheckE_exp = new BinaryExpression(presentCheckE_call, BinaryOperator.COMPARE_EQ, new IntegerLiteral(0));

		//Check async condition
        ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "async");
        Expression asyncID = getAsyncExpression(tAnnot);

		presentCheck_call.addArgument(asyncID.clone());
		presentCheckS_call.addArgument(asyncID.clone());
		presentCheckE_call.addArgument(asyncID.clone());

		//[DEBUG] To allow separate compilation, GPU data presence check should be added to all update directives.
		//if( (dataClauseT == DataClauseType.UpdateOnly) && addNewGpuSymbol && (mallocT != MallocType.ConstantMalloc) ) {
		if( (dataClauseT == DataClauseType.UpdateOnly) && (mallocT != MallocType.ConstantMalloc) ) {
			IfStatement ifStmt = new IfStatement(presentCheck_exp.clone(), presentErrorCode.clone());
			if( memtrT == MemTrType.CopyIn ) {
				preambleList.add(0, ifStmt);
			}
			if( memtrT == MemTrType.CopyOut ) {
				postscriptList.add(0, ifStmt);
			}
			
		}
		
		if( dataClauseT == DataClauseType.CheckOnly ) {
			if( mallocT != MallocType.ConstantMalloc ) {
				Statement inPt = inStmts.get(0);
				CompoundStatement pStmt = (CompoundStatement)inPt.getParent();
				CompoundStatement elseBody = null;
				if( preambleList.size() > 0 ) {
					elseBody = new CompoundStatement();
					for( int k=0; k<preambleList.size(); k++ ) {
						elseBody.addStatement(preambleList.get(k));
					}
				}
				IfStatement pIfStmt = null;
				if( elseBody != null ) {
					pIfStmt = new IfStatement(presentCheck_exp.clone(), presentErrorCode, elseBody);
				} else {
					pIfStmt = new IfStatement(presentCheck_exp.clone(), presentErrorCode);
				}
				if( dRegionType == DataRegionType.ImplicitProgramRegion ) {
					pStmt.addStatementAfter(inPt, pIfStmt);
					for( int i=1; i<inStmts.size(); i++ ) {
						TranslationUnit tUnt = null;
						inPt = inStmts.get(i);
						pStmt = (CompoundStatement)inPt.getParent();
						tt = inPt;
						while( (tt != null) ) {
							if( tt instanceof TranslationUnit ) {
								tUnt = (TranslationUnit)tt;
								break;
							}
							tt = tt.getParent();
						}
						if( tUnt != null ) {
							VariableDeclaration tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, gpuVar);
							if( tDecl != null ) {
								gpuVar = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
							}
							if( hostVar instanceof IDExpression ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, (IDExpression)hostVar);
								if( tDecl != null ) {
									hostVar = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							} else {
								//FIXME: we have to handle the case where hostVar is an AccessExpression.
								//For now, not handling will cause inconsistency in the output IR, but not in the
								//output CUDA code.
							}
							if( mallocT == MallocType.TextureMalloc ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, textureRefID);
								if( tDecl != null ) {
									textureRefID = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							}
							if( mallocT == MallocType.PitchedMalloc ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, pitchID);
								if( tDecl != null ) {
									pitchID = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							}
							if( mallocT == MallocType.ConstantMalloc ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, constantID);
								if( tDecl != null ) {
									constantID = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							}
							pIfStmt = pIfStmt.clone();
							TransformTools.replaceAll(pIfStmt, gpuVar, gpuVar);
							TransformTools.replaceAll(pIfStmt, hostVar, hostVar);
							if( textureRefID != null ) {
								TransformTools.replaceAll(pIfStmt, textureRefID, textureRefID);
							}
							if( pitchID != null ) {
								TransformTools.replaceAll(pIfStmt, pitchID, pitchID);
							}
							pStmt.addStatementAfter(inPt, pIfStmt);
						}
					}
				} else {
					if( ifCondExp == null ) {
						pStmt.addStatementBefore(inPt, pIfStmt);
					} else {
						IfStatement IfStmt = new IfStatement(ifCondExp.clone(), pIfStmt);
						pStmt.addStatementBefore(inPt, IfStmt);
					}
				}
			}
		} else {
			////////////////////////////////////////////////////////////////////
			// Insert malloc/free/memcopy statements into target code region. //
			////////////////////////////////////////////////////////////////////
			Statement inPt = inStmts.get(0);
			CompoundStatement pStmt = (CompoundStatement)inPt.getParent();
			if( dRegionType == DataRegionType.ImplicitProgramRegion ) {
				if( preambleList.size() > 0 ) {
					if( checkPresent ) {
						CompoundStatement cpStmt = new CompoundStatement();
						for( int k=0; k<preambleList.size(); k++ ) {
							cpStmt.addStatement(preambleList.get(k));
						}
						IfStatement cpIfStmt = new IfStatement(presentCheckS_exp.clone(), cpStmt);
						pStmt.addStatementAfter(inPt, cpIfStmt);
					} else {
						for( int k=preambleList.size()-1; k>=0; k-- ) {
							pStmt.addStatementAfter(inPt, preambleList.get(k));
						}
					}
					for( int i=1; i<inStmts.size(); i++ ) {
						TranslationUnit tUnt = null;
						inPt = inStmts.get(i);
						pStmt = (CompoundStatement)inPt.getParent();
						tt = inPt;
						while( (tt != null) ) {
							if( tt instanceof TranslationUnit ) {
								tUnt = (TranslationUnit)tt;
								break;
							}
							tt = tt.getParent();
						}
						if( tUnt != null ) {
							VariableDeclaration tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, gpuVar);
							if( tDecl != null ) {
								gpuVar = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
							}
							if( mallocT == MallocType.TextureMalloc ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, textureRefID);
								if( tDecl != null ) {
									textureRefID = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							}
							if( mallocT == MallocType.PitchedMalloc ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, pitchID);
								if( tDecl != null ) {
									pitchID = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							}
							if( mallocT == MallocType.ConstantMalloc ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, constantID);
								if( tDecl != null ) {
									constantID = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							}
							tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, cloned_bytes);
							if( tDecl != null ) {
								cloned_bytes = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
							}
							if( hostVar instanceof IDExpression ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, (IDExpression)hostVar);
								if( tDecl != null ) {
									hostVar = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							} else {
								//FIXME: we have to handle the case where hostVar is an AccessExpression.
								//For now, not handling will cause inconsistency in the output IR, but not in the
								//output CUDA code.
							}

							if( checkPresent ) {
								presentCheckS_exp = presentCheckS_exp.clone();
								TransformTools.replaceAll(presentCheckS_exp, gpuVar, gpuVar);
								TransformTools.replaceAll(presentCheckS_exp, hostVar, hostVar);
								CompoundStatement cpStmt = new CompoundStatement();
								for( int k=0; k<preambleList.size(); k++ ) {
									Statement stmt = preambleList.get(k).clone();
									if( gpuVar != null ) {
										TransformTools.replaceAll(stmt, gpuVar, gpuVar);
									}
									if( hostVar != null ) {
										TransformTools.replaceAll(stmt, hostVar, hostVar);
									}
									if( cloned_bytes != null ) {
										TransformTools.replaceAll(stmt, cloned_bytes, cloned_bytes);
									}
									if( pitchID != null ) {
										TransformTools.replaceAll(stmt, pitchID, pitchID);
									}
									if( textureRefID != null ) {
										TransformTools.replaceAll(stmt, textureRefID, textureRefID);
									}
									if( constantID != null ) {
										TransformTools.replaceAll(stmt, constantID, constantID);
									}
									cpStmt.addStatement(stmt);
								}
								IfStatement cpIfStmt = new IfStatement(presentCheckS_exp, cpStmt);
								pStmt.addStatementAfter(inPt, cpIfStmt);
							} else {
								for( int k=preambleList.size()-1; k>=0; k-- ) {
									Statement stmt = preambleList.get(k).clone();
									if( gpuVar != null ) {
										TransformTools.replaceAll(stmt, gpuVar, gpuVar);
									}
									if( hostVar != null ) {
										TransformTools.replaceAll(stmt, hostVar, hostVar);
									}
									if( cloned_bytes != null ) {
										TransformTools.replaceAll(stmt, cloned_bytes, cloned_bytes);
									}
									if( pitchID != null ) {
										TransformTools.replaceAll(stmt, pitchID, pitchID);
									}
									if( textureRefID != null ) {
										TransformTools.replaceAll(stmt, textureRefID, textureRefID);
									}
									if( constantID != null ) {
										TransformTools.replaceAll(stmt, constantID, constantID);
									}
									pStmt.addStatementAfter(inPt, stmt);
								}
							}
						}
					}
				}

				if( postscriptList.size() > 0 ) {
					Statement outPt = outStmts.get(0);
					pStmt = (CompoundStatement)outPt.getParent();
					if( checkPresent ) {
						CompoundStatement cpStmt = new CompoundStatement();
						for( int k=0; k<postscriptList.size(); k++ ) {
							cpStmt.addStatement(postscriptList.get(k));
						}
						IfStatement cpIfStmt = new IfStatement(presentCheckE_exp.clone(), cpStmt);
						pStmt.addStatementBefore(outPt, cpIfStmt);
					} else {
						for( int k=0; k<postscriptList.size(); k++ ) {
							pStmt.addStatementBefore(outPt, postscriptList.get(k));
						}
					}
					for( int i=1; i<outStmts.size(); i++ ) {
						TranslationUnit tUnt = null;
						outPt = outStmts.get(i);
						pStmt = (CompoundStatement)outPt.getParent();
						tt = outPt;
						while( (tt != null) ) {
							if( tt instanceof TranslationUnit ) {
								tUnt = (TranslationUnit)tt;
								break;
							}
							tt = tt.getParent();
						}
						if( tUnt != null ) {
							VariableDeclaration tDecl = null;
							if( mallocT == MallocType.TextureMalloc ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, textureRefID);
								if( tDecl != null ) {
									textureRefID = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							}
							if( addCopyOutStmt ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, gpuVar);
								if( tDecl != null ) {
									gpuVar = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
								if( mallocT == MallocType.PitchedMalloc ) {
									tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, pitchID);
									if( tDecl != null ) {
										pitchID = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
									}
								}
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, cloned_bytes);
								if( tDecl != null ) {
									cloned_bytes = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							}
							if( hostVar instanceof IDExpression ) {
								tDecl = (VariableDeclaration)SymbolTools.findSymbol(tUnt, (IDExpression)hostVar);
								if( tDecl != null ) {
									hostVar = new Identifier((VariableDeclarator)tDecl.getDeclarator(0));
								}
							} else {
								//FIXME: we have to handle the case where hostVar is an AccessExpression.
								//For now, not handling will cause inconsistency in the output IR, but not in the
								//output CUDA code.
							}

							if( checkPresent && (memtrT != MemTrType.CopyInOut) ) {
								presentCheckE_exp = presentCheckE_exp.clone();
								TransformTools.replaceAll(presentCheckE_exp, gpuVar, gpuVar);
								TransformTools.replaceAll(presentCheckE_exp, hostVar, hostVar);
								CompoundStatement cpStmt = new CompoundStatement();
								for( int k=0; k<postscriptList.size(); k++ ) {
									Statement stmt = postscriptList.get(k).clone();
									if( gpuVar != null ) {
										TransformTools.replaceAll(stmt, gpuVar, gpuVar);
									}
									if( hostVar != null ) {
										TransformTools.replaceAll(stmt, hostVar, hostVar);
									}
									if( cloned_bytes != null ) {
										TransformTools.replaceAll(stmt, cloned_bytes, cloned_bytes);
									}
									if( pitchID != null ) {
										TransformTools.replaceAll(stmt, pitchID, pitchID);
									}
									if( textureRefID != null ) {
										TransformTools.replaceAll(stmt, textureRefID, textureRefID);
									}
									cpStmt.addStatement(stmt);
								}
								IfStatement cpIfStmt = new IfStatement(presentCheckE_exp, cpStmt);
								pStmt.addStatementBefore(outPt, cpIfStmt);
							} else {
								for( int k=0; k<postscriptList.size(); k++ ) {
									Statement stmt = postscriptList.get(k).clone();
									if( gpuVar != null ) {
										TransformTools.replaceAll(stmt, gpuVar, gpuVar);
									}
									if( hostVar != null ) {
										TransformTools.replaceAll(stmt, hostVar, hostVar);
									}
									if( cloned_bytes != null ) {
										TransformTools.replaceAll(stmt, cloned_bytes, cloned_bytes);
									}
									if( pitchID != null ) {
										TransformTools.replaceAll(stmt, pitchID, pitchID);
									}
									if( textureRefID != null ) {
										TransformTools.replaceAll(stmt, textureRefID, textureRefID);
									}
									pStmt.addStatementBefore(outPt, stmt);
								}
							}
						}
					}
				}
			} else { //not implicit-program-level data region
				if( preambleList.size() > 0 ) {
					if( ifCondExp == null ) {
						if( checkPresent ) {
							CompoundStatement cpStmt = new CompoundStatement();
							for( int k=0; k<preambleList.size(); k++ ) {
								cpStmt.addStatement(preambleList.get(k));
							}
							IfStatement cpIfStmt;
							cpIfStmt = new IfStatement(presentCheckS_exp.clone(), cpStmt);
							pStmt.addStatementBefore(inPt, cpIfStmt);

						} else {
							for( int k=0; k<preambleList.size(); k++ ) {
								pStmt.addStatementBefore(inPt, preambleList.get(k));
							}
						}
					} else {
						CompoundStatement cBody = new CompoundStatement();
						for( int k=0; k<preambleList.size(); k++ ) {
							cBody.addStatement(preambleList.get(k));
						}
						Expression ifCondExp2 = ifCondExp.clone();
						if( checkPresent ) {
							ifCondExp2 = new BinaryExpression(ifCondExp.clone(), BinaryOperator.LOGICAL_AND, presentCheckS_exp.clone());
						}
						IfStatement ifStmt = new IfStatement(ifCondExp2, cBody);
						pStmt.addStatementBefore(inPt, ifStmt);
					}
				}
				
				if( postscriptList.size() > 0 ) {
					//Handle free/copyout statements.
					if( dRegionType == DataRegionType.ImplicitProcedureRegion ) {
						if( checkPresent ) {
							CompoundStatement cpStmt = new CompoundStatement();
							if( outStmts.isEmpty() ) {
								CompoundStatement pBody = parentProc.getBody();
								for( int k=0; k<postscriptList.size(); k++ ) {
									cpStmt.addStatement(postscriptList.get(k));
								}
								IfStatement cpIfStmt;
								cpIfStmt = new IfStatement(presentCheckE_exp.clone(), cpStmt);
								pBody.addStatement(cpIfStmt);
							} else {
								for( Statement outPt : outStmts ) {
									pStmt = (CompoundStatement)outPt.getParent();
									for( int k=0; k<postscriptList.size(); k++ ) {
										cpStmt.addStatement(postscriptList.get(k).clone());
									}
									IfStatement cpIfStmt;
									if( dataClauseT == DataClauseType.UpdateOnly ) {
										cpIfStmt = new IfStatement(presentCheck_exp.clone(), cpStmt);
									} else {
										cpIfStmt = new IfStatement(presentCheckE_exp.clone(), cpStmt);
									}
									pStmt.addStatementBefore(outPt, cpIfStmt);
								}
							}
						} else {
							if( outStmts.isEmpty() ) {
								CompoundStatement pBody = parentProc.getBody();
								for( int k=0; k<postscriptList.size(); k++ ) {
									pBody.addStatement(postscriptList.get(k));
								}
							} else {
								for( Statement outPt : outStmts ) {
									pStmt = (CompoundStatement)outPt.getParent();
									for( int k=0; k<postscriptList.size(); k++ ) {
										pStmt.addStatementBefore(outPt, postscriptList.get(k).clone());
									}
								}
							}
						}
					} else { //explicit data region or compute region
						if( ifCondExp == null ) {
							if( checkPresent ) {
								CompoundStatement cpStmt = new CompoundStatement();
								for( int k=0; k<postscriptList.size(); k++ ) {
									cpStmt.addStatement(postscriptList.get(k));
								}
								IfStatement cpIfStmt;
								cpIfStmt = new IfStatement(presentCheckE_exp.clone(), cpStmt);
								pStmt.addStatementAfter(inPt, cpIfStmt);
							} else {
								for( int k=postscriptList.size()-1; k>=0; k-- ) {
									pStmt.addStatementAfter(inPt, postscriptList.get(k));
								}
							}
						} else {
							CompoundStatement cBody = new CompoundStatement();
							for( int k=0; k<postscriptList.size(); k++ ) {
								cBody.addStatement(postscriptList.get(k));
							}
							Expression ifCondExp2 = ifCondExp.clone();
							if( checkPresent ) {
								ifCondExp2 = new BinaryExpression(ifCondExp.clone(), BinaryOperator.LOGICAL_AND, presentCheckE_exp.clone());
							}
							IfStatement ifStmt = new IfStatement(ifCondExp2, cBody);
							pStmt.addStatementAfter(inPt, ifStmt);
						}
						//Insert kernel-result-compare codes right after HI_waitS1() call statement.
/*						if( (asyncW1Stmt != null) && !kernelVerifyCodes.isEmpty() ) {
							CompoundStatement cAStmt = (CompoundStatement)asyncW1Stmt.getParent();
							for( int k=kernelVerifyCodes.size()-1; k>=0; k-- ) {
								cAStmt.addStatementAfter(asyncW1Stmt, kernelVerifyCodes.get(k));
							}
						}*/
					}
				}
			}
		}
	}
	
	
	protected void handleUpdateClauses(ACCAnnotation uAnnot, boolean IRSymbolOnly) {
		Annotatable at = uAnnot.getAnnotatable();
		//DEBUG: This implementation assumes that a CUDA clause exists at most once per an annotatable object.
		Set<Symbol> constantSet = new HashSet<Symbol>();
		Set<Symbol> textureSet = new HashSet<Symbol>();
		Set<Symbol> sharedROSet = new HashSet<Symbol>();
		Set<Symbol> ROSymSet = new HashSet<Symbol>();
		ARCAnnotation tCAnnot = at.getAnnotation(ARCAnnotation.class, "constant");
		Set<SubArray> dataSet;
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("constant");
			constantSet.addAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "noconstant");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("noconstant");
			constantSet.removeAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "texture");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("texture");
			textureSet.addAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "notexture");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("notexture");
			textureSet.removeAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "sharedRO");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("sharedRO");
			sharedROSet.addAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		tCAnnot = at.getAnnotation(ARCAnnotation.class, "noshared");
		if( tCAnnot != null ) {
			dataSet = (Set<SubArray>)tCAnnot.get("noshared");
			sharedROSet.removeAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
		}
		ACCAnnotation ROAnnot = at.getAnnotation(ACCAnnotation.class, "accreadonly");
		if( ROAnnot != null ) {
			ROSymSet.addAll((Set<Symbol>)ROAnnot.get("accreadonly"));
		}
		//Check if condition
		Expression ifCond = null;
		ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "if");
		if( tAnnot != null ) {
			ifCond = (Expression)tAnnot.get("if");
			ifCond = Symbolic.simplify(ifCond);
			if( ifCond instanceof IntegerLiteral ) {
				if( ((IntegerLiteral)ifCond).getValue() != 0 ) {
					ifCond = null; //Compiler knows that this region will be executed; ignore the if-condition.
				} else { //compiler knows that this region will not be outlined as a GPU kernel; skip conversion.
					return;
				}
			}
		}
		//Check async condition
		Expression asyncID = null;
		tAnnot = at.getAnnotation(ACCAnnotation.class, "async");
		if( tAnnot != null ) {
			Object obj = tAnnot.get("async");
			if( obj instanceof String ) {
				//asyncID = new NameID("INT_MAX");
                asyncID = new NameID("acc_async_noval");
			} else if( obj instanceof Expression ) {
				asyncID = (Expression)obj;
			}
		}
		
		Statement profileRegion = null;
		if( enableCustomProfiling ) {
			ARCAnnotation profAnnot = at.getAnnotation(ARCAnnotation.class, "profile");
			if( profAnnot != null ) {
				Set<String> modeSet = profAnnot.get("mode");
				if( (modeSet != null) && (modeSet.contains(CustomProfilingTransformation.mode_all) || 
						modeSet.contains(CustomProfilingTransformation.mode_memorytransfer)) ) {
					//Find a parent profile region containing track directive 
					//If no parent profile region is found, use the current profile region.
					Traversable tt = at.getParent();
					while ((tt != null) && (tt instanceof Annotatable)) {
						Annotatable atObj = (Annotatable)tt;
						tCAnnot = atObj.getAnnotation(ARCAnnotation.class, "track");
						if( tCAnnot != null ) {
							profileRegion = (Statement)atObj;
							break;
						}
						tt = tt.getParent();
					}
					if( profileRegion == null ) {
						profileRegion = (Statement)at;
					}
				}
			}
		}
		
		List<Statement> inStmts = new LinkedList<Statement>();
		List<Statement> outStmts = new LinkedList<Statement>();
		inStmts.add((AnnotationStatement)at);
		outStmts.add((AnnotationStatement)at);
		DataClauseType dataClauseT = DataClauseType.UpdateOnly;
		DataRegionType regionT = DataRegionType.ExplicitDataRegion; //Update directive is not a data region, but this works well with 
		                                                            //genCUDACodesForDataClause().
		for( String key: uAnnot.keySet() ) {
			MemTrType memtrT = MemTrType.NoCopy;
			MallocType mallocT = MallocType.NormalMalloc;
			if( key.equals("host") || key.equals("self") ) {
				memtrT = MemTrType.CopyOut;
			} else if( key.equals("device") ) {
				memtrT = MemTrType.CopyIn;
			}
			if( memtrT != MemTrType.NoCopy ) {
				Object value = uAnnot.get(key);
				boolean isFirstData = true;
				if( value instanceof Set ) {
					Collection vSet = AnalysisTools.getSortedCollection((Set)value);
					for( Object elm : vSet ) {
						if( elm instanceof SubArray ) {
							SubArray sArray = (SubArray)elm;
							Expression varName = sArray.getArrayName();
							Symbol sym = SymbolTools.getSymbolOf(varName);
							List<Expression> startList = new LinkedList<Expression>();
							List<Expression> lengthList = new LinkedList<Expression>();
							boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, at);
							if( !foundDimensions ) {
								Tools.exit("[ERROR in ACC2CUDATranslator.handleUpdateClauses()] Dimension information " +
										"of the following variable is" +
										"unknown: " + sArray.getArrayName() + ", OpenACC directive: " + uAnnot +
								"; the ACC2GPU translation failed!");
							}
							List<Specifier> typeSpecs = new ArrayList<Specifier>();
							Symbol IRSym = sym;
							if( IRSym instanceof PseudoSymbol ) {
								IRSym = ((PseudoSymbol)IRSym).getIRSymbol();
							}
							if( IRSymbolOnly ) {
								sym = IRSym;
								varName = new Identifier(sym);
								typeSpecs.addAll(((VariableDeclaration)sym.getDeclaration()).getSpecifiers());
							} else {
								Symbol tSym = sym;
								while( tSym instanceof AccessSymbol ) {
									tSym = ((AccessSymbol)tSym).getMemberSymbol();
								}
								typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
							}
							int dimension = lengthList.size();
							mallocT = MallocType.NormalMalloc;
							if( constantSet.contains(sym) ) {
								mallocT = MallocType.ConstantMalloc;
							} else if( textureSet.contains(sym) ) {
								if( dimension == 1 ) {
									if( sym.getTypeSpecifiers().contains(Specifier.DOUBLE) ) {
										PrintTools.println("\n[WARNING] CUDA does not support double textures; " +
												"textureing of the following double variable will be skipped!\n" + 
												"Variable: " + varName + ", ACCAnnotation: " + uAnnot + "\n", 0);
									} else {
										mallocT = MallocType.TextureMalloc;
									}
								} else {
									PrintTools.println("\n[WARNING] current implementation supports only 1D texture reference " +
											"bound to linear memory; textureing of the following variable will be skipped!\n" + 
											"Variable: " + varName + ", ACCAnnotation: " + uAnnot + "\n", 0);
								}
							} else if( opt_MallocPitch ) {
								//FIXME: for now, MallocPitch is allowed only to 2D array
								//if( (dimension == 2) || (dimension == 3) ) 
								if( dimension == 2 ) {
									mallocT = MallocType.PitchedMalloc;
								}
							}

							//Step1: find symboltable containing current host variable.
							//Step2: find GPU device variable for the host variable.
							//       If not existing, 
							//           - If host variable is global, check mainTrUnt.
							//               - If mainTrUnt does not have, error. Otherwise, create extern copy.
							//           - If host variable is local, error.
							//Step3: create and insert memory transfer code.
							boolean ROSymbol = false;
							if( ROSymSet.contains(IRSym) ) {
								ROSymbol = true;
							}
							genCUDACodesForDataClause(uAnnot, IRSym, varName, startList, lengthList, typeSpecs, 
									ifCond, asyncID, dataClauseT, mallocT, memtrT, regionT, inStmts, outStmts, null, 
									profileRegion, isFirstData, ROSymbol);
							isFirstData = false;
						} else {
							break;
						}
					}
				}
				
			}
		}
		
	}
	
	/**
	 * 
	 */
	protected void extractComputeRegion(Procedure cProc, ACCAnnotation cAnnot, String cRegionKind, String new_func_name,
			boolean IRSymbolOnly) {
		PrintTools.println("[extractComputeRegion() begins] current Procedure: " + cProc.getSymbolName()
				+ "\nOpenACC annotation: " + cAnnot +"\n", 1);
		Statement region = (Statement)cAnnot.getAnnotatable();
		SymbolTable global_table = (SymbolTable) cProc.getParent();
		TranslationUnit parentTrUnt = (TranslationUnit)cProc.getParent();

		//////////////////////////////////////////////////////////////////
		// Extract internal directives attached to this compute region. //
		//////////////////////////////////////////////////////////////////
		HashSet<Symbol> accsharedSet = new HashSet<Symbol>();
		HashSet<Symbol> accreadonlySet = new HashSet<Symbol>();
		//HashSet<Symbol> accprivateSet = new HashSet<Symbol>();
		//HashSet<Symbol> rcreateSet = new HashSet<Symbol>();
		HashSet<Symbol> accreductionSet = new HashSet<Symbol>();
		////////////////////////////////////////////////////////////////
		// Create a mapping between a shared symbol and its subarray. //
		////////////////////////////////////////////////////////////////
		Map<Symbol, SubArray> accSharedMap = new HashMap<Symbol, SubArray>();
		Set<Symbol> accDevicePtrSet = new HashSet<Symbol>();
		
		
		//////////////////////////////////////////////////////////////
		// Extract CUDA directives attached to this compute region. //
		//////////////////////////////////////////////////////////////
		Set<Symbol> cudaRegisterROSet = new HashSet<Symbol>();
		Set<Symbol> cudaRegisterSet = new HashSet<Symbol>();
		Set<Symbol> cudaNoRegisterSet = new HashSet<Symbol>();
		Set<Symbol> cudaSharedROSet = new HashSet<Symbol>();
		Set<Symbol> cudaSharedSet = new HashSet<Symbol>();
		Set<Symbol> cudaNoSharedSet = new HashSet<Symbol>();
		Set<Symbol> cudaTextureSet = new HashSet<Symbol>();
		Set<Symbol> cudaNoTextureSet = new HashSet<Symbol>();
		Set<Symbol> cudaConstantSet = new HashSet<Symbol>();
		Set<Symbol> cudaNoConstantSet = new HashSet<Symbol>();
		//Set<Symbol> cudaNoRedUnrollSet = new HashSet<Symbol>();
		Map<Symbol, Set<SubArray>> shrdArryOnRegMap = new HashMap<Symbol, Set<SubArray>>();
		Set<SubArray> ROShrdArryOnRegSet = new HashSet<SubArray>();
		
		List<ACCAnnotation> atomicList = IRTools.collectPragmas(cProc.getBody(), ACCAnnotation.class, "atomic_var");
		for(ACCAnnotation atomicAnnot: atomicList)
		{
			HashSet<Identifier> idSet = (HashSet<Identifier>)atomicAnnot.get("atomic_var");
			if(idSet != null)
			{
				for(Identifier id : idSet)
				{
					cudaNoRegisterSet.add(id.getSymbol());
					cudaNoSharedSet.add(id.getSymbol());
				}
				atomicAnnot.remove("atomic_var");
			}
		}

		boolean noloopcollapse = false;
		Declaration tLastACCDecl = OpenACCHeaderEndMap.get(parentTrUnt);
		
		List<ACCAnnotation> accAnnots = region.getAnnotations(ACCAnnotation.class);
		if( accAnnots != null ) {
			for( ACCAnnotation cannot : accAnnots ) {
				Set<Symbol> symSet = (Set<Symbol>)cannot.get("accshared");
				if( symSet != null ) {
					accsharedSet.addAll(symSet);
				}
				symSet = (Set<Symbol>)cannot.get("accreadonly");
				if( symSet != null ) {
					accreadonlySet.addAll(symSet);
				}
/*
				symSet = (Set<Symbol>)cannot.get("accprivate");
				if( symSet != null ) {
					accprivateSet.addAll(symSet);
				}
				symSet = (Set<Symbol>)cannot.get("rcreate");
				if( symSet != null ) {
					rcreateSet.addAll(symSet);
				}
*/				
				symSet = (Set<Symbol>)cannot.get("accreduction");
				if( symSet != null ) {
					accreductionSet.addAll(symSet);
				}
				//Create shared symbol to its subarray mapping.
				for( String dataClause : ACCAnnotation.dataClauses ) {
					HashSet<SubArray> dataSet = (HashSet<SubArray>)cannot.get(dataClause);
					if( dataSet != null ) {
						for( SubArray sArray : dataSet ) {
							Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
							if( dataClause.equals("deviceptr") ) {
								accDevicePtrSet.add(sym);
							} 
							accSharedMap.put(sym, sArray);
						}
					}
				}				
			}
		}
		
		List<ARCAnnotation> arcAnnots = region.getAnnotations(ARCAnnotation.class);
		if( arcAnnots != null ) {
			for( ARCAnnotation cannot : arcAnnots ) {
				HashSet<SubArray> dataSet = (HashSet<SubArray>)cannot.get("registerRO");
				if( dataSet != null ) {
					for( SubArray sArray : dataSet ) {
						Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
						Symbol tSym = sym;
						if( sym instanceof AccessSymbol ) {
							tSym = ((AccessSymbol)sym).getMemberSymbol();
						}
						if( SymbolTools.isArray(tSym) || SymbolTools.isPointer(tSym) ) {
							Set<SubArray> sSet = null;
							if( shrdArryOnRegMap.containsKey(tSym) ) {
								sSet = shrdArryOnRegMap.get(tSym);
							} else {
								sSet = new HashSet<SubArray>();
								shrdArryOnRegMap.put(tSym, sSet);
							}
							sSet.add(sArray);
							ROShrdArryOnRegSet.add(sArray);
						}
						cudaRegisterSet.add(sym);
						cudaRegisterROSet.add(sym);
					}
				}
				dataSet = (HashSet<SubArray>)cannot.get("registerRW");
				if( dataSet != null ) {
					for( SubArray sArray : dataSet ) {
						Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
						Symbol tSym = sym;
						if( sym instanceof AccessSymbol ) {
							tSym = ((AccessSymbol)sym).getMemberSymbol();
						}
						if( SymbolTools.isArray(tSym) || SymbolTools.isPointer(tSym) ) {
							Set<SubArray> sSet = null;
							if( shrdArryOnRegMap.containsKey(tSym) ) {
								sSet = shrdArryOnRegMap.get(tSym);
							} else {
								sSet = new HashSet<SubArray>();
								shrdArryOnRegMap.put(tSym, sSet);
							}
							sSet.add(sArray);
						}
						cudaRegisterSet.add(sym);
					}
				}
				dataSet = (HashSet<SubArray>)cannot.get("noregister");
				if( dataSet != null ) {
					for( SubArray sArray : dataSet ) {
						Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
						cudaNoRegisterSet.add(sym);
					}
				}
				dataSet = (HashSet<SubArray>)cannot.get("sharedRO");
				if( dataSet != null ) {
					for( SubArray sArray : dataSet ) {
						Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
						cudaSharedSet.add(sym);
						cudaSharedROSet.add(sym);
					}
				}
				dataSet = (HashSet<SubArray>)cannot.get("sharedRW");
				if( dataSet != null ) {
					for( SubArray sArray : dataSet ) {
						Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
						cudaSharedSet.add(sym);
					}
				}
				dataSet = (HashSet<SubArray>)cannot.get("noshared");
				if( dataSet != null ) {
					for( SubArray sArray : dataSet ) {
						Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
						cudaNoSharedSet.add(sym);
					}
				}
				dataSet = (HashSet<SubArray>)cannot.get("texture");
				if( dataSet != null ) {
					for( SubArray sArray : dataSet ) {
						Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
						cudaTextureSet.add(sym);
					}
				}
				dataSet = (HashSet<SubArray>)cannot.get("notexture");
				if( dataSet != null ) {
					for( SubArray sArray : dataSet ) {
						Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
						cudaNoTextureSet.add(sym);
					}
				}
				dataSet = (HashSet<SubArray>)cannot.get("constant");
				if( dataSet != null ) {
					for( SubArray sArray : dataSet ) {
						Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
						cudaConstantSet.add(sym);
					}
				}
				dataSet = (HashSet<SubArray>)cannot.get("noconstant");
				if( dataSet != null ) {
					for( SubArray sArray : dataSet ) {
						Symbol sym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
						cudaNoConstantSet.add(sym);
					}
				}
/*				dataSet = (HashSet<SubArray>)cannot.get("noreductionunroll");
				if( dataSet != null ) {
					cudaNoRedUnrollSet.addAll(dataSet);
				}
*/				String sData = (String)cannot.get("noloopcollapse");
				if( sData != null ) {
					noloopcollapse = true;
				}
			}
		}
		cudaRegisterSet.removeAll(cudaNoRegisterSet);
		cudaRegisterROSet.removeAll(cudaNoRegisterSet);
		cudaSharedSet.removeAll(cudaNoSharedSet);
		cudaSharedROSet.removeAll(cudaNoSharedSet);
		cudaTextureSet.removeAll(cudaNoTextureSet);
		cudaConstantSet.removeAll(cudaNoConstantSet);
		for( Symbol tSm : cudaNoRegisterSet ) {
			Set<SubArray> sSet = shrdArryOnRegMap.remove(tSm);
			if( sSet != null ) {
				ROShrdArryOnRegSet.removeAll(sSet);
			}
		}
		//Reduction array should be removed from shrdArryElmtCachingOnReg set.
		for( Symbol tSm : accreductionSet ) {
			Set<SubArray> sSet = shrdArryOnRegMap.remove(tSm);
			if( sSet != null ) {
				ROShrdArryOnRegSet.removeAll(sSet);
			}
		}
		
		//Check if condition
		Expression ifCond = null;
		Expression redIfCond = null;
		ACCAnnotation tAnnot = region.getAnnotation(ACCAnnotation.class, "if");
		if( tAnnot != null ) {
			ifCond = (Expression)tAnnot.get("if");
			ifCond = Symbolic.simplify(ifCond);
			if( ifCond instanceof IntegerLiteral ) {
				if( ((IntegerLiteral)ifCond).getValue() != 0 ) {
					ifCond = null; //Compiler knows that this kernel will be executed; ignore the if-condition.
				} else { //compiler knows that this region will not be outlined as a GPU kernel; skip conversion.
					return;
				}
			}
		}
		CompoundStatement ifCondBody = null;
		if( ifCond != null ) {
			ifCondBody = new CompoundStatement();
		}
		
		//Check async condition
		Expression asyncID = null;
		tAnnot = region.getAnnotation(ACCAnnotation.class, "async");
		if( tAnnot != null ) {
			Object obj = tAnnot.get("async");
			if( obj instanceof String ) {
				//asyncID = new NameID("INT_MAX");
                asyncID = new NameID("acc_async_noval");
			} else if( obj instanceof Expression ) {
				asyncID = (Expression)obj;
			}
		}
		
		//Check repeat clause in resilience region.
		boolean containsRepeatClause = false;
		ARCAnnotation tCAnnot = region.getAnnotation(ARCAnnotation.class, "resilience");
		if( enableFaultInjection && (tCAnnot != null) && tCAnnot.containsKey("repeat") ) {
			Expression ftcond = tCAnnot.get("ftcond");
			if( (ftcond ==null) || !(ftcond instanceof IntegerLiteral) 
					|| (((IntegerLiteral)ftcond).getValue() != 0) ) {
				containsRepeatClause = true;
			}
		}
		
		Expression loopIndex = null;
		//Find enclosing ForLoop if existing.
		Traversable tt1 = region.getParent();
		while( (tt1 != null) && !(tt1 instanceof Procedure) ) {
			if( tt1 instanceof ForLoop ) {
				loopIndex = LoopTools.getIndexVariable((ForLoop)tt1);
				break;
			} else {
				tt1 = tt1.getParent();
			}
		}
		
		//Find optimal point to insert GPU kernel configuration statements,
		//which is used both for privateTransformation and reductionTransformation.
		Statement confRefStmt = region;
		String iKey = "kernelConfPt_" + new_func_name;
		ACCAnnotation rAnnot = confRefStmt.getAnnotation(ACCAnnotation.class, iKey);
		if( rAnnot == null ) {
			rAnnot = AnalysisTools.ipFindFirstPragmaInParent(confRefStmt, ACCAnnotation.class, iKey, null, null);
		}
		if( rAnnot == null ) {
			PrintTools.println("[WARNING for ACC2CUDATranslator.extractComputeRegion()] kernel configuraion insertion" +
					" point is not found; original compute region will be used instead.\nEnclosing procedure: " + 
					cProc.getSymbolName() + "\n", 0);
			confRefStmt = region;
		} else {
			confRefStmt = (Statement)rAnnot.getAnnotatable();
		}
		CompoundStatement confRefParent = (CompoundStatement)confRefStmt.getParent();
		CompoundStatement prefixStmts = new CompoundStatement();
		CompoundStatement postscriptStmts = new CompoundStatement();
		
		if( kernelVerification ) {
			Statement clonedRegion = region.clone();
			AnalysisTools.removePragmas(clonedRegion, ACCAnnotation.class);
			AnalysisTools.removePragmas(clonedRegion, ARCAnnotation.class);
			AnalysisTools.removePragmas(clonedRegion, CetusAnnotation.class);
			if( enableFaultInjection ) {
				//Remove device function to inject faults.
				//FIXME: device functions called in the compute region should be removed too.
				List<FunctionCall> fCallList = IRTools.getFunctionCalls(clonedRegion);
				if( fCallList != null ) {
					for( FunctionCall fCall : fCallList ) {
						String fName = fCall.getName().toString();
						if( fName.startsWith("dev__HI_ftinjection") ) {
							Statement fCallStmt = fCall.getStatement();
							CompoundStatement pSt = (CompoundStatement)fCallStmt.getParent();
							pSt.removeStatement(fCallStmt);
						}
					}
				}
			}
			CompoundStatement tPStmt = (CompoundStatement)region.getParent();
			if( containsRepeatClause ) {
				ACCAnnotation resAnnot = region.getAnnotation(ACCAnnotation.class, "rpt_index");
				if( resAnnot == null ) {
					Tools.exit("[Internal ERROR in ACC2CUDATranslator()] can't find rpt_index internal variable; exit!" +
								"\nOpenACC annotation: " + cAnnot +
								"\nEnclosing Procedure: " + cProc.getSymbolName() + "\n");
				} else {
					Expression rpt_index = resAnnot.get("rpt_index");
					resAnnot.remove("rpt_index");
					CompoundStatement rptIfBody = new CompoundStatement();
					rptIfBody.addStatement(clonedRegion);
					IfStatement rptIfStmt = new IfStatement( new BinaryExpression(rpt_index.clone(),
							BinaryOperator.COMPARE_EQ, new IntegerLiteral(0)), rptIfBody );
					CompoundStatement cBody = cProc.getBody();
					Set<Symbol> outSymSet = new HashSet<Symbol>();
					outSymSet.addAll(accsharedSet);
					outSymSet.removeAll(accreadonlySet);
					for( Symbol outSym : outSymSet ) {
						SubArray outSA = accSharedMap.get(outSym);
						if( outSA == null ) {
							Tools.exit("[ERROR in FaultInjectionTransformation()] can not find subarray for symbol " + 
									outSym.getSymbolName() + " in the following OpenACC annotation:\n" + 
									"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n");
						}
						Expression hostVar = null;
						Expression ftrefVar = null;
						Expression ftoutVar = null;
						if( outSym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)outSym, null);
						} else {
							hostVar = new Identifier(outSym);
						}
						List<Expression> startList = new LinkedList<Expression>();
						List<Expression> lengthList = new LinkedList<Expression>();
						boolean foundDimensions = AnalysisTools.extractDimensionInfo(outSA, startList, lengthList, IRSymbolOnly, region);
						if( !foundDimensions ) {
							Tools.exit("[ERROR in FaultInjectionTransformation()] Dimension information " +
									"of the following variable is " +
									"unknown: " + outSA.getArrayName() + "\nOpenACC directive: " + cAnnot +
							"\nThe ACC2GPU translation failed!");
						}

						List<Specifier> typeSpecs = new ArrayList<Specifier>();
						Symbol IRSym = outSym;
						Symbol sym = outSym;
						if( outSym instanceof PseudoSymbol ) {
							IRSym = ((PseudoSymbol)outSym).getIRSymbol();
						}
						if( IRSymbolOnly ) {
							sym = IRSym;
							typeSpecs.addAll(((VariableDeclaration)outSym.getDeclaration()).getSpecifiers());
						} else {
							Symbol tSym = outSym;
							while( tSym instanceof AccessSymbol ) {
								tSym = ((AccessSymbol)tSym).getMemberSymbol();
							}
							typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
						}
						StringBuilder str = new StringBuilder(80);
						if( hostVar instanceof AccessExpression ) {
							str.append(TransformTools.buildAccessExpressionName((AccessExpression)hostVar));
						} else {
							str.append(hostVar.toString());
						}
						String symNameBase = str.toString();
						String ftrefName = "ftref__" + symNameBase;
						String ftoutName = "ftout__" + symNameBase;
						List<Specifier> clonedspecs = new ChainedList<Specifier>();
						clonedspecs.addAll(typeSpecs);
						clonedspecs.remove(Specifier.STATIC);
						///////////////////////////////////////////
						// GPU variables should not be constant. //
						///////////////////////////////////////////
						clonedspecs.remove(Specifier.CONST);
						//////////////////////////////
						// Remove extern specifier. //
						//////////////////////////////
						clonedspecs.remove(Specifier.EXTERN);
/*						if( clonedspecs.remove(Specifier.RESTRICT) ) {
							clonedspecs.add(CUDASpecifier.RESTRICT);
						}*/
						//Add "gpuBytes = SIZE_a * sizeof(float);" before the repeat loop.
						SizeofExpression sizeof_expr = new SizeofExpression(clonedspecs);
						Expression biexp = sizeof_expr.clone();
						for( int i=0; i<lengthList.size(); i++ )
						{
							biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
						}
						VariableDeclaration bytes_decl = (VariableDeclaration)SymbolTools.findSymbol(parentTrUnt, "gpuBytes");
						Identifier cloned_bytes = new Identifier((VariableDeclarator)bytes_decl.getDeclarator(0));			
						AssignmentExpression assignex = new AssignmentExpression(cloned_bytes.clone(),AssignmentOperator.NORMAL, 
								biexp);
						Statement gpuBytes_stmt = new ExpressionStatement(assignex);
						rptIfBody.addStatement(gpuBytes_stmt);
						/////////////////////////////////////////////////////////////////////////
						// Create a variable to keep the original value (ex: float *ftref__a;) //
						/////////////////////////////////////////////////////////////////////////
						Set<Symbol> symSet = cBody.getSymbols();
/*						Symbol ftref_sym = AnalysisTools.findsSymbol(symSet, ftrefName);
						if( ftref_sym != null ) {
							ftrefVar = new Identifier(ftref_sym);
						} else {
							//ERROR
						}*/
						///////////////////////////////////////////////////////////////////////////
						// Create a variable to keep the CPU-output value (ex: float *ftout__a;) //
						///////////////////////////////////////////////////////////////////////////
						symSet = cBody.getSymbols();
						Symbol ftout_sym = AnalysisTools.findsSymbol(symSet, ftoutName);
						if( ftout_sym != null ) {
							ftoutVar = new Identifier(ftout_sym);
						} else {
							//ERROR
							Tools.exit("[Internal ERROR in ACC2CUDATranslator()] can't find ft-out internal variable; exit!" +
									"\nOpenACC annotation: " + cAnnot +
									"\nEnclosing Procedure: " + cProc.getSymbolName() + "\n");
						}
						//Add memcpy() statement from host variable to  the ft-out host variable
						FunctionCall copy_call = new FunctionCall(new NameID("memcpy"));
						copy_call.addArgument(ftoutVar.clone());
						if( lengthList.size() == 0 ) { //hostVar is scalar.
							copy_call.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
									hostVar.clone()));
						} else {
							copy_call.addArgument(hostVar.clone());
						}
						copy_call.addArgument(cloned_bytes.clone());
						Statement memcpy_stmt = new ExpressionStatement(copy_call);
						rptIfBody.addStatement(memcpy_stmt);
					}
					tPStmt.addStatementAfter(region, rptIfStmt);
				}
			} else {
				tPStmt.addStatementAfter(region, clonedRegion);
			}
		}
		
		if( ifCond != null ) {
			//If if-condition exists, create a copy of this region to be executed on 
			//a host if the condition fails.
			Statement clonedRegion = region.clone();
			AnalysisTools.removePragmas(clonedRegion, ACCAnnotation.class);
			AnalysisTools.removePragmas(clonedRegion, ARCAnnotation.class);
			AnalysisTools.removePragmas(clonedRegion, CetusAnnotation.class);
			Statement dummyStmt = new AnnotationStatement();
			IfStatement ifStmt = new IfStatement(ifCond.clone(), dummyStmt,  clonedRegion);
			region.swapWith(ifStmt);
			dummyStmt.swapWith(region);
			WorkerSingleModeTransformation.removeWorkerSingleModeWrapper(clonedRegion);
			if( asyncID != null ) {
				//If reduction is asynchronous, post-reduction statements will not 
				//be in the true-clause of the if statement.
				redIfCond = ifCond.clone();
				if( confRefStmt == region ) {
					ifCond = null;
				}
			} else {
				//if confRefStmt == region, all translated statements will be in the true-clause 
				//of the if statement; we don't need the if-condition.
				if( confRefStmt == region ) {
					ifCond = null;
					redIfCond = null;
				} else {
					redIfCond = ifCond.clone();
				}
			}
			confRefParent = (CompoundStatement)confRefStmt.getParent();
			SymbolTools.linkSymbol(clonedRegion);
		}
		
		//Remove OpenMP pragmas existing in the current compute region.
		AnalysisTools.removePragmas(region, OmpAnnotation.class);
		
		
		//If acc_on_device() is called in a compute region, inline it!
		List<FunctionCall> dfCallList = IRTools.getFunctionCalls(region);
		if( dfCallList != null ) {
			for( FunctionCall fCall : dfCallList ) {
				String funcName = fCall.getName().toString();
				if( funcName.equals("acc_on_device") ) {
					Expression newExp = null;
					Expression fArg = fCall.getArgument(0);
					String fArgStr = fArg.toString();
					if( fArgStr.equals("acc_device_default") || fArgStr.equals("acc_device_not_host") ||
							fArgStr.equals("acc_device_nvidia") || fArgStr.equals("acc_device_gpu") ) {
						newExp = new IntegerLiteral(1);
					} else if( fArgStr.equals("acc_device_host") || fArgStr.equals("acc_device_none") ) {
						newExp = new IntegerLiteral(0);
					} else {
						newExp = new BinaryExpression(new BinaryExpression(fArg.clone(), BinaryOperator.COMPARE_NE, new NameID("acc_device_host")),
								BinaryOperator.LOGICAL_AND, 
								new BinaryExpression(fArg.clone(), BinaryOperator.COMPARE_NE, new NameID("acc_device_none")));
					}
					fCall.swapWith(newExp);
				}
			}
		}
		
		//Perform loop permutation if permute clause existing.
		List<ARCAnnotation> pmAnnots = AnalysisTools.ipCollectPragmas(region, ARCAnnotation.class, "permute", null);
		if( pmAnnots != null ) {
			for( ARCAnnotation pmAnnot : pmAnnots ) {
				Annotatable fAt = pmAnnot.getAnnotatable();
				if( fAt instanceof ForLoop ) {
					List<Expression> permuteList = (List<Expression>)pmAnnot.get("permute");
					ForLoop targetLoop = (ForLoop)fAt;
					ForLoop outermostLoop = TransformTools.permuteLoops(targetLoop, permuteList, false);
					if( outermostLoop == null ) {
						PrintTools.println("\n[WARNING] The permute transformation for the following region is failed. " +
								"\n Please check whether inserted clause is correct." +
								"\nOpenACC annotation: " + cAnnot +
								"\nEnclosing Procedure: " + cProc.getSymbolName() + "\n", 0);
						continue;
					} else {
						if( outermostLoop != targetLoop ) {
							pmAnnot.remove("permute");
							ARCAnnotation transAnnot = outermostLoop.getAnnotation(ARCAnnotation.class, "transform");
							if( transAnnot == null ) {
								transAnnot = new ARCAnnotation("transform", "_directive");
								outermostLoop.annotate(transAnnot);
							}
							transAnnot.put("permute", permuteList);
						}
					}
				} else {
					PrintTools.println("\n[WARNING] permute clause is applicable only to for-loops; the clause in the " +
							"following region will be ignored." + 
							"\nOpenACC annotation: " + cAnnot +
							"\nEnclosing Procedure: " + cProc.getSymbolName() + "\n", 0);
					continue;
				}
			}
		}
		
		////////////////////////////////////////////////////////
		// Auxiliary variables used for GPU kernel conversion //
		////////////////////////////////////////////////////////
		VariableDeclaration bytes_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuBytes");
		Identifier cloned_bytes = new Identifier((VariableDeclarator)bytes_decl.getDeclarator(0));
		VariableDeclaration gmem_decl = null;
		Identifier gmemsize = null;
		VariableDeclaration smem_decl = null;
		Identifier smemsize = null;
		ExpressionStatement gMemAdd_stmt = null;
		ExpressionStatement gMemSub_stmt =  null;
		if( opt_addSafetyCheckingCode ) {
			gmem_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuGmemSize");
			gmemsize = new Identifier((VariableDeclarator)gmem_decl.getDeclarator(0));					
			smem_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuSmemSize");
			smemsize = new Identifier((VariableDeclarator)smem_decl.getDeclarator(0));					
			gMemAdd_stmt = new ExpressionStatement( new AssignmentExpression(gmemsize,
					AssignmentOperator.ADD, (Identifier)cloned_bytes.clone()) );
			gMemSub_stmt = new ExpressionStatement( new AssignmentExpression((Identifier)gmemsize.clone(),
					AssignmentOperator.SUBTRACT, (Identifier)cloned_bytes.clone()) );
		}
		VariableDeclaration numBlocks_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuNumBlocks");
		Identifier numBlocks = new Identifier((VariableDeclarator)numBlocks_decl.getDeclarator(0));					
		VariableDeclaration numThreads_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuNumThreads");
		Identifier numThreads = new Identifier((VariableDeclarator)numThreads_decl.getDeclarator(0));					
		VariableDeclaration totalNumThreads_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "totalGpuNumThreads");
		Identifier totalNumThreads = new Identifier((VariableDeclarator)totalNumThreads_decl.getDeclarator(0));					
		ExpressionStatement gpuBytes_stmt = null;
		VariableDeclarator rowidSymbol = null;
		
		// The following variables will be added to each GPU kernel.
		// int _bid;
		// int _bsize;
		// int _tid;
		// int _gtid;
		Identifier bid = null;
		Identifier bsize = null;
		Identifier tid = null;
		Identifier gtid = null;
	
		
		///////////////////////////////////////////////////////////////////////////////////////
		// Create a kernel procedure, to which the current compute region is converted into, //
		// and a function call to the kernel procedure.                                      //
		///////////////////////////////////////////////////////////////////////////////////////
		List<Specifier> new_proc_ret_type = new LinkedList<Specifier>();
        new_proc_ret_type.add(CUDASpecifier.EXTERN);
        new_proc_ret_type.add(OpenACCSpecifier.EXTERN_C);
		new_proc_ret_type.add(CUDASpecifier.CUDA_GLOBAL);
		new_proc_ret_type.add(Specifier.VOID);

		Procedure new_proc = new Procedure(new_proc_ret_type,
				new ProcedureDeclarator(new NameID(new_func_name),
						new LinkedList()), new CompoundStatement());
		List<Expression> kernelConf = new ArrayList<Expression>();
		KernelFunctionCall call_to_new_proc = new KernelFunctionCall(new NameID(
				new_func_name), new LinkedList(), kernelConf);
        call_to_new_proc.setLinkedProcedure(new_proc);
		Statement kernelCall_stmt = new ExpressionStatement(call_to_new_proc);
		
		/////////////////////////////////////////////////////////////////////////////
		// Apply LoopCollapse optimization; currently LoopCollapse optimization is //
		// applied to Sparse Matrix-Vector Product (SPMV) patterns only.           //
		/////////////////////////////////////////////////////////////////////////////
		//DEBUG: temporarily disabled due to incomplete implemenation of CUDALoopCollapse.
/*		if( opt_LoopCollapse && !noloopcollapse ) {
			loopCollapseHandler.handleSMVP(region, false);
			rowidSymbol = loopCollapseHandler.getGpuRowidSymbol();
			if( rowidSymbol != null) {
				if( region instanceof ForLoop ) {
					region = ((ForLoop)region).getBody();
				}
				call_to_new_proc.addArgument(new Identifier(rowidSymbol));
				new_proc.addDeclaration(loopCollapseHandler.getRowidDecl());
			}
		}*/
		
		/////////////////////////////////////////////////////////////////////////////////////////
		// Apply stripmining transformation to fit the iteration size of a worksharing loop to //
		// the specified gang/worker sizes.                                                    //
		/////////////////////////////////////////////////////////////////////////////////////////
		ForLoop newLoop = worksharingLoopStripmining(cProc, cAnnot, cRegionKind);
		if( newLoop != null ) { //Target region is changed; update related local references.
			if( confRefStmt == region ) {
				confRefStmt = newLoop;
				confRefParent = (CompoundStatement)newLoop.getParent();
			}
			region = newLoop;
			cAnnot = region.getAnnotation(ACCAnnotation.class, cRegionKind);
		}
		
		///////////////////////////////////////////////
		// Handle array-element-caching on register. //
		///////////////////////////////////////////////
		Set<Symbol> arrayElmtCacheSymbols = ACC2GPUTranslationTools.arrayCachingOnRegister(region, shrdArryOnRegMap, ROShrdArryOnRegSet);
		
		///////////////////////////////////////////////////
		//Set GPU kernel configuration parameters partI. //
		///////////////////////////////////////////////////
		// dim3 dimGrid_kernelname(num_gangsX, num_gangsY, num_gangsZ);
		// dim3 dimBlock_kernelname(num_workersX, num_workersY, num_workersZ);
		// gpuNumBlocks = num_gangsX * num_gangsY * num_gangsZ;
		// gpuNumThreads = num_workersX * num_workersY * num_workersZ;
		// totalGpuNumThreads = gpuNumThreads * gpuNumBlocks;
		
		List<Expression> num_workers = new LinkedList<Expression>();
		List<Expression> num_gangs = new LinkedList<Expression>();
		Expression totalnumgangs = null;
		Expression totalnumworkers = null;
		tAnnot = region.getAnnotation(ACCAnnotation.class, "seq");
		boolean isSingleTask = false;
		if( tAnnot != null ) {
			if( !AnalysisTools.ipContainPragmas(region, ACCAnnotation.class, ACCAnnotation.parallelWorksharingClauses, false, null) ) {
				isSingleTask = true;
			}
		}
		if( isSingleTask ) {
			num_gangs.add(new IntegerLiteral(1));
			num_gangs.add(new IntegerLiteral(1));
			num_gangs.add(new IntegerLiteral(1));
			totalnumgangs = new IntegerLiteral(1);
			num_workers.add(new IntegerLiteral(1));
			num_workers.add(new IntegerLiteral(1));
			num_workers.add(new IntegerLiteral(1));
			totalnumworkers = new IntegerLiteral(1);
		} else {
			if( cRegionKind.equals("parallel") ) {
				tAnnot = region.getAnnotation(ACCAnnotation.class, "num_gangs");
				if( tAnnot == null ) {
					Tools.exit("[ERROR in ACC2CUDATranslator.extractComputeRegion()] num_gangs clause is missing;\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n");
				} else {
					num_gangs.add(((Expression)tAnnot.get("num_gangs")).clone());
					num_gangs.add(new IntegerLiteral(1));
					num_gangs.add(new IntegerLiteral(1));
				}
				totalnumgangs = num_gangs.get(0).clone();
				tAnnot = region.getAnnotation(ACCAnnotation.class, "num_workers");
				if( tAnnot == null ) {
					num_workers.add(new IntegerLiteral(defaultNumWorkers));
				} else {
					num_workers.add(((Expression)tAnnot.get("num_workers")).clone());
				}
				num_workers.add(new IntegerLiteral(1));
				num_workers.add(new IntegerLiteral(1));
				totalnumworkers = num_workers.get(0).clone();
				if( totalnumgangs.toString().equals("1") && totalnumworkers.toString().equals("1") ) {
					isSingleTask = true;
				}
			} else {
				tAnnot = region.getAnnotation(ACCAnnotation.class, "gangconf");
				if( tAnnot == null ) {
					Tools.exit("[ERROR in ACC2CUDATranslator.extractComputeRegion()] internal gangconf clause is missing;\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n");
				} else {
					List<Expression> gangConfs = tAnnot.get("gangconf");
					int tsize = gangConfs.size();
					for( int i=0; i<tsize; i++ ) {
						num_gangs.add(i, gangConfs.get(i).clone());
					}
					for( int i=tsize; i<3; i++ ) {
						num_gangs.add(i, new IntegerLiteral(1));
					}
				}
				tAnnot = region.getAnnotation(ACCAnnotation.class, "totalnumgangs");
				if( tAnnot == null ) {
					Tools.exit("[ERROR in ACC2CUDATranslator.extractComputeRegion()] internal totalnumgangs clause is missing;\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n");
				} else {
					totalnumgangs = ((Expression)tAnnot.get("totalnumgangs")).clone();
				}

				List<ACCAnnotation> tAnnotList = IRTools.collectPragmas(region, ACCAnnotation.class, "workerconf");
				if( (tAnnotList == null) || tAnnotList.isEmpty() ) {
					Tools.exit("[ERROR in ACC2CUDATranslator.extractComputeRegion()] internal workerconf clause is missing;\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n");
				} else {
					int m = 0;
					for( ACCAnnotation tAn : tAnnotList ) {
						List<Expression> workerConfs = tAn.get("workerconf");
						int tsize = workerConfs.size();
						if( m == 0 ) {
							for( int i=0; i<tsize; i++ ) {
								num_workers.add(i, workerConfs.get(i).clone());
							}
							for( int i=tsize; i<3; i++ ) {
								num_workers.add(i, new IntegerLiteral(1));
							}
						} else {
							for( int i=0; i<tsize; i++ ) {
								Expression exp1 = num_workers.get(i);
								Expression exp2 = workerConfs.get(i).clone();
								Expression exp3 = Symbolic.simplify(new MinMaxExpression(false, exp1, exp2));
								num_workers.set(i, exp3);
							}
						}
						m++;
					}
				}
				tAnnotList = IRTools.collectPragmas(region, ACCAnnotation.class, "totalnumworkers");
				if( (tAnnotList == null) || tAnnotList.isEmpty() ) {
					Tools.exit("[ERROR in ACC2CUDATranslator.extractComputeRegion()] internal totalnumworkers clause is missing;\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n");
				} else {
					int m = 0;
					for( ACCAnnotation tAn : tAnnotList ) {
						Expression exp1 = tAn.get("totalnumworkers");
						if( m == 0 ) {
							totalnumworkers = exp1.clone();
						} else {
							totalnumworkers = Symbolic.simplify(new MinMaxExpression(false, totalnumworkers, exp1));
						}
						m++;
					}
				}
				if( totalnumgangs.toString().equals("1") && totalnumworkers.toString().equals("1") ) {
					isSingleTask = true;
				}
			}
		}
		
		if( enableFaultInjection ) {
			//Assign a random number to target thread ID (_ti_targetThread).
			//e.g., _ti_targetThread = HI_genrandom_int(# of total threads);
			Symbol tThreadSym =  (Symbol)cAnnot.remove("targetThread");
			if( tThreadSym != null ) {
				//[TODO] Check ftthread clause in the current region or in enclosing 
				//resilience region.
				Expression ftthread = null;
				ARCAnnotation ttAnnot = region.getAnnotation(ARCAnnotation.class, "ftthread");
				if(  ttAnnot == null ) {
					ttAnnot = AnalysisTools.ipFindFirstPragmaInParent(region, 
							ARCAnnotation.class, "resilience", null, null);
				}
				if( (ttAnnot != null) && ttAnnot.containsKey("ftthread") ) {
					ftthread = ttAnnot.get("ftthread");
				}
				Expression REXP = null;
				if( ftthread == null ) {
					FunctionCall fCall = new FunctionCall(new NameID("HI_genrandom_int"));
					fCall.addArgument(Symbolic.multiply(totalnumgangs.clone(), totalnumworkers.clone()));
					REXP = fCall;
				} else {
					REXP = ftthread.clone();
				}
				Statement thStmt = new ExpressionStatement(new AssignmentExpression(new Identifier(tThreadSym),
						AssignmentOperator.NORMAL, REXP));
				CompoundStatement tPStmt = (CompoundStatement)region.getParent();
				tPStmt.addStatementBefore(region, thStmt);

			}
		}

		List<Statement> preList = new LinkedList<Statement>(); 
		List<Statement> postList = new LinkedList<Statement>(); 
		//////////////////////////////////////////
		// Handle private/firstprivate clauses. //
		//////////////////////////////////////////
		CUDATranslationTools.privateTransformation(cProc, region, cRegionKind, ifCond, asyncID, confRefStmt, prefixStmts, 
				postscriptStmts, preList, postList, call_to_new_proc, new_proc, main_TrUnt, OpenACCHeaderEndMap, IRSymbolOnly, 
				opt_addSafetyCheckingCode, arrayElmtCacheSymbols, isSingleTask);
		if( SkipGPUTranslation == 3 ) {
			return;
		}
		
		//////////////////////////////////////
		// Perform reduction transformation //
		//////////////////////////////////////
		CUDATranslationTools.reductionTransformation(cProc, region, cRegionKind, redIfCond, asyncID, confRefStmt, prefixStmts, 
				postscriptStmts, preList, postList, call_to_new_proc, new_proc, main_TrUnt, OpenACCHeaderEndMap, IRSymbolOnly, 
				opt_addSafetyCheckingCode, opt_UnrollingOnReduction, maxBlockSize, totalnumgangs.clone(), kernelVerification,
				memtrVerification, marginOfError, warpSize, minCheckValue, localRedVarConf, isSingleTask);
		if( SkipGPUTranslation == 4 ) {
			return;
		}
		
		///////////////////////////////////////////////////
		// Handle kernels/parallel loop with seq clause. //
		///////////////////////////////////////////////////
		if( (region instanceof ForLoop) && region.containsAnnotation(ACCAnnotation.class, "seq") &&
				!AnalysisTools.ipContainPragmas(region, ACCAnnotation.class, ACCAnnotation.parallelWorksharingClauses, false, null)) {
			ACC2GPUTranslationTools.seqKernelLoopTransformation(cProc, (ForLoop)region, cRegionKind, ifCond, asyncID, confRefStmt,
					preList, postList, prefixStmts, postscriptStmts, call_to_new_proc, new_proc, main_TrUnt, 
					OpenACCHeaderEndMap, IRSymbolOnly, opt_addSafetyCheckingCode, targetModel, opt_AssumeNoAliasing);
		}

		//////////////////////////////////////////////////////////////////////////
		// Insert __syncthreads() calls for each #pragam acc barrier directive. //
		// __syncthreads() is used to coordinate communication between the      //
		//threads of the same block. When some threads within a block access    //
		//the same addresses in shared or global memory, there are potential    //
		//read-after-write, write-after-read, or write-after- write hazards for //
		//some of these memory accesses. These data hazards can be avoided by   //
		//synchronizing threads in-between these accesses.                      //
		//////////////////////////////////////////////////////////////////////////
		List<ACCAnnotation> barrierAnnots = AnalysisTools.ipCollectPragmas(
				region, ACCAnnotation.class, "barrier", null);
		if( barrierAnnots != null ) {
			FunctionCall syncCall = new FunctionCall(new NameID("__syncthreads"));
			ExpressionStatement syncCallStmt = new ExpressionStatement(syncCall);
			for( ACCAnnotation bAnnot : barrierAnnots ) {
				Statement bStmt = (Statement)bAnnot.getAnnotatable();
				Statement syncCStmt = syncCallStmt.clone();
				syncCStmt.annotate(bAnnot);
				bStmt.swapWith(syncCStmt);
			}
		}
		
		pitchedSymMap.clear();
		textureSymMap.clear();
		textureOffsetMap.clear();
		constantSymMap.clear();
		Set<Symbol> callerProcSymSet = new HashSet<Symbol>();
		// Perform kernel code conversion for each shared symbol
		Collection<Symbol> sortedSet = AnalysisTools.getSortedCollection(accSharedMap.keySet());
		for( Symbol sharedSym : sortedSet ) {
			SubArray sArray = accSharedMap.get(sharedSym);
			Expression hostVar = sArray.getArrayName();

			List<Specifier> removeSpecs = new ArrayList<Specifier>();
			removeSpecs.add(Specifier.STATIC);
			removeSpecs.add(Specifier.CONST);
			removeSpecs.add(Specifier.EXTERN);
			List<Specifier> typeSpecs = new ArrayList<Specifier>();
			Boolean isStruct = false;
			Symbol IRSym = sharedSym;
			if( sharedSym instanceof PseudoSymbol ) {
				IRSym = ((PseudoSymbol)sharedSym).getIRSymbol();
			}
			if( IRSymbolOnly ) {
				hostVar = new Identifier(IRSym);
				typeSpecs.addAll(((VariableDeclaration)IRSym.getDeclaration()).getSpecifiers());
				isStruct = SymbolTools.isStruct(IRSym, region);
			} else {
				Symbol tSym = sharedSym;
				while( tSym instanceof AccessSymbol ) {
					tSym = ((AccessSymbol)tSym).getMemberSymbol();
				}
				typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
				isStruct = SymbolTools.isStruct(tSym, region);
			}
			typeSpecs.removeAll(removeSpecs);
			//Replace C restrict keyword to CUDA __restrict__ keyword.
/*			if( typeSpecs.remove(Specifier.RESTRICT) ) {
				typeSpecs.add(CUDASpecifier.RESTRICT);
			}*/

			Boolean isArray = SymbolTools.isArray(sharedSym);
			Boolean isPointer = SymbolTools.isPointer(sharedSym);
			if( sharedSym instanceof NestedDeclarator ) {
				isPointer = true;
			}
			for( Object tObj : typeSpecs ) {
				if( tObj instanceof UserSpecifier ) {
					IDExpression tExp = ((UserSpecifier)tObj).getIDExpression();
					String tExpStr = tExp.getName();
					if( !tExpStr.startsWith("struct") && !tExpStr.startsWith("enum") ) {
						Declaration tDecl = SymbolTools.findSymbol(global_table, tExp);
						if( tDecl != null ) {
							if( tDecl instanceof VariableDeclaration ) {
								if( ((VariableDeclaration)tDecl).getSpecifiers().contains(Specifier.TYPEDEF) ) {
									Declarator tDeclr = ((VariableDeclaration)tDecl).getDeclarator(0);
									if( tDeclr instanceof NestedDeclarator ) {
										isPointer =  true;
										break;
									} else if( tDeclr instanceof VariableDeclarator ) {
										if( SymbolTools.isArray((VariableDeclarator)tDeclr) ) {
											isArray= true;
											break;
										} else if( SymbolTools.isPointer((VariableDeclarator)tDeclr) ) {
											isPointer= true;
											break;
										}
									}
								}
							}
						}
					}
					break;
				}
			}

			Boolean isScalar = !isArray && !isPointer;
			
			List<Expression> startList = new LinkedList<Expression>();
			List<Expression> lengthList = new LinkedList<Expression>();
			boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, region);
			int dimension = lengthList.size();
			if( (!foundDimensions) && (dimension>1) ) {
				//It's OK to miss the left-most dimension.
				boolean missingDimFound = false;
				for(int m=1; m<dimension; m++) {
					if( lengthList.get(m) == null ) {
						missingDimFound = true;
						break;
					}
				}
				if( missingDimFound ) {
					Tools.exit("[ERROR in ACC2CUDATranslator.extractComputeRegion()] Dimension information of the following variable is" +
							"unknown: " + sArray.getArrayName() + "\nOpenACC directive: " + cAnnot +
					"\nThe ACC2GPU translation failed!");
				}
			}
			//PrintTools.println("sArray: " + sArray + ", dimension: " + lengthList.size() + ", sArray.getArrayDimension(): " + sArray.getArrayDimension() + "\n" , 0);
			
			Symbol gpuSym = null;
			Identifier gpuVar = null;
			Identifier kParamVar = null;
			String symNameBase = null;
			if( sharedSym instanceof AccessSymbol) {
				symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)sharedSym);
			} else {
				symNameBase = sharedSym.getSymbolName();
			}
			String gpuVarName = "gpu__" + symNameBase;
			String constVarName = "const__" + symNameBase;
			String textureVarName = "texture__" + symNameBase;
			String pitchVarName = "pitch__" + symNameBase;
			String kParamVarName = symNameBase;
			if( !SymbolTools.isGlobal(IRSym) ) {
				constVarName += "__" + cProc.getSymbolName();
				textureVarName += "__" + cProc.getSymbolName();
			}

			
			if( accDevicePtrSet.contains(sharedSym) ) {
				//Create a kernel parameter for the shared array variable.
				kParamVar = TransformTools.declareClonedVariable(new_proc, sharedSym, kParamVarName, removeSpecs, null, true, opt_AssumeNoAliasing);
				callerProcSymSet.add(kParamVar.getSymbol());
				if( dimension == 1 ) {
					call_to_new_proc.addArgument(hostVar.clone());
				} else {
					// Insert argument to the kernel function call
					//Cast the device pointer variable to pointer-to-array type if it is. 
					// Ex: (float (*)[SIZE2]) x
					List castspecs = new LinkedList();
					castspecs.addAll(typeSpecs);
					/*
					 * FIXME: NestedDeclarator was used for (*)[SIZE2], but this may not be 
					 * semantically correct way to represent (*)[SIZE2] in IR.
					 */
					List tindices = new LinkedList();
					for( int i=1; i<dimension; i++) {
						tindices.add(lengthList.get(i).clone());
					}
					ArraySpecifier aspec = new ArraySpecifier(tindices);
					List tailSpecs = new ArrayList(1);
					tailSpecs.add(aspec);
					VariableDeclarator childDeclr = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, new NameID(""));
					NestedDeclarator nestedDeclr = new NestedDeclarator(new ArrayList(), childDeclr, null, tailSpecs);
					castspecs.add(nestedDeclr);
					call_to_new_proc.addArgument(new Typecast(castspecs, (Identifier)hostVar.clone()));
				}
				// Replace all instances of the shared variable to the parameter variable
				if( sharedSym instanceof AccessSymbol ) {
					TransformTools.replaceAccessExpressions(region, (AccessSymbol)sharedSym, kParamVar);
				} else {
					TransformTools.replaceAll(region, hostVar, kParamVar);
				}
				continue;
			}

			if( isScalar && !isStruct ) {
				if( cudaSharedROSet.contains(sharedSym) ) {
					// Create a GPU kernel parameter corresponding to shared_var
					VariableDeclarator kParam_declarator = new VariableDeclarator(new NameID(kParamVarName));
					callerProcSymSet.add(kParam_declarator);
					VariableDeclaration kParam_decl = new VariableDeclaration(typeSpecs,
							kParam_declarator);
					kParamVar = new Identifier(kParam_declarator);
					new_proc.addDeclaration(kParam_decl);

					// Insert argument to the kernel function call
					if( sharedSym instanceof AccessSymbol ) {
						AccessExpression accExp = AnalysisTools.accessSymbolToExpression((AccessSymbol)sharedSym, null);
						call_to_new_proc.addArgument(accExp);
					} else {
						call_to_new_proc.addArgument(new Identifier(sharedSym));
					}

					// Replace the instance of shared variable with the new gpu_var.
					if( sharedSym instanceof AccessSymbol ) {
						TransformTools.replaceAccessExpressions(region, (AccessSymbol)sharedSym, kParamVar);
					} else {
						TransformTools.replaceAll(region, new Identifier(sharedSym), kParamVar);
					}
					continue;
				}
			}

			MallocType mallocT = MallocType.NormalMalloc;
			if( cudaConstantSet.contains(sharedSym) ) {
				//CUDA does not allow dynamically allocated constant memory, and thus data should have
				//constant size.
				if( dimension == 0 ) {
					mallocT = MallocType.ConstantMalloc;
				} else {
					boolean constantDimension = true;
					for( Expression tDim : lengthList ) {
						if( (tDim == null) || !(tDim instanceof IntegerLiteral) ) {
							constantDimension = false;
							break;
						}
					}
					if( constantDimension ) {
						mallocT = MallocType.ConstantMalloc;
					}
				}
			} else if( cudaTextureSet.contains(sharedSym) ) {
				if( dimension == 1 ) {
					if( sharedSym.getTypeSpecifiers().contains(Specifier.DOUBLE) ) {
						PrintTools.println("\n[WARNING] CUDA does not support double textures; " +
								"textureing of the following double variable will be skipped!\n" + 
								"Variable: " + hostVar + ", ACCAnnotation: " + cAnnot + "\n", 0);
					} else {
						mallocT = MallocType.TextureMalloc;
					}
				} else {
					PrintTools.println("\n[WARNING] current implementation supports only 1D texture reference " +
							"bound to linear memory; textureing of the following variable will be skipped!\n" + 
							"Variable: " + hostVar + ", ACCAnnotation: " + cAnnot + "\n", 2);
				}
			} else if( opt_MallocPitch ) {
				//FIXME: for now, MallocPitch is allowed only to 2D array
				//if( (dimension == 2) || (dimension == 3) ) 
				if( dimension == 2 ) {
					mallocT = MallocType.PitchedMalloc;
				}
			}
			
			Set<Symbol> symSet = null;
			
			if( mallocT == MallocType.ConstantMalloc ) {
				//symSet = parentTrUnt.getSymbols();
                symSet = kernelsTranslationUnit.getSymbols();
				Symbol constSym = AnalysisTools.findsSymbol(symSet, constVarName);
				if( constSym == null ) {
					//constant symbol should have been created by either handleDataClause() or handleUpdate().
					Tools.exit("[ERROR in ACC2CUDATranslation.extractComputeRegion()] Can't find __constant__ variable (" + constVarName + 
							") corresponding to the host variable, " + hostVar + "; exit the program!\nEnclosing procedure: " + 
							cProc.getSymbolName() + "\nACCAnnotation: " + cAnnot.toString() +"\n");
				}
				Identifier constVar = new Identifier(constSym);
				// Replace the instance of shared variable with the new gpu_var.
				if( sharedSym instanceof AccessSymbol ) {
					TransformTools.replaceAccessExpressions(region, (AccessSymbol)sharedSym, constVar);
				} else {
					TransformTools.replaceAll(region, new Identifier(sharedSym), constVar);
				}
				constantSymMap.put(constSym, constSym);
				callerProcSymSet.add(constSym);
				continue;
			}
			
			SymbolTable targetSymbolTable = AnalysisTools.getIRSymbolScope(IRSym, region.getParent());
			if( targetSymbolTable instanceof Procedure ) {
				targetSymbolTable = ((Procedure)targetSymbolTable).getBody();
			}
			if( targetSymbolTable instanceof CompoundStatement ) {
				if( AnalysisTools.ipFindFirstPragmaInParent(region, OmpAnnotation.class, new HashSet(Arrays.asList("parallel", "task")), false, null, null) != null ) { 
					targetSymbolTable = (CompoundStatement)region.getParent();
				}
			}
			
			//FIXME: below will work only for scalar variable, since multiple instances of array variables
			//are possible in the registerRO and registerRW clauses.
			boolean useRegister = false;
			boolean useSharedMemory = false;
			boolean ROData = false;
			if( cudaRegisterSet.contains(sharedSym) ) {
				useRegister = true;
			}
			if( cudaRegisterROSet.contains(sharedSym) ) {
				useRegister = true;
				ROData = true;
			}
			if( cudaSharedSet.contains(sharedSym) ) {
				useSharedMemory = true;
			}
			if( cudaSharedROSet.contains(sharedSym) ) {
				useSharedMemory = true;
				ROData = true;
			}

			/////////////////////////////////////////////////////////////
			// Find a GPU device variable corresponding to shared_var. //
			// Ex: float * gpu__b;                                     //
			/////////////////////////////////////////////////////////////
			symSet = targetSymbolTable.getSymbols();
			gpuSym = AnalysisTools.findsSymbol(symSet, gpuVarName);
			gpuVar = new Identifier(gpuSym);
			if( gpuSym == null ) {
				Tools.exit("[ERROR in ACC2CUDATranslation.extractComputeRegion()] Can't find device variable corresponding" +
						" to the host variable, " + hostVar + "; exit the program!\nEnclosing procedure: " + 
						cProc.getSymbolName() + "\nACCAnnotation: " + cAnnot.toString() +"\n");
			}
			
			if( isScalar ) {
				CUDATranslationTools.scalarSharedConv(sharedSym, symNameBase, typeSpecs,
						gpuSym, region, new_proc, call_to_new_proc, useRegister, false, ROData);
				//We done need to insert scalar symbol to callerProcSymSet.
			} else {
				//Create a kernel parameter for the shared array variable.
				kParamVar = TransformTools.declareClonedVariable(new_proc, sharedSym, kParamVarName, removeSpecs, null, true, opt_AssumeNoAliasing);
				Symbol kParamSym = kParamVar.getSymbol();
				callerProcSymSet.add(kParamSym);
				// Insert argument to the kernel function call
				if( dimension == 1 ) {
					call_to_new_proc.addArgument(gpuVar.clone());
				} else {
					//Cast the gpu variable to pointer-to-array type 
					// Ex: (float (*)[SIZE2]) gpu__x
					List castspecs = new LinkedList();
					castspecs.addAll(typeSpecs);
					/*
					 * FIXME: NestedDeclarator was used for (*)[SIZE2], but this may not be 
					 * semantically correct way to represent (*)[SIZE2] in IR.
					 */
					List tindices = new LinkedList();
					for( int i=1; i<dimension; i++) {
						tindices.add(lengthList.get(i).clone());
					}
					ArraySpecifier aspec = new ArraySpecifier(tindices);
					List tailSpecs = new ArrayList(1);
					tailSpecs.add(aspec);
					VariableDeclarator childDeclr = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, new NameID(" "));
					NestedDeclarator nestedDeclr = new NestedDeclarator(new ArrayList(), childDeclr, null, tailSpecs);
					castspecs.add(nestedDeclr);
					call_to_new_proc.addArgument(new Typecast(castspecs, (Identifier)gpuVar.clone()));
				}
				
				if( mallocT == MallocType.TextureMalloc ) {
					//Get from kernel translation unit instead
					//symSet = parentTrUnt.getSymbols();
					symSet = kernelsTranslationUnit.getSymbols();
					Symbol textureSym = AnalysisTools.findsSymbol(symSet, textureVarName);
					if( textureSym == null ) {
						//texture reference symbol should have been created by either handleDataClause() or handleUpdate().
						Tools.exit("[ERROR in ACC2CUDATranslation.extractComputeRegion()] Can't find texture reference " +
								"variable corresponding to the host variable, " + hostVar + 
								"; exit the program!\nEnclosing procedure: " + 
								cProc.getSymbolName() + "\nACCAnnotation: " + cAnnot.toString() +"\n");
					}
					Identifier textureRefID = new Identifier(textureSym);
					CUDATranslationTools.textureConv(sharedSym, textureRefID, region, null);
					textureSymMap.put(kParamSym, textureSym);
				} else if( mallocT == MallocType.PitchedMalloc ) {
					symSet = targetSymbolTable.getSymbols();
					Symbol pitchSym = AnalysisTools.findsSymbol(symSet, pitchVarName);
					if( pitchSym == null ) {
						//pitch symbol should have been created by either handleDataClause() or handleUpdate().
						Tools.exit("[ERROR in ACC2CUDATranslation.extractComputeRegion()] Can't find texture reference " +
								"variable corresponding to the host variable, " + hostVar + 
								"; exit the program!\nEnclosing procedure: " + 
								cProc.getSymbolName() + "\nACCAnnotation: " + cAnnot.toString() +"\n");
					}
					Identifier paramPitchVar = TransformTools.declareClonedVariable(new_proc, pitchSym, pitchSym.getSymbolName(), removeSpecs, null, true, opt_AssumeNoAliasing);
					Identifier pitchVar = new Identifier(pitchSym);
					call_to_new_proc.addArgument(pitchVar.clone());
					/* 
					 * If MallocPitch is used to allocate 2 dimensional array, gpu_a,
					 * replace array access expression with pointer access expression with pitch
					 * Ex: gpu__a[i][k] => *((float *)((char *)gpu__a + i * pitch__a) + k)
					 */
					CUDATranslationTools.pitchedAccessConv(sharedSym, kParamVar, typeSpecs,
							paramPitchVar, region);
					pitchedSymMap.put(kParamSym, paramPitchVar.getSymbol());
				} else if( mallocT == MallocType.NormalMalloc ) {
					//Some array-caching optimization may be put here.
				}
				// Replace all instances of the shared variable to the paremeter variable
				if( sharedSym instanceof AccessSymbol ) {
					TransformTools.replaceAccessExpressions(region, (AccessSymbol)sharedSym, kParamVar);
				} else {
					TransformTools.replaceAll(region, hostVar, kParamVar);
				}
			}
		}
		
		/////////////////////////////////////////////////////////////
		//Handle device functions called in the current GPU kernel.//
		/////////////////////////////////////////////////////////////
		String TrUntCnt = null;
		int i = 0;
		for( Traversable tt : program.getChildren() ) {
			if( parentTrUnt.equals(tt) ) {
				TrUntCnt = new IntegerLiteral(i).toString();
				break;
			}
			i++;
		}
		//If the same procedure is called in kernel regions in different translation units,
		//these should be handled separately.
		devProcMap = tr2DevProcMap.get(parentTrUnt);
		if( devProcMap == null ) {
			devProcMap = new HashMap<Procedure, Map<String, Procedure>>();
			tr2DevProcMap.put(parentTrUnt, devProcMap);
		}
		Stack<Procedure> devProcStack = new Stack<Procedure>();
		devProcCloning(region, parentTrUnt, TrUntCnt, callerProcSymSet, devProcStack);


		///////////////////////////////////////////////////
		//Set GPU kernel configuration parameters partII. //
		///////////////////////////////////////////////////

		//Dim3Specifier dim3Spec = new Dim3Specifier(num_gangs.get(0), num_gangs.get(1), num_gangs.get(2));
		//VariableDeclarator dimGrid_declarator = new VariableDeclarator(new NameID("dimGrid_"+new_func_name), dim3Spec);
		//Identifier dimGrid = new Identifier(dimGrid_declarator);
		//Declaration dimGrid_decl = new VariableDeclaration(CUDASpecifier.CUDA_DIM3, dimGrid_declarator);
		//TransformTools.addStatementBefore(confRefParent, confRefStmt, new DeclarationStatement(dimGrid_decl));

		VariableDeclarator dimGrid_declarator = new VariableDeclarator(new NameID("dimGrid_"+new_func_name), new ArraySpecifier(new IntegerLiteral(3)));
		Identifier dimGrid = new Identifier(dimGrid_declarator);
		Declaration dimGrid_decl = new VariableDeclaration(CUDASpecifier.INT, dimGrid_declarator);
		DeclarationStatement dimGrid_stmt = new DeclarationStatement(dimGrid_decl);
		TransformTools.addStatementBefore(confRefParent, confRefStmt, dimGrid_stmt);
        for(int j = 2; j >= 0; j--)
        {
            AssignmentExpression assignmentExpression = new AssignmentExpression(
                    new ArrayAccess(new NameID("dimGrid_"+new_func_name), new IntegerLiteral(j)),
                    AssignmentOperator.NORMAL,
                    num_gangs.get(j).clone()
            );
            TransformTools.addStatementAfter(confRefParent, dimGrid_stmt, new ExpressionStatement(assignmentExpression));
        }

		//Dim3Specifier dim3Spec = new Dim3Specifier(num_workers.get(0), num_workers.get(1), num_workers.get(2));
		//VariableDeclarator dimBlock_declarator = new VariableDeclarator(new NameID("dimBlock_"+new_func_name), dim3Spec);
		//Identifier dimBlock = new Identifier(dimBlock_declarator);
		//Declaration dimBlock_decl = new VariableDeclaration(CUDASpecifier.CUDA_DIM3, dimBlock_declarator);
		//TransformTools.addStatementBefore(confRefParent, confRefStmt, new DeclarationStatement(dimBlock_decl));

		VariableDeclarator dimBlock_declarator = new VariableDeclarator(new NameID("dimBlock_"+new_func_name), new ArraySpecifier(new IntegerLiteral(3)));
		Identifier dimBlock = new Identifier(dimBlock_declarator);
		Declaration dimBlock_decl = new VariableDeclaration(CUDASpecifier.INT, dimBlock_declarator);
		DeclarationStatement dimBlock_stmt = new DeclarationStatement(dimBlock_decl);
		TransformTools.addStatementBefore(confRefParent, confRefStmt, dimBlock_stmt);
        for(int j = 2; j >= 0; j--)
        {
            AssignmentExpression assignmentExpression = new AssignmentExpression(
                    new ArrayAccess(new NameID("dimBlock_"+new_func_name), new IntegerLiteral(j)),
                    AssignmentOperator.NORMAL,
                    num_workers.get(j).clone()
            );
            TransformTools.addStatementAfter(confRefParent, dimBlock_stmt, new ExpressionStatement(assignmentExpression));
        }

		AssignmentExpression assignExp = new AssignmentExpression(numBlocks.clone(), AssignmentOperator.NORMAL, totalnumgangs);
		ExpressionStatement estmt = new ExpressionStatement(assignExp);
		if( ifCond == null ) {
			confRefParent.addStatementBefore(confRefStmt, estmt);
		} else {
			ifCondBody.addStatement(estmt);
		}
		
		assignExp = new AssignmentExpression(numThreads.clone(), AssignmentOperator.NORMAL, totalnumworkers);
		estmt = new ExpressionStatement(assignExp);
		if( ifCond == null ) {
			confRefParent.addStatementBefore(confRefStmt, estmt);
		} else {
			ifCondBody.addStatement(estmt);
		}
		
		assignExp = new AssignmentExpression(totalNumThreads.clone(), AssignmentOperator.NORMAL, 
				Symbolic.multiply(totalnumgangs.clone(), totalnumworkers.clone()));
		estmt = new ExpressionStatement(assignExp);
		if( ifCond == null ) {
			confRefParent.addStatementBefore(confRefStmt, estmt);
		} else {
			ifCondBody.addStatement(estmt);
		}
		List<Traversable> prefixList = prefixStmts.getChildren();
		for( Traversable tPref : prefixList ) {
			if( tPref instanceof DeclarationStatement ) {
				DeclarationStatement declStmt = (DeclarationStatement)tPref;
				Declaration decl = declStmt.getDeclaration();
				decl.setParent(null);
				confRefParent.addDeclaration(decl);
			} else {
				Statement stmt = (Statement)tPref;
				stmt.setParent(null);
				if( ifCond == null ) {
					confRefParent.addStatementBefore(confRefStmt, stmt);
				} else {
					ifCondBody.addStatement(stmt);
				}
			}
		}
		if( ifCond != null ) {
			IfStatement ifStmt = new IfStatement(ifCond.clone(), ifCondBody);
			confRefParent.addStatementBefore(confRefStmt, ifStmt);
		}
		
		
		List<Traversable> postscriptList = postscriptStmts.getChildren();
		List<DeclarationStatement> declStmts = new LinkedList<DeclarationStatement>();
		for(Traversable tPref : postscriptList ) {
			if( tPref instanceof DeclarationStatement ) {
				DeclarationStatement declStmt = (DeclarationStatement)tPref;
				declStmts.add(declStmt);
			}
		}
		for( DeclarationStatement declS : declStmts ) {
			postscriptStmts.removeChild(declS);
			Declaration decl = declS.getDeclaration();
			decl.setParent(null);
			confRefParent.addDeclaration(decl);
		}
		if( postscriptStmts.getChildren().size() > 0 ) {
/*			assignExp = new AssignmentExpression(numBlocks.clone(), AssignmentOperator.NORMAL, totalnumgangs.clone());
			estmt = new ExpressionStatement(assignExp);
			Traversable pS = postscriptStmts.getChildren().get(0);
			postscriptStmts.addStatementBefore((Statement)pS, estmt);
*/			if( ifCond != null ) {
				IfStatement ifStmt = new IfStatement(ifCond.clone(), postscriptStmts);
				confRefParent.addStatementAfter(confRefStmt, ifStmt);
			} else {
				postscriptList = postscriptStmts.getChildren();
				int tSize = postscriptList.size();
				for( int m = tSize-1; m>=0; m-- ) {
					Traversable tP = postscriptList.get(m);
					tP.setParent(null);
					confRefParent.addStatementAfter(confRefStmt, (Statement)tP);
				}
			}
		}
		assignExp = new AssignmentExpression(numBlocks.clone(), AssignmentOperator.NORMAL, totalnumgangs.clone());
		estmt = new ExpressionStatement(assignExp);
		((CompoundStatement)region.getParent()).addStatementAfter(region, estmt);
		
		//////////////////////////////////////////////////////////////////////////////////
		//If extractTuningParameters option is on, insert iteration space infomation to //
		//the tuning-parameter file as a comment.                                       //
		//////////////////////////////////////////////////////////////////////////////////
		if( tuningParamFile != null ) {
			ARCAnnotation aAnnot = region.getAnnotation(ARCAnnotation.class, "ainfo");
			String kernelID;
			if( aAnnot != null ) {
				kernelID = aAnnot.toString();
			} else {
				kernelID = "#" + call_to_new_proc.getName().toString();
			}
			try {
				BufferedWriter out = new BufferedWriter(new FileWriter(tuningParamFile, true));
				out.write(kernelID + " totalnumthreads=" + 
						Symbolic.multiply(totalnumgangs.clone(), totalnumworkers.clone()));
				out.newLine();
				out.close();
			} catch (Exception e) {
				PrintTools.println("[ERROR in ACC2CUDATranslator.extractComputeRegion()] writing to a file, "+ tuningParamFile + 
						", failed.", 0);
			}
		}
		
		StringLiteral refName = null;
		if( memtrVerification ) {
			List<FunctionCall> fCallList = IRTools.getFunctionCalls(program);
			//Get refname to be used for memory-transfer verification.
			ACCAnnotation iAnnot = region.getAnnotation(ACCAnnotation.class, "refname");
			if( iAnnot == null ) {
				StringBuilder str = new StringBuilder("[ERROR in ACC2CUDATranslator.extractComputeRegion()] can not find referenc name " +
						"used for memory transfer verification; please turn off the verification option " +
						"(programVerification != 1).\n" +
						"OpenACC Annotation: " + cAnnot + "\n");
				if( cProc != null ) {
					str.append("Enclosing Procedure: " + cProc.getSymbolName() + "\n");
				} else if( parentTrUnt != null ) {
					str.append("Enclosing File: " + parentTrUnt.getInputFilename() + "\n");
				}
				Tools.exit(str.toString());
			} else {
				refName = new StringLiteral((String)iAnnot.get("refname"));
			}
			tAnnot = region.getAnnotation(ACCAnnotation.class, "tempinternal");
			if( tAnnot != null ) {
				List<ACCAnnotation> pragmas = region.getAnnotations(ACCAnnotation.class);
				Map<Symbol, Symbol> g2lSymMap = new HashMap<Symbol, Symbol>();
				//Remove tempinternal annotation.
				region.removeAnnotations(ACCAnnotation.class);
				for( ACCAnnotation nAnnot : pragmas ) {
					if( !nAnnot.containsKey("tempinternal") ) {
						region.annotate(nAnnot);
					}
				}
				CompoundStatement cStmt = (CompoundStatement)region.getParent();
				Set<Symbol> accessedSyms = null;
				Set<Symbol> firstWriteSet = tAnnot.get("firstwriteSet");
				Set<Symbol> firstReadSet = tAnnot.get("firstreadSet");
				Set<Symbol> mayKilledSet = tAnnot.get("maykilled");
				Set<Symbol> deadSet = tAnnot.get("dead");
				Set<Symbol> checkSet = new HashSet<Symbol>();
				if( firstWriteSet != null ) {
					checkSet.addAll(firstWriteSet);
				}
				if( firstReadSet != null ) {
					checkSet.addAll(firstReadSet);
				}
				if( mayKilledSet != null ) {
					checkSet.addAll(mayKilledSet);
				}
				if( deadSet != null ) {
					checkSet.addAll(deadSet);
				}
				if( !checkSet.isEmpty() ) {
					//Find local symbol visible in the current procedure scope.
					//accessedSyms = AnalysisTools.getAccessedVariables(region, IRSymbolOnly);
					accessedSyms = new HashSet<Symbol>(sortedSet);
					if( accessedSyms != null ) {
						for( Symbol lSym : accessedSyms ) {
							List symbolInfo = new ArrayList(2);
							if( AnalysisTools.SymbolStatus.OrgSymbolFound(
									AnalysisTools.findOrgSymbol(lSym, region, true, null, symbolInfo, fCallList)) ) {
								Symbol gSym = (Symbol)symbolInfo.get(0);
								if( checkSet.contains(gSym) ) {
									g2lSymMap.put(gSym, lSym);
								}
							}
						}
					}
				}
				if( firstWriteSet != null ) {
					for( Symbol gsym : firstWriteSet ) {
						FunctionCall checkCall = new FunctionCall(new NameID("HI_check_write"));
						Expression hostVar = null;
						Symbol lsym = g2lSymMap.get(gsym);
						if( lsym == null ) {
							Tools.exit("[ERROR in ACC2CUDATranslator.extractComputeRegion()] can't find locally visible symbol " +
									"for the first-write symbol: " + gsym + "\nEnclosing procedure: " + 
									cProc.getSymbolName() + "\n");
						}
						if( lsym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
						} else {
							hostVar = new Identifier(lsym);
						}
						if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
							checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
									hostVar.clone()));
						} else {
							checkCall.addArgument(hostVar.clone());
						}
						checkCall.addArgument(new NameID("acc_device_nvidia"));
						checkCall.addArgument(new StringLiteral(hostVar.toString()));
						checkCall.addArgument(refName.clone());
						if( loopIndex != null ) {
							checkCall.addArgument(loopIndex.clone());
						} else {
							checkCall.addArgument(new NameID("INT_MIN"));
						}
						cStmt.addStatementBefore(region, new ExpressionStatement(checkCall));
					}
				}
				if( firstReadSet != null ) {
					for( Symbol gsym : firstReadSet ) {
						FunctionCall checkCall = new FunctionCall(new NameID("HI_check_read"));
						Expression hostVar = null;
						Symbol lsym = g2lSymMap.get(gsym);
						if( lsym == null ) {
							Tools.exit("[ERROR in ACC2CUDATranslator.extractComputeRegion()] can't find locally visible symbol " +
									"for the first-read symbol: " + gsym + "\nEnclosing procedure: " + 
									cProc.getSymbolName() + "\n");
						}
						if( cudaSharedROSet.contains(lsym) ) {
							continue;   //Skip generating check_read() call for scalar variable cached in
										//CUDA shared memory.
						}
						if( lsym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
						} else {
							hostVar = new Identifier(lsym);
						}
						if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
							checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
									hostVar.clone()));
						} else {
							checkCall.addArgument(hostVar.clone());
						}
						checkCall.addArgument(new NameID("acc_device_nvidia"));
						checkCall.addArgument(new StringLiteral(hostVar.toString()));
						checkCall.addArgument(refName.clone());
						if( loopIndex != null ) {
							checkCall.addArgument(loopIndex.clone());
						} else {
							checkCall.addArgument(new NameID("INT_MIN"));
						}
						cStmt.addStatementBefore(region, new ExpressionStatement(checkCall));
					}
				}
				if( mayKilledSet != null ) {
					for( Symbol gsym : mayKilledSet ) {
						FunctionCall checkCall = new FunctionCall(new NameID("HI_reset_status"));
						Expression hostVar = null;
						Symbol lsym = g2lSymMap.get(gsym);
						if( lsym == null ) {
							Tools.exit("[ERROR in ACC2CUDATranslator.extractComputeRegion()] can't find locally visible symbol " +
									"for the may-killed symbol: " + gsym + "\nEnclosing procedure: " + 
									cProc.getSymbolName() + "\n");
						}
						if( lsym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
						} else {
							hostVar = new Identifier(lsym);
						}
						if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
							checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
									hostVar.clone()));
						} else {
							checkCall.addArgument(hostVar.clone());
						}
						checkCall.addArgument(new NameID("acc_device_host"));
						checkCall.addArgument(new NameID("HI_maystale"));
						//checkCall.addArgument(new NameID("INT_MIN"));
						checkCall.addArgument(new NameID("DEFAULT_QUEUE"));
						cStmt.addStatementAfter(region, new ExpressionStatement(checkCall));
					}
				}
				if( deadSet != null ) {
					for( Symbol gsym : deadSet ) {
						FunctionCall checkCall = new FunctionCall(new NameID("HI_reset_status"));
						Expression hostVar = null;
						Symbol lsym = g2lSymMap.get(gsym);
						if( lsym == null ) {
							System.err.println("g2lSymMap" + g2lSymMap);
							Tools.exit("[ERROR in ACC2CUDATranslator.extractComputeRegion()] can't find locally visible symbol " +
									"for the dead symbol: " + gsym + "\nEnclosing procedure: " + 
									cProc.getSymbolName() + "\n");
						}
						if( lsym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
						} else {
							hostVar = new Identifier(lsym);
						}
						if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
							checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
									hostVar.clone()));
						} else {
							checkCall.addArgument(hostVar.clone());
						}
						checkCall.addArgument(new NameID("acc_device_host"));
						checkCall.addArgument(new NameID("HI_notstale"));
						//checkCall.addArgument(new NameID("INT_MIN"));
						checkCall.addArgument(new NameID("DEFAULT_QUEUE"));
						cStmt.addStatementAfter(region, new ExpressionStatement(checkCall));
					}
				}
			}
		}
		
		//////////////////////////////////////////////////////////////
		//Add GPU kernel configuration to the kernel function call. //
		//////////////////////////////////////////////////////////////
		//the dimension of the grid is the first argument, and then that of thread block comes.
		kernelConf.add((Identifier)dimGrid.clone()); 
		kernelConf.add((Identifier)dimBlock.clone());
		kernelConf.add(new IntegerLiteral(0));
		if( asyncID == null ) {
			kernelConf.add(null);
		} else {
			//FunctionCall getAsyncHandle = new FunctionCall(new NameID("HI_get_async_handle"));
			//getAsyncHandle.addArgument(asyncID.clone());
			//kernelConf.add(getAsyncHandle);
			kernelConf.add(asyncID.clone());
		}
		call_to_new_proc.setConfArguments(kernelConf);
		
		//////////////////////////////
		//Perform actual outlining. //
		//////////////////////////////
		region.swapWith(kernelCall_stmt);
		if( confRefStmt == region ) {
			confRefStmt = kernelCall_stmt;
			confRefParent = (CompoundStatement)confRefStmt.getParent();
		}
		CompoundStatement kernelRegion = null;
		if( region instanceof ForLoop ) {
			kernelRegion = new_proc.getBody();
			Statement dummyStmt = new AnnotationStatement();
			kernelRegion.addStatement(dummyStmt);
			dummyStmt.swapWith(region);
			for( Statement preS : preList ) {
				kernelRegion.addStatementBefore(region, preS);
			}
			int pSize = postList.size();
			for( int m=pSize-1; m>=0; m-- ) {
				kernelRegion.addStatementAfter(region, postList.get(m));
			}
		} else {
			kernelRegion = new_proc.getBody();
			region.swapWith(kernelRegion);
			kernelRegion = (CompoundStatement)region;
		}
		while ( !devProcStack.isEmpty() ) {
			Procedure tProc = devProcStack.pop();
			parentTrUnt.removeChild(tProc);
			kernelsTranslationUnit.addDeclaration(tProc);
		}

        // Put the new_proc in the separate file
        kernelsTranslationUnit.addDeclaration(new_proc);
		accKernelsList.add(new_proc.getSymbolName());
		
		//If typedef or other user-types are used, their definition should 
		//be added to the kernelsTranslationUnit.
		Set<Symbol> usedSymbols = SymbolTools.getSymbols(new_proc);
		usedSymbols.addAll(SymbolTools.getLocalSymbols(new_proc.getBody()));
		//Add enum symbols.
		Set<Symbol> tAccessedSymbols = SymbolTools.getAccessedSymbols(new_proc.getBody());
		for( Symbol tASym : tAccessedSymbols ) {
			Declaration tADecl = tASym.getDeclaration();
			if( tADecl instanceof Enumeration ) {
				usedSymbols.add(tASym);
			}
		}
		copyUserSpecifierDeclarations(kernelsTranslationUnit, kernelCall_stmt, usedSymbols, accHeaderDecl);

		/* put new_proc before the calling proc (avoids prototypes) */
        //((TranslationUnit) cProc.getParent()).addDeclarationBefore(cProc,new_proc);
		
		//FIXME: seq loop may still need _tid for Worker-Single Mode.
		//Disable the below condition.
		//if( !isSingleTask ) {
		if( true ) {
			/*
			 * Create expressions for calculating global GPU thread ID (_gtid), local thread ID (_tid), 
			 * global thread block ID (_bid), and thread block size (_bsize).
			 *     _bid = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
			 *     _bsize = blockDim.x * blockDim.y * blockDim.z;
			 *     _tid = threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y);
			 *     _gtid = _tid + (_bid * _bsize);
			 * [CAUTION] threadIdx.x, blockIdx.x, and blockDim.x are CUDA-built-in
			 * variables, and thus they don't have any declarations; Range Analysis 
			 * can not decide the types of these variables, and therefore, ignore these.
			 */ 
			VariableDeclarator gtid_declarator = new VariableDeclarator(new NameID("_gtid"));
			//gtid_declarator.setInitializer(new Initializer(biexp2));
			VariableDeclaration gtid_decl = new VariableDeclaration(Specifier.INT, gtid_declarator);
			gtid = new Identifier(gtid_declarator);
			BinaryExpression biexp1 = new BinaryExpression(new NameID("blockIdx.y"), 
					BinaryOperator.MULTIPLY, new NameID("gridDim.x"));
			BinaryExpression biexp2 = new BinaryExpression(new NameID("blockIdx.x"),
					BinaryOperator.ADD, biexp1);
			biexp1 = new BinaryExpression(new NameID("blockIdx.z"), BinaryOperator.MULTIPLY, 
					new BinaryExpression(new NameID("gridDim.x"), BinaryOperator.MULTIPLY, new NameID("gridDim.y")));
			biexp2 = new BinaryExpression(biexp2, BinaryOperator.ADD, biexp1);
			VariableDeclarator bid_declarator = new VariableDeclarator(new NameID("_bid"));
			//bid_declarator.setInitializer(new Initializer(biexp2));
			Declaration bid_decl = new VariableDeclaration(Specifier.INT, bid_declarator);
			bid = new Identifier(bid_declarator);
			ExpressionStatement bidInitStmt = new ExpressionStatement(new AssignmentExpression(bid.clone(), AssignmentOperator.NORMAL,
					biexp2));
			Statement gtidRefStmt = null;
			boolean bidIncluded = false;
			if( IRTools.containsExpression(kernelRegion, bid) || IRTools.containsExpression(kernelRegion, gtid) ) {
				bidIncluded = true;
				kernelRegion.addDeclaration(bid_decl);
				Statement last_decl_stmt = IRTools.getLastDeclarationStatement(kernelRegion);
				kernelRegion.addStatementAfter(last_decl_stmt, bidInitStmt);
				gtidRefStmt = bidInitStmt;
				TransformTools.replaceAll(kernelRegion, bid, bid);
			}

			biexp1 = new BinaryExpression(new BinaryExpression(new NameID("blockDim.x"), BinaryOperator.MULTIPLY, 
					new NameID("blockDim.y")), BinaryOperator.MULTIPLY, new NameID("blockDim.z"));
			VariableDeclarator bsize_declarator = new VariableDeclarator(new NameID("_bsize"));
			//bsize_declarator.setInitializer(new Initializer(biexp1));
			Declaration bsize_decl = new VariableDeclaration(Specifier.INT, bsize_declarator);
			bsize = new Identifier(bsize_declarator);
			ExpressionStatement bsizeInitStmt = new ExpressionStatement(new AssignmentExpression(bsize.clone(), AssignmentOperator.NORMAL,
					biexp1));
			if( IRTools.containsExpression(kernelRegion, bsize) || IRTools.containsExpression(kernelRegion, gtid) ) {
				kernelRegion.addDeclaration(bsize_decl);
				Statement last_decl_stmt = IRTools.getLastDeclarationStatement(kernelRegion);
				kernelRegion.addStatementAfter(last_decl_stmt, bsizeInitStmt);
				TransformTools.replaceAll(kernelRegion, bsize, bsize);
			}

			biexp1 = new BinaryExpression(new NameID("threadIdx.y"), 
					BinaryOperator.MULTIPLY, new NameID("blockDim.x"));
			biexp2 = new BinaryExpression(new NameID("threadIdx.x"),
					BinaryOperator.ADD, biexp1);
			biexp1 = new BinaryExpression(new BinaryExpression(new NameID("threadIdx.z"), BinaryOperator.MULTIPLY, 
					new NameID("blockDim.x")), BinaryOperator.MULTIPLY, new NameID("blockDim.y"));
			biexp1 = new BinaryExpression(new NameID("threadIdx.z"), BinaryOperator.MULTIPLY, 
					new BinaryExpression(new NameID("blockDim.x"), BinaryOperator.MULTIPLY, new NameID("blockDim.y")));
			biexp2 = new BinaryExpression(biexp2, BinaryOperator.ADD, biexp1);
			VariableDeclarator tid_declarator = new VariableDeclarator(new NameID("_tid"));
			//tid_declarator.setInitializer(new Initializer(biexp2));
			Declaration tid_decl = new VariableDeclaration(Specifier.INT, tid_declarator);
			tid = new Identifier(tid_declarator);
			ExpressionStatement tidInitStmt = new ExpressionStatement(new AssignmentExpression(tid.clone(), AssignmentOperator.NORMAL,
					biexp2));
			boolean tidIncluded = false;
			if( IRTools.containsExpression(kernelRegion, tid) || IRTools.containsExpression(kernelRegion, gtid)) {
				tidIncluded = true;
				kernelRegion.addDeclaration(tid_decl);
				Statement last_decl_stmt = IRTools.getLastDeclarationStatement(kernelRegion);
				kernelRegion.addStatementAfter(last_decl_stmt, tidInitStmt);
				if( gtidRefStmt == null ) {
					gtidRefStmt = tidInitStmt;
				}
				TransformTools.replaceAll(kernelRegion, tid, tid);
			}

			biexp1 = new BinaryExpression(bid.clone(), 
					BinaryOperator.MULTIPLY, bsize.clone());
			biexp2 = new BinaryExpression(tid.clone(), BinaryOperator.ADD, biexp1);
			ExpressionStatement gtidInitStmt = new ExpressionStatement(new AssignmentExpression(gtid.clone(), AssignmentOperator.NORMAL,
					biexp2));
			if( IRTools.containsExpression(kernelRegion, gtid) ) {
				if( !bidIncluded ) {
					kernelRegion.addDeclaration(bid_decl);
					Statement last_decl_stmt = IRTools.getLastDeclarationStatement(kernelRegion);
					kernelRegion.addStatementAfter(last_decl_stmt, bidInitStmt);
					if( gtidRefStmt == null ) {
						gtidRefStmt = bidInitStmt;
					}
				}
				if( !tidIncluded ) {
					kernelRegion.addDeclaration(tid_decl);
					Statement last_decl_stmt = IRTools.getLastDeclarationStatement(kernelRegion);
					kernelRegion.addStatementAfter(last_decl_stmt, tidInitStmt);
					if( gtidRefStmt == null ) {
						gtidRefStmt = tidInitStmt;
					}
				}
				kernelRegion.addDeclaration(gtid_decl);
				//Statement last_decl_stmt = IRTools.getLastDeclarationStatement(kernelRegion);
				kernelRegion.addStatementAfter(gtidRefStmt, gtidInitStmt);
				TransformTools.replaceAll(kernelRegion, gtid, gtid);
			}

			/////////////////////////////////////////////////////////////// 
			// Modify target region to be outlined as a kernel function  //
			//     - Remove the outmost OMP parallel for loop.           //
			//     - Add necessary GPU thread mapping statements.        //
			/////////////////////////////////////////////////////////////// 
		}
		
		Traversable parent = kernelCall_stmt.getParent();
		
		//Convert worksharing loops into if-statements. 
		CUDATranslationTools.worksharingLoopTransformation(cProc, kernelRegion, region, cRegionKind, defaultNumWorkers, opt_skipKernelLoopBoundChecking);
		
		//[DEBUG] We don't need this since each kernel call in the default queue will be followed by HI_synchronize() call.
/*		if( opt_forceSyncKernelCall ) {
            FunctionCall syncCall = new FunctionCall(new NameID("HI_synchronize"));
			if( parent instanceof CompoundStatement ) {
				((CompoundStatement)parent).addStatementAfter(kernelCall_stmt, new ExpressionStatement(syncCall));
			} else {
				Tools.exit(pass_name + "[Error in extractKernelRegion()] Kernel call statement (" +
						kernelCall_stmt + ") does not have a parent!");
			}
		}*/
		
		if( opt_addSafetyCheckingCode ) {
			/////////////////////////////////////////////
			// Add GPU global memory usage check code. //
			/////////////////////////////////////////////
			Expression MemCheckExp = new BinaryExpression((Identifier)gmemsize.clone(),
					BinaryOperator.COMPARE_GT, new NameID("MAX_GMSIZE")); 
			FunctionCall MemWarningCall = new FunctionCall(new NameID("printf"));
			StringLiteral warningMsg = new StringLiteral("[WARNING] size of allocated GPU global memory" +
			" (%u) exceeds the given limit (%d)\\n");
			MemWarningCall.addArgument(warningMsg);
			MemWarningCall.addArgument((Identifier)gmemsize.clone());
			MemWarningCall.addArgument( new NameID("MAX_GMSIZE"));
			IfStatement gMemCheckStmt = new IfStatement(MemCheckExp, 
					new ExpressionStatement(MemWarningCall));
			/////////////////////////////////////////////
			// Add GPU shared memory usage check code. //
			/////////////////////////////////////////////
			MemCheckExp = new BinaryExpression((Identifier)smemsize.clone(),
					BinaryOperator.COMPARE_GT, new NameID("MAX_SMSIZE")); 
			MemWarningCall = new FunctionCall(new NameID("printf"));
			warningMsg = new StringLiteral("[WARNING] size of allocated GPU shared memory" +
			" (%d) exceeds the given limit (%d)\\n");
			MemWarningCall.addArgument(warningMsg);
			MemWarningCall.addArgument((Identifier)smemsize.clone());
			MemWarningCall.addArgument( new NameID("MAX_SMSIZE"));
			IfStatement sMemCheckStmt = new IfStatement(MemCheckExp, 
					new ExpressionStatement(MemWarningCall));
			if( parent instanceof CompoundStatement ) {
				((CompoundStatement)parent).addStatementBefore(kernelCall_stmt, gMemCheckStmt);
				((CompoundStatement)parent).addStatementBefore(kernelCall_stmt, sMemCheckStmt);
			} else {
				Tools.exit(pass_name + "[Error in extractKernelRegion()] Kernel call statement (" +
						kernelCall_stmt + ") does not have a parent!");
			}
		}
		
		/*
		 * The original OpenACC annotation will be inserted into this new kernel function
		 * if the original region is compoundstatement.
		 * [CAUTION] Symbols in the annotation are not the ones used in
		 * the kernel region; they refer to original CPU symbols, but not
		 * GPU device symbols.
		 * [CAUTION] This insertion violates OpenACC semantics.
		 */
		if( region instanceof CompoundStatement ) {
			List<Annotation> annots = region.getAnnotations();
			if( annots != null ) {
				for(Annotation annot : annots) {
					kernelCall_stmt.annotate(annot);
				}
				region.removeAnnotations();
			}
		} else {
			//profile-related annotations and resilience annotations are moved to the new kernel function.
			List<Annotation> annots = region.getAnnotations();
			List<Annotation> newAnnots = new LinkedList<Annotation>();
			if( annots != null ) {
				for(Annotation annot : annots) {
					if( annot.containsKey("resilience") || annot.containsKey("profile") ) {
						kernelCall_stmt.annotate(annot);
					} else {
						newAnnots.add(annot);
					}
				}
				region.removeAnnotations();
				if( !newAnnots.isEmpty() ) {
					for( Annotation annot : newAnnots ) {
						region.annotate(annot);
					}
				}
			}
		}

		PrintTools.println("[extractComputeRegion() ends] current Procedure: " + cProc.getSymbolName()
				+ "\nOpenACC annotation: " + cAnnot +"\n", 1);

	}
	
	private void devProcCloning(Traversable at, TranslationUnit trUnt, String TrCnt, 
			Set<Symbol> callerParamSymSet, Stack<Procedure> devProcStack) {
		TranslationUnit currTrUnt = IRTools.getParentTranslationUnit(at);
		List<FunctionCall> funcList = IRTools.getFunctionCalls(at);
		if( funcList != null ) {
			for( FunctionCall fCall : funcList ) {
				if( StandardLibrary.contains(fCall) ) {
					continue;
				}
				Procedure c_proc = AnalysisTools.findProcedure(fCall);
				if( (c_proc == null) && (currTrUnt != trUnt) ) {
					List<Procedure> procList = IRTools.getProcedureList(program);
					for( Procedure tProc : procList ) {
						if( tProc.getSymbolName().equals(fCall.getName().toString()) ) {
							c_proc = tProc;
							break;
						}
					}
				}
				if( (c_proc != null) ) {
					FunctionCall new_fCall = null;
					String callContext = "";
					String offsetTails = ""; //used to differentiate offsets to texture variable argument. 
					Map<String, Procedure> devProcContextMap;
					if( !devProcMap.containsKey(c_proc) ) {
						devProcContextMap = new HashMap<String, Procedure>();
						devProcMap.put(c_proc, devProcContextMap);
					} else {
						devProcContextMap = devProcMap.get(c_proc);
					}
					//Generate calling context string.
					List<Symbol> argSymList = new ArrayList<Symbol>(fCall.getArguments().size());
					for( Expression argExp : fCall.getArguments() ) {
						//Step1: find argument symbol which is a parameber symbol of the calling procedure.
						Symbol argSym = SymbolTools.getSymbolOf(argExp);
						if( argSym == null ) {
							if( argExp instanceof BinaryExpression ) {
								//find argSym which is a parameter symbol of the calling procedure.
								Set<Symbol> sSet = SymbolTools.getAccessedSymbols(argExp);
								sSet.retainAll(callerParamSymSet);
								for( Symbol tSym : sSet ) {
									if( argSym == null ) {
										argSym = tSym;
									} else {
										if( SymbolTools.isPointer(tSym) || SymbolTools.isArray(tSym) ) {
											argSym = tSym;
											//FIXME: if multiple non-scalar parameter symbols exist, we can not
											//know which is correct symbol, but not checked here.
										}
									}
								}
							}
						}
						if( argSym instanceof AccessSymbol ) {
							argSym = null; 	//if argument is access expression, 
							//it should be considered as normal type.
						}
						if( !callerParamSymSet.contains(argSym) ) {
							argSym = null;
						}
						//Step2: find argument symbol type.
						argSymList.add(argSym);
						if( argSym == null ) {
							callContext += "0"; //normal type
						} else {
							if( pitchedSymMap.containsKey(argSym) ) {
								callContext += "1";
							} else if( textureSymMap.containsKey(argSym)) {
								callContext += "2";
								if( argExp instanceof BinaryExpression ) {
									BinaryExpression tExp = (BinaryExpression)argExp;
									Symbol tSym = SymbolTools.getSymbolOf(tExp.getLHS());
									Expression offset = null;
									boolean foundArgSym = false;
									if( argSym == tSym ) {
										offset = tExp.getRHS();
										foundArgSym = true;
									} else {
										tSym = SymbolTools.getSymbolOf(tExp.getRHS());
										if( argSym == tSym ) {
											offset = tExp.getLHS();
											foundArgSym = true;
										}
									}
									boolean tooComplex = false;
									if( !foundArgSym ) {
										tooComplex = true;
									} else {
										//DEBUG: current implementation handles only simple binary expressions 
										//such as (a + 1).
										if( offset instanceof IntegerLiteral ) {
											BinaryOperator bOp = tExp.getOperator();
											if( bOp.equals(BinaryOperator.ADD)) {
												textureOffsetMap.put(argSym, offset);
												offsetTails += offset.toString();
											} else if( bOp.equals(BinaryOperator.SUBTRACT) ) {
												long iVal = ((IntegerLiteral)offset).getValue();
												offset = new IntegerLiteral(-1*iVal);
												offsetTails += offset.toString();
												textureOffsetMap.put(argSym, offset);
											} else {
												tooComplex = true;
											}
										} else {
											tooComplex = true;
										}
									}
									if( tooComplex ) {
										Tools.exit("[ERROR in ACC2CUDATranslator.devProcCloning()] texture symbol (" + 
												argSym.getSymbolName() + ") is passed to a device function (" + c_proc.getSymbolName() +
												") in a complex argument expression (" + argExp + "); the current implemenation can not " +
												"handle this. Either manually inline this device function, or do not cache the variable on " +
										"the texture cache.");
									}
								}
							} else if( constantSymMap.containsKey(argSym)) {
								callContext += "3";
							} else { //normal type
								callContext += "0";
							}
						}
					}
					if( !offsetTails.equals("") ) {
						callContext += offsetTails;
					}
					TranslationUnit tu = (TranslationUnit)c_proc.getParent();
					if( !devProcContextMap.containsKey(callContext) ) {
						boolean cloneProcedure = true;
						//FIXME: the original procedure should not be used for device procedure if it is used kernels across different
						//translation units and calling contexts are different; for now, cloning is always enforced conservatively.
						/*						if( devProcContextMap.isEmpty() && (tu == trUnt) ) {
							ACCAnnotation rAnnot = c_proc.getAnnotation(ACCAnnotation.class, "routine");
							if (rAnnot != null ) {
								if (rAnnot.containsKey("nohost")) {
									cloneProcedure = false;
								}
							}
						}*/
						List<VariableDeclaration> oldParamList = 
							(List<VariableDeclaration>)c_proc.getParameters();
						CompoundStatement body;
						String new_proc_name;
						Procedure new_proc;
						if( cloneProcedure ) {
							/////////////////////////////
							// Clone current procedure //
							/////////////////////////////
							//FIXME: if the current procedure has routine bind clause, and if the argument is different
							//from the procedure name, below transformation should be skipped, but the current kernel
							//translation pass may not recognize the other implementation correctly. To fix this, the
							//other implementation should not have any global variable accessed without parameter passing.
							//If a procedure has a static variable, it should not be cloned.
							//Set<Symbol> symSet = SymbolTools.getVariableSymbols(c_proc.getBody());
							Set<Symbol> symSet = SymbolTools.getLocalSymbols(c_proc.getBody());
							Set<Symbol> staticSyms = AnalysisTools.getStaticVariables(symSet);
							if( !staticSyms.isEmpty() ) {
								Tools.exit("[ERROR in GlobalVariableParameterization] if a procedure has static variables," +
										"it can not be cloned; for correct transformation, either \"disableStatic2GlobalConversion\" " +
										"option should be disabled or static variables should be manually promoted to global ones.\n" +
										"Procedure name: " + c_proc.getSymbolName() + "\n");
							}
							List<Specifier> return_types = c_proc.getReturnType();
							return_types.add(0, CUDASpecifier.CUDA_DEVICE);
							body = (CompoundStatement)c_proc.getBody().clone();
							new_proc_name = "dev__" + c_proc.getSymbolName() + "_TU" + TrCnt + "_CT" + devProcContextMap.size();
							NameID new_procID = new NameID(new_proc_name);
							new_proc = new Procedure(return_types,
									new ProcedureDeclarator(new_procID,
											new LinkedList()), body);	
							//////////////////////////////////////////////////////////
							// Create a new function call for the cloned procedure. //
							//////////////////////////////////////////////////////////
							if( fCall != null ) {
								new_fCall = new FunctionCall(new NameID(new_proc_name));
								fCall.swapWith(new_fCall);
							}
						} else {
							body = c_proc.getBody();
							new_proc_name = c_proc.getSymbolName();
							new_proc = c_proc;
							new_fCall = fCall;
						}
						devProcContextMap.put(callContext, new_proc);
						///////////////////////////////////////////////
						// Update function parameters and arguments. //
						///////////////////////////////////////////////
						Set<Symbol> newCallerParamSymSet = new HashSet<Symbol>();
						List<Expression> oldArgList = (List<Expression>)fCall.getArguments();
						List<VariableDeclaration> extraParamList = new LinkedList<VariableDeclaration>();
						List<Expression> extraArgList = new LinkedList<Expression>();
						int oldParamListSize = oldParamList.size();
						if( oldParamListSize == 1 ) {
							Object obj = oldParamList.get(0);
							String paramS = obj.toString();
							// Remove any leading or trailing whitespace.
							paramS = paramS.trim();
							if( paramS.equals(Specifier.VOID.toString()) ) {
								oldParamListSize = 0;
							}
						}
						if( oldParamListSize > 0 ) {
							int i=0;
							for( VariableDeclaration param : oldParamList ) {
								Symbol param_declarator = (Symbol)param.getDeclarator(0);
								List<Specifier> typeSpecs = new ArrayList<Specifier>();
								typeSpecs.addAll(param.getSpecifiers());
								/*								if( typeSpecs.remove(Specifier.RESTRICT) ) {
									param.getSpecifiers().remove(Specifier.RESTRICT);
									param.getSpecifiers().add(CUDASpecifier.RESTRICT);
									typeSpecs.add(CUDASpecifier.RESTRICT);
								}*/
								VariableDeclaration cloned_decl = (VariableDeclaration)param.clone();
								Identifier paramID = new Identifier(param_declarator);
								Symbol cloned_param_declr = (Symbol) cloned_decl.getDeclarator(0);
								if( cloneProcedure ) {
									newCallerParamSymSet.add(cloned_param_declr);
								} else {
									newCallerParamSymSet.add(param_declarator);
								}
								Symbol argSym = argSymList.get(i);
								if( !callContext.equals("") ) {
									char symType = callContext.charAt(i);
									if( symType == '1' ) { //pitched malloc
										String pitchVarName = "pitch__" + cloned_param_declr.getSymbolName();
										VariableDeclarator pitch_declarator = new VariableDeclarator(new NameID(pitchVarName));
										VariableDeclaration pitch_decl = new VariableDeclaration(OpenACCSpecifier.SIZE_T, 
												pitch_declarator);
										Identifier paramPitchID = new Identifier(pitch_declarator);
										extraParamList.add(pitch_decl);
										Identifier argPitchID = new Identifier(pitchedSymMap.get(argSym));
										extraArgList.add(argPitchID);
										if( cloneProcedure ) {
											pitchedSymMap.put(cloned_param_declr, pitch_declarator);
										} else {
											pitchedSymMap.put(param_declarator, pitch_declarator);
										}
										/* 
										 * If MallocPitch is used to allocate 2 dimensional array, gpu_a,
										 * replace array access expression with pointer access expression with pitch
										 * Ex: gpu__a[i][k] => *((float *)((char *)gpu__a + i * pitch__a) + k)
										 */
										if( cloneProcedure ) {
											CUDATranslationTools.pitchedAccessConv(param_declarator, 
													new Identifier(cloned_param_declr), typeSpecs,
													paramPitchID, body);
										} else {
											CUDATranslationTools.pitchedAccessConv(param_declarator, 
													new Identifier(param_declarator), typeSpecs,
													paramPitchID, body);
										}
									} else if( symType == '2' ) { //texture memory
										Symbol textureSym = textureSymMap.get(argSym);
										if( cloneProcedure ) {
											textureSymMap.put(cloned_param_declr, textureSym);
										} else {
											textureSymMap.put(param_declarator, textureSym);
										}
										Expression offset = textureOffsetMap.get(argSym);
										CUDATranslationTools.textureConv(param_declarator, new Identifier(textureSym), body, offset);
									} else if( symType == '3' ) { //constant memory
										//Symbol constantSym = constantSymMap.get(argSym);
										if( cloneProcedure ) {
											constantSymMap.put(cloned_param_declr, cloned_param_declr);
										} else {
											constantSymMap.put(param_declarator, param_declarator);
										}
										//DEBUG: above assumes address passing of constant symbol is allowed.
										//Then, we don't need additional replacement.
									}
								}
								if( cloneProcedure ) {
									new_proc.addDeclaration(cloned_decl);
									//Instead of cloning, move the argument to the new function call.
									//new_fCall.addArgument(oldArgList.get(i).clone());
									Expression argExp = oldArgList.get(i);
									//fCall.removeChild(argExp);
									//Expression.removeChild() is not allowed; instead use swapping.
									Expression dummyExp = new NameID("dummyArg");
									new_fCall.addArgument(dummyExp);
									dummyExp.swapWith(argExp);
									Identifier cloned_ID = new Identifier(cloned_param_declr);
									TransformTools.replaceAll((Traversable) body, paramID, cloned_ID);
								}
								i++;
							}
							//Add extra parameters and arguments.
							i=0;
							for( VariableDeclaration extParam : extraParamList ) {
								new_proc.addDeclaration(extParam);
								new_fCall.addArgument(extraArgList.get(i));
								i++;
							}
						}

						if( cloneProcedure ) {
							//Replace consant array symbols with the compiler-generated global constant array symbol.
							Set<Symbol> pAccessedSymbols = AnalysisTools.getAccessedVariables(body, true);
							Set<Symbol> kernelGlobalSymbols = kernelsTranslationUnit.getSymbols();
							for(Symbol pSym : pAccessedSymbols) {
								if( SymbolTools.isGlobal(pSym) ) {
									if( SymbolTools.isArray(pSym) && !SymbolTools.isPointer(pSym) 
											&& pSym.getTypeSpecifiers().contains(Specifier.CONST) ) {
										String symNameBase = null;
										if( pSym instanceof AccessSymbol) {
											symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)pSym);
										} else {
											symNameBase = pSym.getSymbolName();
										}
										String constVarName = "const__" + symNameBase;
										Symbol IRSym = pSym;
										if( pSym instanceof PseudoSymbol ) {
											IRSym = ((PseudoSymbol)pSym).getIRSymbol();
										}
										if( !SymbolTools.isGlobal(IRSym) ) {
											constVarName += "__" + c_proc.getSymbolName();
										}
										Symbol constSym = AnalysisTools.findsSymbol(kernelGlobalSymbols, constVarName);
										if( constSym == null ) {
											//constant symbol should have been created by either handleDataClause() or handleUpdate().
											Tools.exit("[ERROR in ACC2CUDATranslation.devProcCloning()] Can't find __constant variable (" + constVarName + 
													") corresponding to the host variable, " + pSym.getSymbolName() + "; exit the program!\nEnclosing procedure: " + 
													c_proc.getSymbolName() + "\n");
										}
										Identifier constVar = new Identifier(constSym);
										// Replace the instance of shared variable with the new gpu_var.
										if( pSym instanceof AccessSymbol ) {
											TransformTools.replaceAccessExpressions(body, (AccessSymbol)pSym, constVar);
										} else {
											TransformTools.replaceAll(body, new Identifier(pSym), constVar);
										}
									}

								}
							}
						}

						if( cloneProcedure ) {
							////////////////////////////
							// Add the new procedure. //
							////////////////////////////
							trUnt.addDeclaration(new_proc);
							devProcStack.push(new_proc);
							////////////////////////////////////////////////////////////////////////
							//If the current procedure has annotations, copy them to the new one. //
							////////////////////////////////////////////////////////////////////////
							List<Annotation> cAnnotList = c_proc.getAnnotations();
							if( (cAnnotList != null) && (!cAnnotList.isEmpty()) ) {
								for( Annotation cAn : cAnnotList ) {
									//OpenMP annotations are not added to the cloned device function.
									if( !(cAn instanceof OmpAnnotation) ) {
										new_proc.annotate(cAn.clone());
									}
								}
							}
							///////////////////////////////////////////////////
							// Add routine directive for this new procedure. //
							///////////////////////////////////////////////////
							ACCAnnotation rAnnot = new_proc.getAnnotation(ACCAnnotation.class, "routine");
							if( rAnnot == null ) {
								rAnnot = new ACCAnnotation("routine", "_directive");
								new_proc.annotate(rAnnot);
							}
							rAnnot.put("nohost", "_clause");

							//////////////////////////////////////////////////////////////////
							//If declaration statement exists for the original procedure,   //
							//create a new declaration statement for the new procedure too. //
							//////////////////////////////////////////////////////////////////
							FlatIterator Fiter = new FlatIterator(program);
							while (Fiter.hasNext())
							{
								TranslationUnit cTu = (TranslationUnit)Fiter.next();
								DFIterator<ProcedureDeclarator> iter = new DFIterator<ProcedureDeclarator>(cTu, ProcedureDeclarator.class);
								iter.pruneOn(ProcedureDeclarator.class);
								iter.pruneOn(Procedure.class);
								iter.pruneOn(Statement.class);
								for (;;)
								{
									ProcedureDeclarator procDeclr = null;

									try {
										procDeclr = (ProcedureDeclarator)iter.next();
									} catch (NoSuchElementException e) {
										break;
									}
									if( procDeclr.getID().equals(c_proc.getName()) ) {
										Traversable parent = procDeclr.getParent();
										if( parent instanceof VariableDeclaration ) {
											//Found function declaration.
											VariableDeclaration procDecl = (VariableDeclaration)parent;
											//Create a new function declaration.
											VariableDeclaration newProcDecl = 
												new VariableDeclaration(procDecl.getSpecifiers(), new_proc.getDeclarator().clone());
											//Insert the new function declaration.
											if( !AnalysisTools.isInHeaderFile(procDecl, cTu) ) {
												cTu.addDeclarationAfter(procDecl, newProcDecl);
											} else {
												Procedure firstProc = AnalysisTools.findFirstProcedure(cTu);
												if( firstProc == null ) {
													cTu.addDeclaration(newProcDecl);
												} else {
													cTu.addDeclarationBefore(firstProc, newProcDecl);
												}
											}
											////////////////////////////////////////////////////////////////////////////////////
											//If the current procedure declaration has annotations, copy them to the new one. //
											////////////////////////////////////////////////////////////////////////////////////
											cAnnotList = procDecl.getAnnotations();
											if( (cAnnotList != null) && (!cAnnotList.isEmpty()) ) {
												for( Annotation cAn : cAnnotList ) {
													newProcDecl.annotate(cAn.clone());
												}
											}
											rAnnot = newProcDecl.getAnnotation(ACCAnnotation.class, "routine");
											if( rAnnot == null ) {
												rAnnot = new ACCAnnotation("routine", "_directive");
												newProcDecl.annotate(rAnnot);
											}
											rAnnot.put("nohost", "_clause");

											ACCAnalysis.updateSymbolsInACCAnnotations(newProcDecl, null);
											break;
										}
									}
								}
							}
							/////////////////////////////////////////////////////////////////////////
							// Update the newly cloned procedure:                                  //
							//     1) Update symbols in the new procedure, including symbols       //
							//        in ACCAnnoations.                                            //
							/////////////////////////////////////////////////////////////////////////
							//[DEBUG] the new device function will be moved to the kernelTranslationUnit at the end of 
							//the ACC2CUDATranslator, and thus below checking will complain missing declation if
							//a device function accesses kernel-file-global-constant array.
							SymbolTools.linkSymbol(new_proc, 0);
							ACCAnalysis.updateSymbolsInACCAnnotations(new_proc, null);
						}

						////////////////////////////////////////////////////////////////////////
						// Check functions called in the current device function recursively. //
						////////////////////////////////////////////////////////////////////////
						devProcCloning(body, trUnt, TrCnt, newCallerParamSymSet, devProcStack);
					} else {
						//cloned device procedure already exist; just change function calls.
						Procedure new_proc = devProcContextMap.get(callContext);
						String new_proc_name = new_proc.getSymbolName();
						//////////////////////////////////////////////////////////
						// Create a new function call for the cloned procedure. //
						//////////////////////////////////////////////////////////
						if( fCall != null ) {
							new_fCall = new FunctionCall(new NameID(new_proc_name));
							List<Expression> argList = (List<Expression>)fCall.getArguments();
							if( argList != null ) {
								for( Expression exp : argList ) {
									//Instead of cloning, move the argument to the new function call.
									//new_fCall.addArgument(exp.clone());
									//fCall.removeChild(exp);
									//Expression.removeChild() is not allowed; instead use swapping.
									Expression dummyExp = new NameID("dummyArg");
									new_fCall.addArgument(dummyExp);
									dummyExp.swapWith(exp);
								}
							}
							fCall.swapWith(new_fCall);
							if(!callContext.equals("")) {
								int i=0;
								for( Symbol argSym : argSymList ) {
									char symType = callContext.charAt(i);
									if( symType == '1' ) { //pitched malloc
										Identifier argPitchID = new Identifier(pitchedSymMap.get(argSym));
										new_fCall.addArgument(argPitchID);
									} else if( symType == '2' ) { //texture memory
										//FIXME: if offest exists, that should be passed too.
									} else if( symType == '3' ) { //constant memory
										//DEBUG: above assumes address passing of constant symbol is allowed.
										//Then, we don't need additional replacement.
									}
									i++;
								}
							}
						}
					}
				}
			}
		}
	}

    /**
     * Apply stripmining to fit the iteration size of a worksharing loop into the specified gang/worker configuration.
     *
     * @param cAnnot
     * @param cRegionKind
     */
    protected ForLoop worksharingLoopStripmining(Procedure cProc, ACCAnnotation cAnnot, String cRegionKind) {
        PrintTools.println("[worksharingLoopStripmining() begins]", 2);
        Statement region = (Statement)cAnnot.getAnnotatable();
        ForLoop newLoop = null;
        //////////////////////////////////////////////////////////////////////////////
        // Original kernels loop type 1:                                            //
        //     #pragma acc kernels loop gang(num_gangs), worker(num_workers)        //
        //     for( k = LB; k <= UB; k++ ) { }                                      //
        // Cyclic-unrolled loop:                                                    //
        //     for( i = 0; i < (num_gangs * num_workers); i++ ) {                   //
        //          int temp_i = num_gangs * num_workers;                           //
        //          for( k = i+LB; k <= UB; k += temp_i ) { }                       //
        //     }                                                                    //
        //////////////////////////////////////////////////////////////////////////////
        // Original kernels loop type 2:                                            //
        //     #pragma acc kernels loop gang(num_gangs)                             //
        //     for( k = LB; k <= UB; k++ ) { }                                      //
        // Cyclic-unrolled loop:                                                    //
        //     for( i = 0; i < num_gangs; i++ ) {                                   //
        //          for( k = i+LB; k <= UB; k += num_gangs ) { }                    //
        //     }                                                                    //
        //////////////////////////////////////////////////////////////////////////////
        // Original kernels loop type 3:                                            //
        //     #pragma acc kernels loop worker(num_workers)                         //
        //     for( k = LB; k <= UB; k++ ) { }                                      //
        // Cyclic-unrolled loop:                                                    //
        //     for( i = 0; i < num_workers; i++ ) {                                 //
        //          for( k = i+LB; k <= UB; k += num_workers ) { }                  //
        //     }                                                                    //
        //////////////////////////////////////////////////////////////////////////////
        // Original parallel region type 1:                                         //
        //     #pragma acc parallel loop gang, worker, num_gang(num_gangs),         //
        //     num_worker(num_workers)                                              //
        //     for( k = LB; k <= UB; k++ ) { }                                      //
        // Cyclic-unrolled loop:                                                    //
        //     for( i = 0; i < (num_gangs * num_workers); i++ ) {                   //
        //          int temp_i = num_gangs * num_workers;                           //
        //          for( k = i+LB; k <= UB; k += temp_i ) { }                       //
        //     }                                                                    //
        //////////////////////////////////////////////////////////////////////////////
        // Original parallel region type 2:                                         //
        //////////////////////////////////////////////////////////////////////////////
        // Due to ACCLoopDirectivePreprocessor.CheckWorkSharingLoopNestingOrder(),  //
        // no nested gang loops/worker loops exist in the Parallel region.          //
        //////////////////////////////////////////////////////////////////////////////
        List<ACCAnnotation> loopAnnots = AnalysisTools.ipCollectPragmas(region, ACCAnnotation.class, "gang", null);
        if( loopAnnots != null ) {
            for( ACCAnnotation lAnnot : loopAnnots ) {
                ForLoop ploop = (ForLoop)lAnnot.getAnnotatable();
                ACCAnnotation iAnnot = ploop.getAnnotation(ACCAnnotation.class, "iterspace");
                Expression iterspace = iAnnot.get("iterspace"); //each gang loop contains iterspace internal clause.
                Expression nestLevel = iAnnot.get("gangdim");
                Expression num_gangs = null;
                Expression num_workers = null;
                if( cRegionKind.equals("kernels") ) {
                    num_gangs = lAnnot.get("gang");
                    num_workers = lAnnot.get("worker");
                } else {
                    num_gangs = cAnnot.get("num_gangs");
                    num_workers = cAnnot.get("num_workers");
                }
                if( num_workers == null ) {
                    num_workers = new IntegerLiteral(defaultNumWorkers);
                }
                if( num_gangs == null ) {
                    Tools.exit("[ERROR in ACC2GPUTranslator.worksharingLoopUnrolling()] number of gangs for the following worksharing" +
                            " loop is not specified; exit!\n" + "OpenACC annotation: " + cAnnot + "\nEnclosing Procedure: " +
                            cProc.getSymbolName() + "\n");
                }
                boolean containsWorkerClause = false;
                if( ploop.containsAnnotation(ACCAnnotation.class, "worker") ) {
                    containsWorkerClause = true;
                }
                Expression tItrSize = null;
                if( containsWorkerClause ) {
                    if( num_gangs instanceof Typecast ) {
                        Expression tExp = ((Typecast)num_gangs).getExpression();
                        if( tExp instanceof FunctionCall ) {
                            if( ((FunctionCall)tExp).getName().toString().equals("ceil") ) {
                                tExp = ((FunctionCall)tExp).getArgument(0);
                            }
                        }
                        if( (tExp instanceof BinaryExpression) ) {
                            BinaryExpression tBExp = ((BinaryExpression)tExp);
                            if( tBExp.getOperator() == BinaryOperator.DIVIDE ) {
                                Expression LHS = tBExp.getLHS();
                                if( LHS instanceof Typecast ) {
                                    LHS = ((Typecast)LHS).getExpression();
                                }
                                Expression RHS = tBExp.getRHS();
                                if( RHS instanceof Typecast ) {
                                    RHS = ((Typecast)RHS).getExpression();
                                }
                                if( RHS instanceof FloatLiteral ) {
                                    RHS = new IntegerLiteral((long)((FloatLiteral)RHS).getValue());
                                }
                                if( RHS.equals(num_workers) ) {
                                    tItrSize = LHS.clone();
                                }
                            }
                        }
                    }
                    if( tItrSize == null ) {
                        tItrSize = Symbolic.simplify(Symbolic.multiply(num_gangs, num_workers));
                    }
                } else {
                    tItrSize = num_gangs;
                }
                iterspace = Symbolic.simplify(iterspace);
                tItrSize = Symbolic.simplify(tItrSize);
                if( tItrSize.equals(iterspace) ) {
                    continue; //we don't need to unroll this loop.
                } else if( (tItrSize instanceof IntegerLiteral) && (iterspace instanceof IntegerLiteral) ) {
                    if( ((IntegerLiteral)tItrSize).getValue() >= ((IntegerLiteral)iterspace).getValue() ) {
                        continue;
                    }
                }
                long suffix = 500;
                if( nestLevel instanceof IntegerLiteral ) {
                    long level = ((IntegerLiteral)nestLevel).getValue();
                    suffix = 500 + level;
                    if( level == 1 ) {
                        if( containsWorkerClause ) {
                            tItrSize = new BinaryExpression(new NameID("gridDim.x"), BinaryOperator.MULTIPLY, num_workers.clone());
                        } else {
                            tItrSize = new NameID("gridDim.x");
                        }
                    } else if( level == 2 ) {
                        if( containsWorkerClause ) {
                            tItrSize = new BinaryExpression(new NameID("gridDim.y"), BinaryOperator.MULTIPLY, num_workers.clone());
                        } else {
                            tItrSize = new NameID("gridDim.y");
                        }
                    } else if( level == 3 ) {
                        if( containsWorkerClause ) {
                            tItrSize = new BinaryExpression(new NameID("gridDim.z"), BinaryOperator.MULTIPLY, num_workers.clone());
                        } else {
                            tItrSize = new NameID("gridDim.z");
                        }
                    }
                }
                CompoundStatement targetRegion = null;
                if( region instanceof CompoundStatement ) {
                    targetRegion = (CompoundStatement)region;
                }
                boolean targetLoopChanged = false;
                if( ploop == region ) {
                    targetLoopChanged = true;
                }
                ForLoop wLoop = TransformTools.stripmining(ploop, tItrSize, suffix, targetRegion);
                if( targetLoopChanged ) {
                    newLoop = wLoop;
                }
            }
        }
        loopAnnots = AnalysisTools.ipCollectPragmas(region, ACCAnnotation.class, "worker", null);
        if( loopAnnots != null ) {
            for( ACCAnnotation lAnnot : loopAnnots ) {
                ForLoop ploop = (ForLoop)lAnnot.getAnnotatable();
                boolean containsGangClause = false;
                if( ploop.containsAnnotation(ACCAnnotation.class, "gang") ) {
                    containsGangClause = true;
                }
                if( containsGangClause ) {
                    continue;
                }
                ACCAnnotation iAnnot = ploop.getAnnotation(ACCAnnotation.class, "workerdim");
                Expression nestLevel = iAnnot.get("workerdim");
                Expression num_workers = null;
                if( cRegionKind.equals("kernels") ) {
                    num_workers = lAnnot.get("worker");
                } else {
                    num_workers = cAnnot.get("num_workers");
                }
                if( num_workers == null ) {
                    num_workers = new IntegerLiteral(defaultNumWorkers);
                }
                //Calculate iteration size of the pure worker loop.
                Expression lb = LoopTools.getLowerBoundExpression(ploop);
                Expression ub = LoopTools.getUpperBoundExpression(ploop);
                Expression tItrSize = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
                num_workers = Symbolic.simplify(num_workers);
                tItrSize = Symbolic.simplify(tItrSize);
                if( tItrSize.equals(num_workers) ) {
                    continue; //we don't need to unroll this loop.
                }
                long suffix = 200;
                if( nestLevel instanceof IntegerLiteral ) {
                    suffix = 200 + ((IntegerLiteral)nestLevel).getValue();
                }
                CompoundStatement targetRegion = null;
                if( region instanceof CompoundStatement ) {
                    targetRegion = (CompoundStatement)region;
                }
                boolean targetLoopChanged = false;
                if( ploop == region ) {
                    targetLoopChanged = true;
                }
                ForLoop wLoop = TransformTools.stripmining(ploop, num_workers, suffix, targetRegion);
                if( targetLoopChanged ) {
                    newLoop = wLoop;
                }
            }
        }
        //DEBUG: do we still need this? No.
/*        if( firstMainStmt == region ) {
            //////////////////////////////////////////////////////////////////////////////////////////
            //Current region, which is pointed by firstMainStmt, will be moved into the if-statement//
            //, which is generated by this transformation. Therefore, we have to set firstMainStmt  //
            //to a new first statement in the main body.                                            //
            //////////////////////////////////////////////////////////////////////////////////////////
            firstMainStmt = IRTools.getFirstNonDeclarationStatement(main.getBody());
        }*/
        PrintTools.println("[worksharingLoopStripmining() ends]", 2);
        return newLoop;
    }

    protected void handleAtomicAnnots(List<ACCAnnotation> atomicAnnots)
    {
        for(ACCAnnotation annot : atomicAnnots)
        {
            Annotatable at = annot.getAnnotatable();

            if(at instanceof ExpressionStatement)
            {
                ExpressionStatement atomicStmt = (ExpressionStatement)at;
                Expression atomicExpr = atomicStmt.getExpression();

                if(atomicExpr instanceof UnaryExpression)
                {
                    UnaryExpression expr = (UnaryExpression)atomicExpr;
                    if((expr.getOperator() == UnaryOperator.POST_DECREMENT) ||
                            (expr.getOperator() == UnaryOperator.PRE_DECREMENT) ||
                            (expr.getOperator() == UnaryOperator.POST_INCREMENT) ||
                            (expr.getOperator() == UnaryOperator.PRE_INCREMENT))
                    {
                        if(expr.getExpression() instanceof Identifier)
                        {
                            if(annot.containsKey("atomic_var"))
                            {
                                Set<Identifier> atomicVar = annot.get("atomic_var");
                                atomicVar.add((Identifier) expr.getExpression());
                            }
                            else
                            {
                                Set<Identifier> atomicVar = new HashSet<Identifier>();
                                atomicVar.add((Identifier)expr.getExpression());
                                annot.put("atomic_var", atomicVar);
                            }
                            int val = 0;
                            UnaryOperator op = expr.getOperator();

                            if(op == UnaryOperator.POST_DECREMENT)
                                val = -1;
                            else if(op == UnaryOperator.PRE_DECREMENT)
                                val = -1;
                            else if(op == UnaryOperator.POST_INCREMENT)
                                val = 1;
                            else if(op == UnaryOperator.PRE_INCREMENT)
                                val = 1;

                            FunctionCall atomicCall = new FunctionCall(new NameID("atomicAdd"),
                                    new UnaryExpression(UnaryOperator.ADDRESS_OF, expr.getExpression().clone()),
                                    new IntegerLiteral(val));
                            CompoundStatement parentStmt = (CompoundStatement) atomicStmt.getParent();
                            ExpressionStatement atomicCallStmt =  new ExpressionStatement(atomicCall);
                            atomicCallStmt.annotate(annot.clone());
                            //PrintTools.println(atomicCallStmt.toString(),0);
                            parentStmt.addStatementAfter(atomicStmt, atomicCallStmt);
                            parentStmt.removeStatement(atomicStmt);
                        }
                        else
                        {
                            Tools.exit("Atomic operation " + atomicExpr + " is not supported.");
                        }
                    }
                    else
                    {
                        Tools.exit("Atomic operation " + atomicExpr + " is not supported.");
                    }
                }
                // x = ...
                // x binop= ...
                else if(atomicExpr instanceof AssignmentExpression)
                {
                    AssignmentExpression assignExpr = (AssignmentExpression)atomicExpr;
                    Identifier atomicVar = (Identifier)assignExpr.getLHS();

                    // x = ...
                    if(assignExpr.getOperator() == AssignmentOperator.NORMAL)
                    {
                        // x = x binop expr
                        // x = expr binop x
                        if(IRTools.containsExpression(assignExpr.getRHS(), assignExpr.getLHS()))
                        {
                            if(!(assignExpr.getRHS() instanceof BinaryExpression))
                            {
                                Tools.exit("Invalid atomic operation: " + assignExpr);
                            }

                            BinaryExpression rhsExpr = (BinaryExpression)assignExpr.getRHS();
                            BinaryOperator atomicOp = rhsExpr.getOperator();
                            Expression valExpr = null;

                            if(rhsExpr.getRHS().equals(atomicVar))
                            {
                                valExpr = rhsExpr.getLHS();
                            }
                            else if(rhsExpr.getLHS().equals(atomicVar))
                            {
                                valExpr = rhsExpr.getRHS();
                            }
                            else
                            {
                                Tools.exit("Invalid atomic operation: " + assignExpr);
                            }

                            if(annot.containsKey("atomic_var"))
                            {
                                Set<Identifier> atomicVarSet = annot.get("atomic_var");
                                atomicVarSet.add(atomicVar);
                            }
                            else
                            {
                                Set<Identifier> atomicVarSet = new HashSet<Identifier>();
                                atomicVarSet.add(atomicVar);
                                annot.put("atomic_var", atomicVarSet);
                            }

                            NameID atomicFunc = null;

                            if(atomicOp == BinaryOperator.ADD)
                            {
                                atomicFunc = new NameID("atomicAdd");
                            }
                            else if(atomicOp == BinaryOperator.SUBTRACT)
                            {
                                // x = expr - x;
                                if(valExpr.equals(rhsExpr.getLHS()))
                                {
                                	atomicFunc = new NameID("atomicSubRHS");
                                	Tools.exit("Atomic RHS-subtraction operation is not supported\nAtomic annotation: " 
                                			+ annot + "\n" + AnalysisTools.getEnclosingAnnotationContext(annot));
                                }
                                // x = x - expr
                                else
                                {
                                    //atomicFunc = new NameID("atomicAdd");
                                    //valExpr = new UnaryExpression(UnaryOperator.MINUS, valExpr.clone());
                                    atomicFunc = new NameID("atomicSub");
                                }
                            }
                            else if(atomicOp == BinaryOperator.BITWISE_AND)
                            {
                                atomicFunc = new NameID("atomicAnd");
                            }
                            else if(atomicOp == BinaryOperator.BITWISE_INCLUSIVE_OR)
                            {
                                atomicFunc = new NameID("atomicOr");
                            }
                            else if(atomicOp == BinaryOperator.BITWISE_EXCLUSIVE_OR)
                            {
                                atomicFunc = new NameID("atomicXor");
                            }
                            else if(atomicOp == BinaryOperator.MULTIPLY)
                            {
                                atomicFunc = new NameID("atomicMul");
                                Tools.exit("Atomic multiplication operation is not supported\nAtomic annotation: " 
                                		+ annot + "\n" + AnalysisTools.getEnclosingAnnotationContext(annot));
                            }
                            else if(atomicOp == BinaryOperator.DIVIDE)
                            {
                                // x = expr / x;
                                if(valExpr.equals(rhsExpr.getLHS()))
                                {
                                    atomicFunc = new NameID("atomicDivRHS");
                                }
                                // x = x / expr
                                else
                                {
                                	atomicFunc = new NameID("atomicDiv");
                                }
                                Tools.exit("Atomic division operation is not supported\nAtomic annotation: " 
                                		+ annot + "\n" + AnalysisTools.getEnclosingAnnotationContext(annot));
                            }

                            FunctionCall atomicCall = new FunctionCall(atomicFunc,
                                    new UnaryExpression(UnaryOperator.ADDRESS_OF, atomicVar.clone()),
                                    valExpr.clone());
                            CompoundStatement parentStmt = (CompoundStatement) atomicStmt.getParent();
                            ExpressionStatement atomicCallStmt =  new ExpressionStatement(atomicCall);
                            atomicCallStmt.annotate(annot.clone());
                            //PrintTools.println(atomicCallStmt.toString(),0);
                            parentStmt.addStatementAfter(atomicStmt, atomicCallStmt);
                            parentStmt.removeStatement(atomicStmt);
                        }
                        // x = expr
                        else
                        {
                            FunctionCall atomicCall = new FunctionCall(new NameID("atomicExch"),
                                    new UnaryExpression(UnaryOperator.ADDRESS_OF, assignExpr.getLHS().clone()),
                                    assignExpr.getRHS().clone());
                            CompoundStatement parentStmt = (CompoundStatement) atomicStmt.getParent();
                            ExpressionStatement atomicCallStmt =  new ExpressionStatement(atomicCall);
                            atomicCallStmt.annotate(annot.clone());
                            //PrintTools.println(atomicCallStmt.toString(),0);
                            parentStmt.addStatementAfter(atomicStmt, atomicCallStmt);
                            parentStmt.removeStatement(atomicStmt);

                            if(annot.containsKey("atomic_var"))
                            {
                                Set<Identifier> atomicVarSet = annot.get("atomic_var");
                                atomicVarSet.add(atomicVar);
                            }
                            else
                            {
                                Set<Identifier> atomicVarSet = new HashSet<Identifier>();
                                atomicVarSet.add(atomicVar);
                                annot.put("atomic_var", atomicVarSet);
                            }
                        }
                    }
                    //x binop= expr
                    else
                    {
                        NameID atomicFunc = null;
                        Expression valExpr = assignExpr.getRHS();
                        if(assignExpr.getOperator() == AssignmentOperator.ADD)
                        {
                            atomicFunc = new NameID("atomicAdd");
                        }
                        else if(assignExpr.getOperator() == AssignmentOperator.SUBTRACT)
                        {
                            //atomicFunc = new NameID("atomicAdd");
                            //valExpr = new UnaryExpression(UnaryOperator.MINUS, valExpr.clone());
                            atomicFunc = new NameID("atomicSub");
                        }
                        else if(assignExpr.getOperator() == AssignmentOperator.BITWISE_AND)
                        {
                            atomicFunc = new NameID("atomicAnd");
                        }
                        else if(assignExpr.getOperator() == AssignmentOperator.BITWISE_INCLUSIVE_OR)
                        {
                            atomicFunc = new NameID("atomicOr");
                        }
                        else if(assignExpr.getOperator() == AssignmentOperator.BITWISE_EXCLUSIVE_OR)
                        {
                            atomicFunc = new NameID("atomicXor");
                        }
                        else if(assignExpr.getOperator() == AssignmentOperator.MULTIPLY)
                        {
                        	atomicFunc = new NameID("atomicMul");
                        	Tools.exit("Atomic multiplication operation is not supported\nAtomic annotation: " 
                        			+ annot + "\n" + AnalysisTools.getEnclosingAnnotationContext(annot));
                        }
                        else if(assignExpr.getOperator() == AssignmentOperator.DIVIDE)
                        {
                            atomicFunc = new NameID("atomicDiv");
                        	Tools.exit("Atomic division operation is not supported\nAtomic annotation: " 
                        			+ annot + "\n" + AnalysisTools.getEnclosingAnnotationContext(annot));
                        }

                        if(annot.containsKey("atomic_var"))
                        {
                        	Set<Identifier> atomicVarSet = annot.get("atomic_var");
                        	atomicVarSet.add(atomicVar);
                        }
                        else
                        {
                        	Set<Identifier> atomicVarSet = new HashSet<Identifier>();
                        	atomicVarSet.add(atomicVar);
                        	annot.put("atomic_var", atomicVarSet);
                        }

                        FunctionCall atomicCall = new FunctionCall(atomicFunc,
                                new UnaryExpression(UnaryOperator.ADDRESS_OF, assignExpr.getLHS().clone()),
                                valExpr.clone());
                        CompoundStatement parentStmt = (CompoundStatement) atomicStmt.getParent();
                        ExpressionStatement atomicCallStmt =  new ExpressionStatement(atomicCall);
                        atomicCallStmt.annotate(annot.clone());
                        //PrintTools.println(atomicCallStmt.toString(),0);
                        parentStmt.addStatementAfter(atomicStmt, atomicCallStmt);
                        parentStmt.removeStatement(atomicStmt);

                    }
//						PrintTools.println("assignment " + assignExpr, 0);
                }
            }
            else if(at instanceof CompoundStatement)
            {
            	Tools.exit("Atomic operation is not supported\nAtomic annotation: " 
            			+ annot + "\n" + AnalysisTools.getEnclosingAnnotationContext(annot));
            }
            else
            {
            	Tools.exit("Atomic operation is not supported\nAtomic annotation: " 
            			+ annot + "\n" + AnalysisTools.getEnclosingAnnotationContext(annot));
            }
        }
    }

    protected void runtimeTransformationForConstMemory(Procedure cProc, List<FunctionCall> fCallList ) {
    	Set<FunctionCall> removeSet = new HashSet<FunctionCall>();
    	for( FunctionCall fCall : fCallList ) {
    		Statement fCallStmt = fCall.getStatement();
    		Annotation tAnnot = fCallStmt.getAnnotation(ARCAnnotation.class, "constant");
    		if( tAnnot != null ) {
    			if( !OpenACCRuntimeLibrary.isOpenARCAPI(fCall) ) {
    				Set<SubArray> dataSet = (Set<SubArray>)tAnnot.get("constant");	
    				SubArray sArray = dataSet.iterator().next();
    				List<Symbol> constantList = new LinkedList<Symbol>();
    				constantList.addAll(AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly));
    				Symbol constSym = constantList.get(0);
    				Symbol IRSym = constSym;
    				if( constSym instanceof PseudoSymbol ) {
    					IRSym = ((PseudoSymbol)constSym).getIRSymbol();
    				}
    				String symNameBase = null;
    				if( constSym instanceof AccessSymbol) {
    					symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)constSym);
    				} else {
    					symNameBase = constSym.getSymbolName();
    				}
    				String constVarName = "const__" + symNameBase;
    				if( !SymbolTools.isGlobal(IRSym) ) {
    					constVarName += "__" + cProc.getSymbolName();
    				}
					List<Expression> startList = new LinkedList<Expression>();
					List<Expression> lengthList = new LinkedList<Expression>();
					boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, fCallStmt);
					if( !foundDimensions ) {
						Tools.exit("[ERROR in ACC2CUDATranslator.runtimeTransformationForConstMemory()] Dimension information " +
								"of the following variable is unknown; exit.\n" + 
								"Variable: " + constSym.getSymbolName() + "\n" +
								"OpenACC Runtime API: " + fCall + "\n" +
								"Enclosing Procedure: " + cProc.getSymbolName() + "\n");
					}
					List<Specifier> typeSpecs = new ArrayList<Specifier>();
					if( IRSymbolOnly ) {
						typeSpecs.addAll(((VariableDeclaration)IRSym.getDeclaration()).getSpecifiers());
					} else {
						Symbol tSym = constSym;
						while( tSym instanceof AccessSymbol ) {
							tSym = ((AccessSymbol)tSym).getMemberSymbol();
						}
						typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
					}
					List<Specifier> clonedspecs = new ChainedList<Specifier>();
					clonedspecs.addAll(typeSpecs);
					clonedspecs.remove(Specifier.STATIC);
					///////////////////////////////////////////
					// GPU variables should not be constant. //
					///////////////////////////////////////////
					clonedspecs.remove(Specifier.CONST);
					//////////////////////////////
					// Remove extern specifier. //
					//////////////////////////////
					clonedspecs.remove(Specifier.EXTERN);
    				String oldFName = fCall.getName().toString();
    				if( OpenACCRuntimeLibrary.isDeviceMallocAPI(fCall) ) {
    					if( OpenACCRuntimeLibrary.isCopyInAPI(fCall) || oldFName.equals("acc_create") || oldFName.equals("acc_pcreate") 
    							|| oldFName.equals("acc_present_or_create")) {
    						//acc_copyin(), acc_pcopyin(), acc_present_or_copyin(), acc_create(), acc_pcreate(),
    						//or acc_present_or_create()
    						//[Step 1] Add global constant memory declaration into the kernel file.
    						/////////////////////////////////////
    						// Create a __constant__ variable. //
    						/////////////////////////////////////
    						//  __constant__ float a[SIZE1][SIZE2];  //
    						/////////////////////////////
    			            Set<Symbol> symbolSet = kernelsTranslationUnit.getSymbols();
    						Symbol constantSym = AnalysisTools.findsSymbol(symbolSet, constVarName);
    						boolean addNewConstSymbol = false;
    						Identifier constantID = null;
    						if( constantSym != null ) {
    							constantID = new Identifier((VariableDeclarator)constantSym);
    							//DEBUG: For implicit program-level data region, multiple constantID may be needed.
    						} else {
    							addNewConstSymbol = true;
    							List<Expression> arryDimList = new ArrayList<Expression>();
    							for( int i=0; i<lengthList.size(); i++ ) {
    								arryDimList.add(lengthList.get(i).clone());
    							}
    							ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
    							VariableDeclarator constantRef_declarator = new VariableDeclarator(new NameID(constVarName), arraySpecs);
    							List<Specifier> constspecs = new ChainedList<Specifier>();
    							constspecs.add(CUDASpecifier.CUDA_CONSTANT);
    							constspecs.addAll(clonedspecs);
    							Declaration constantRef_decl = new VariableDeclaration(constspecs, constantRef_declarator);
    							constantID = new Identifier(constantRef_declarator); 
    							//Insert __constant__ variable declaration.
    							Procedure ttProc = AnalysisTools.findFirstProcedure(kernelsTranslationUnit);
    							if( ttProc == null ) {
    								kernelsTranslationUnit.addDeclaration(constantRef_decl);
    							} else {
    								kernelsTranslationUnit.addDeclarationBefore(ttProc, constantRef_decl);
    							}
    						}

    						if( OpenACCRuntimeLibrary.isCopyInAPI(fCall) ) {
    							//[Step 2] Replace the API call with HI_memcpy_const() or HI_present_or_memcpy_const() call.
    							Expression memcpyName = null;
    							if( OpenACCRuntimeLibrary.isPresentCheckAPI(fCall) ) {
    								memcpyName = new NameID("HI_present_or_memcpy_const");
    							} else {
    								memcpyName = new NameID("HI_memcpy_const");
    							}
    							FunctionCall memcpyFunc = new FunctionCall(memcpyName);
    							memcpyFunc.addArgument(fCall.getArgument(0).clone());
    							memcpyFunc.addArgument(new StringLiteral(constVarName));
    							memcpyFunc.addArgument(new NameID("HI_MemcpyHostToDevice"));
    							memcpyFunc.addArgument(fCall.getArgument(1).clone());
    							ExpressionStatement eStmt = new ExpressionStatement(memcpyFunc);
    							fCallStmt.swapWith(eStmt);
    							removeSet.add(fCall);
    						}
    					} else if( oldFName.equals("acc_copyout") ) {
    						FunctionCall unmapFunc = new FunctionCall(new NameID("acc_unmap_data"));
    						unmapFunc.addArgument(fCall.getArgument(0).clone());
    						ExpressionStatement eStmt = new ExpressionStatement(unmapFunc);
    						fCallStmt.swapWith(eStmt);
    						removeSet.add(fCall);
    					}
    				} else if( oldFName.equals("acc_memcpy_to_device") ) {
    					FunctionCall memcpyFunc = new FunctionCall(new NameID("HI_memcpy_const"));
    					memcpyFunc.addArgument(fCall.getArgument(1).clone());
    					memcpyFunc.addArgument(new StringLiteral(constVarName));
    					memcpyFunc.addArgument(new NameID("HI_MemcpyHostToDevice"));
    					memcpyFunc.addArgument(fCall.getArgument(2).clone());
    					ExpressionStatement eStmt = new ExpressionStatement(memcpyFunc);
    					fCallStmt.swapWith(eStmt);
    					removeSet.add(fCall);
    				} else if( oldFName.equals("acc_delete") ) {
    					FunctionCall unmapFunc = new FunctionCall(new NameID("acc_unmap_data"));
    					unmapFunc.addArgument(fCall.getArgument(0).clone());
    					ExpressionStatement eStmt = new ExpressionStatement(unmapFunc);
    					fCallStmt.swapWith(eStmt);
    					removeSet.add(fCall);
    				}
    			} 
    		}
    	}
    	fCallList.removeAll(removeSet);
    }
}
