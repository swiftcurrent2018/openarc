/**
 * 
 */
package openacc.transforms;

import java.util.*;

import cetus.analysis.CallGraph;
import cetus.exec.Driver;
import cetus.hir.*;
import cetus.transforms.TransformPass;
import openacc.hir.*;
import openacc.analysis.*;

/**
 * <b>KernelCallingProcCloning</b> clones procedures that call compute regions (parallel regions/kernels regions).
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class KernelCallingProcCloning extends TransformPass {

	/**
	 * @param program
	 */
	public KernelCallingProcCloning(Program program) {
		super(program);
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[Kernel-Calling-Procedure Cloning]");
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		String mainEntryFunc = null;
		String value = Driver.getOptionValue("SetAccEntryFunction");
		if( (value != null) && !value.equals("1") ) {
			mainEntryFunc = value;
		}
		List<Procedure> procList = IRTools.getProcedureList(program);
		Procedure main = AnalysisTools.findMainEntryFunction(program, mainEntryFunc);
		if( main == null ) {
			PrintTools.println("\n[WARNING in KernelCallingProcCloning] This transform pass is skipped " +
					"since the compiler can not find accelerator entry function or main function; " +
					"if any compute region is executed in multiple contexts, user SHOULD manually clone " +
					"the function enclosing the compute region such that every compute region is " +
					"executed in a unique context.\n", 0);
			PrintTools.println("To enable this pass, the main entry function should be explicitly specified using " +
					"\"SetAccEntryFunction\" option.\n", 0);
			return;
		}
		// generate a list of procedures in post-order traversal
		//CallGraph callgraph = new CallGraph(program);
		CallGraph callgraph = new CallGraph(program, mainEntryFunc);
		// procedureList contains Procedure in ascending order; the last one is main
		List<Procedure> procedureList = callgraph.getTopologicalCallList();

		List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);
		// cloneProcMap contains procedures containing kernel regions and called multiple times.
		TreeMap<Integer, Procedure> cloneProcMap = new TreeMap<Integer, Procedure>();
		// MayCloneProcMap contains procedures containing kernel regions but called once; these 
		// may be cloned too if parent procedures are cloned.
		TreeMap<Integer, Procedure> MayCloneProcMap = new TreeMap<Integer, Procedure>();

		/* drive the engine; visit every procedure */
		for (Procedure proc : procedureList)
		{
			boolean kernelExists = false;
			int numOfCalls = 0;
			LinkedList<Procedure> callingProcs = new LinkedList<Procedure>();
			HashSet<Procedure> visitedProcs = new HashSet<Procedure>();
			String name = proc.getName().toString();
			/* f2c code uses MAIN__ */
			if ( ((mainEntryFunc != null) && name.equals(mainEntryFunc)) || 
					((mainEntryFunc == null) && (name.equals("main") || name.equals("MAIN__"))) ) {
				continue;
			}
			for(String kernelType : ACCAnnotation.computeRegions) {
				List<ACCAnnotation>  cRegionAnnots = IRTools.collectPragmas(proc.getBody(), ACCAnnotation.class, kernelType);
				if( (cRegionAnnots != null) && (!cRegionAnnots.isEmpty()) ) {
					kernelExists = true;
					break;
				}
			}
			if( !kernelExists ) {
				// Current procedure does not contain any kernel region; skip it.
				continue;
			}
			callingProcs.add(proc);
			while( !callingProcs.isEmpty() ) {
				Procedure c_proc = callingProcs.removeFirst();
				numOfCalls = 0;
				for( FunctionCall funcCall : funcCallList ) {
					if(c_proc.getName().equals(funcCall.getName())) {
						numOfCalls++;
						Traversable t = funcCall.getStatement();
						while( (t != null) && !(t instanceof Procedure) ) {
							t = t.getParent();
						}
						Procedure p_proc = (Procedure)t;
						name = p_proc.getName().toString();
						if ( ((mainEntryFunc != null) && !name.equals(mainEntryFunc)) || 
								(!name.equals("main") && !name.equals("MAIN__")) ) {
							if( !visitedProcs.contains(p_proc) ) {
								callingProcs.add(p_proc);
								visitedProcs.add(p_proc);
							}
						}
					}
				}
				int ind = procedureList.indexOf(c_proc);
				if( ind > -1 ) {
					if( numOfCalls > 1 ) {
						// Current procedure contains a kernel region and is called more than once.
						cloneProcMap.put(new Integer(ind), c_proc);
					}
					else {
						// Current procedure contains a kernel region but is called only once.
						MayCloneProcMap.put(new Integer(ind), c_proc);
					}
				}
			}
		}

		Integer lastKey = null;
		Procedure c_proc = null;
		// Clone procedures in clonedProcMap and MayCloneProcMap.
		while( !cloneProcMap.isEmpty() || !MayCloneProcMap.isEmpty() ) {
			if( !cloneProcMap.isEmpty() ) {
				lastKey = cloneProcMap.lastKey();
				c_proc = cloneProcMap.remove(lastKey);
			} else if( !MayCloneProcMap.isEmpty() ) {
				lastKey = MayCloneProcMap.lastKey();
				c_proc = MayCloneProcMap.remove(lastKey);
			}
			funcCallList = IRTools.getFunctionCalls(program);
			int numOfCalls = 0;
			for( FunctionCall funcCall : funcCallList ) {
				if(c_proc.getName().equals(funcCall.getName())) {
					if( numOfCalls > 0 ) {
						/////////////////////////////
						// Clone current procedure //
						/////////////////////////////
						//If a procedure has a static variable, it should not be cloned.
						//Set<Symbol> symSet = SymbolTools.getVariableSymbols(c_proc.getBody());
						Set<Symbol> symSet = SymbolTools.getLocalSymbols(c_proc.getBody());
						Set<Symbol> staticSyms = AnalysisTools.getStaticVariables(symSet);
						if( !staticSyms.isEmpty() ) {
							Tools.exit("[ERROR in KernelCallingProcCloning] if a procedure has static variables," +
									"it can not be cloned; for correct transformation, either \"disableStatic2GlobalConversion\" " +
									"option should be disabled or static variables should be manually promoted to global ones.\n" +
									"Procedure name: " + c_proc.getSymbolName() + "\n");
						}
						List<Specifier> return_types = c_proc.getReturnType();
						List<VariableDeclaration> oldParamList = 
							(List<VariableDeclaration>)c_proc.getParameters();
						CompoundStatement body = (CompoundStatement)c_proc.getBody().clone();
						String new_proc_name = c_proc.getSymbolName() + "_clnd" + numOfCalls;
						Procedure new_proc = new Procedure(return_types,
								new ProcedureDeclarator(new NameID(new_proc_name),
										new LinkedList()), body);	
						if( oldParamList != null ) {
							for( VariableDeclaration param : oldParamList ) {
								Symbol param_declarator = (Symbol)param.getDeclarator(0);
								VariableDeclaration cloned_decl = (VariableDeclaration)param.clone();
								Identifier paramID = new Identifier(param_declarator);
								Identifier cloned_ID = new Identifier((Symbol)cloned_decl.getDeclarator(0));
								new_proc.addDeclaration(cloned_decl);
								TransformTools.replaceAll((Traversable) body, paramID, cloned_ID);
							}
						}
						TranslationUnit tu = (TranslationUnit)c_proc.getParent();
						////////////////////////////
						// Add the new procedure. //
						////////////////////////////
						tu.addDeclarationAfter(c_proc, new_proc);
						////////////////////////////////////////////////////////////////////////
						//If the current procedure has annotations, copy them to the new one. //
						////////////////////////////////////////////////////////////////////////
						List<Annotation> cAnnotList = c_proc.getAnnotations();
						if( (cAnnotList != null) && (!cAnnotList.isEmpty()) ) {
							for( Annotation cAn : cAnnotList ) {
								new_proc.annotate(cAn.clone());
							}
						}
						//////////////////////////////////////////////////////////////////
						//If declaration statement exists for the original procedure,   //
						//create a new declaration statement for the new procedure too. //
						//////////////////////////////////////////////////////////////////
						FlatIterator Fiter = new FlatIterator(program);
						while (Fiter.hasNext())
						{
							TranslationUnit cTu = (TranslationUnit)Fiter.next();
							//System.out.println("Current translation unit: " + cTu.getInputFilename());
							Declaration firstDecl = cTu.getFirstDeclaration();
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
										if( AnalysisTools.isInHeaderFile(procDecl, cTu) ) {
											if( firstDecl != null ) {
												cTu.addDeclarationBefore(firstDecl, newProcDecl);
											} else {
												cTu.addDeclaration(newProcDecl);
											}
										} else {
											cTu.addDeclarationAfter(procDecl, newProcDecl);
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
										ACCAnalysis.updateSymbolsInACCAnnotations(newProcDecl, null);
										break;
									}
								}
							}
						}
						//////////////////////////////////////////////////////////
						// Create a new function call for the cloned procedure. //
						//////////////////////////////////////////////////////////
						if( funcCall != null ) {
							FunctionCall new_funcCall = new FunctionCall(new NameID(new_proc_name));
							List<Expression> argList = (List<Expression>)funcCall.getArguments();
							if( argList != null ) {
								for( Expression exp : argList ) {
									//new_funcCall.addArgument(exp.clone());
									Expression dummyArg = new NameID("dummyArg");
									new_funcCall.addArgument(dummyArg);
									dummyArg.swapWith(exp);
									
								}
							}
							funcCall.swapWith(new_funcCall);
						}
						/////////////////////////////////////////////////////////////////////////
						// Update the newly cloned procedure:                                  //
						//     1) Update symbols in the new procedure, including symbols       //
						//        in OmpAnnoations.                                            //
						/////////////////////////////////////////////////////////////////////////
						SymbolTools.linkSymbol(new_proc);
						ACCAnalysis.updateSymbolsInACCAnnotations(new_proc, null);
					}
					numOfCalls++;
				}
			}
		}
	}

}
