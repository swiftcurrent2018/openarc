/**
 * 
 */
package openacc.transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.LinkedList;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.Map;
import java.util.HashSet;
import java.util.HashMap;

import cetus.exec.Driver;
import cetus.hir.*;
import openacc.hir.*;
import openacc.analysis.*;
import cetus.transforms.*;
import cetus.analysis.LoopTools;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *
 */
public abstract class TransformTools {
	static private int tempIndexBase = 2000;
	/**
	 * Java doesn't allow a class to be both abstract and final,
	 * so this private constructor prevents any derivations.
	 */
	private TransformTools()
	{
	}
	
	/**
	 * This method is an another version of {@link cetus.transforms.LoopInterchange#swapLoop(ForLoop, ForLoop)}, which
	 * literally swaps loops rather than exchanging init/cond/step components, resulting in annotation swapping. 
	 * 
	 * @param upperloop upper loop to be swapped
	 * @param lowerloop lower loop to be swapped
	 */
    public static void swapLoop(ForLoop upperloop, ForLoop lowerloop) 
    {
    	//Step1: detach lowerloop from upperloop by swapping lowerloop with dummy statement
        Statement dummy = new AnnotationStatement();
        dummy.swapWith(lowerloop);
    	//Step2: swap upperloop body with lowerloop body
        Statement ULoopBody = upperloop.getBody();
        Statement LLoopBody = lowerloop.getBody();
        ULoopBody.swapWith(LLoopBody);
    	//Step3: swap upperloop with lowerloop
        upperloop.swapWith(lowerloop);
    	//Step4: re-attach upperloop to lowerloop by swapping upperloop with dummy statement
        dummy.swapWith(upperloop);
    }
    
    /**
     * This method is an extended version of {@link cetus.transforms.LoopInterchange#swapLoop(ForLoop, ForLoop)}, which literally swaps loops and
     * also considers OpenACC collapse clauses; loops associated with a collapse clause are swapped
     * together
     * 
     * @param upperloop upper loop to be swapped
     * @param lowerloop lower loop to be swapped
     */
    public static void extendLoopSwap(ForLoop upperloop, ForLoop lowerloop) {
    	ACCAnnotation cAnnotU = upperloop.getAnnotation(ACCAnnotation.class, "collapse");
    	ACCAnnotation cAnnotL = lowerloop.getAnnotation(ACCAnnotation.class, "collapse");
    	int collapseLevelU = 0;
    	int collapseLevelL = 0;
    	List<ForLoop> nestedLoopsU = new LinkedList<ForLoop>();
    	List<ForLoop> nestedLoopsL = new LinkedList<ForLoop>();
    	Statement dummy = new AnnotationStatement();
    	Statement bodyU;
    	Statement bodyL;
    	
    	if( cAnnotU != null ) {
    		Object cObj = cAnnotU.get("collapse");
    		if( (cObj != null) && (cObj instanceof IntegerLiteral) ) {
    			collapseLevelU = (int)((IntegerLiteral)cObj).getValue();
    		}
    	}
    	if( cAnnotL != null ) {
    		Object cObj = cAnnotL.get("collapse");
    		if( (cObj != null) && (cObj instanceof Integer) ) {
    			collapseLevelL = (int)((IntegerLiteral)cObj).getValue();
    		}
    	}
    	if( collapseLevelU > 1 ) {
    		if(!AnalysisTools.extendedPerfectlyNestedLoopChecking(upperloop, collapseLevelU, nestedLoopsU, null)) {
    			Tools.exit("[ERROR in transformTools.extendedLoopSwap()] loopswapping is applicalbe only to perfectly nested loops.\n" +
    					"Target loop: \n" + upperloop + "\n");
    		}
    		if( nestedLoopsU.contains(lowerloop) ) {
    			Tools.exit("[ERROR in transformTools.extendedLoopSwap()] swapping loops associated with the same collapse clause is not allowed.\n" +
    					"Target loop: \n" + upperloop + "\n");
    		}
    	}
    	if( collapseLevelL > 1 ) {
    		if(!AnalysisTools.extendedPerfectlyNestedLoopChecking(lowerloop, collapseLevelL, nestedLoopsL, null)) {
    			Tools.exit("[ERROR in transformTools.extendedLoopSwap()] loopswapping is applicalbe only to perfectly nested loops.\n" +
    					"Target loop: \n" + lowerloop + "\n");
    		}
    		if( nestedLoopsL.contains(upperloop) ) {
    			Tools.exit("[ERROR in transformTools.extendedLoopSwap()] swapping loops associated with the same collapse clause is not allowed.\n" +
    					"Target loop: \n" + lowerloop + "\n");
    		}
    	}
    	//Step1: detach lowerloop from upperloop by swapping lowerloop with dummy statement
    	dummy.swapWith(lowerloop);
    	//Step2: swap upperloop body with lowerloop body
    	if( collapseLevelU > 1 ) {
    		bodyU = nestedLoopsU.get(collapseLevelU-2).getBody();
    	} else {
    		bodyU = upperloop.getBody();
    	}
    	if( collapseLevelL > 1 ) {
    		bodyL = nestedLoopsL.get(collapseLevelL-2).getBody();
    	} else {
    		bodyL = lowerloop.getBody();
    	}
    	bodyU.swapWith(bodyL);
    	//Step3: swap upperloop with lowerloop
    	upperloop.swapWith(lowerloop);
    	//Step4: re-attach upperloop to lowerloop by swapping upperloop with dummy statement
    	dummy.swapWith(upperloop);
    }
    
    /**
     * Perform loop permutation for the nested loops rooted at {@code targetLoop} according to {@code permuteList}.
     * 
     * @param targetLoop root loop where permutation will occur.
     * @param permuteList list of index variables, so that permuted loops will be in that order.
     * @param permuteAnnotations  if true, attached annotations are also swapped.
     * @return
     */
    static public ForLoop permuteLoops(ForLoop targetLoop, List<Expression> permuteList, boolean permuteAnnotations) {
    	ForLoop outermostLoop = null;
    	if( (permuteList != null) && !permuteList.isEmpty() && (targetLoop != null) ) {
    		int nestLevel = permuteList.size();
			//List<AnnotationStatement> commentList = new LinkedList<AnnotationStatement>();
			List<ForLoop> loopList = new LinkedList<ForLoop>();
			loopList.add(targetLoop);
			boolean pnest = AnalysisTools.extendedPerfectlyNestedLoopChecking(targetLoop, nestLevel, loopList, null);
			if( pnest ) {
				List<Expression> loopIndices = new LinkedList<Expression>();
				boolean error = false;
				for( ForLoop tLoop : loopList ) {
					Expression indexVar = LoopTools.getIndexVariable(tLoop);
					if( indexVar == null ) {
						error = true;
						break;
					} else {
						loopIndices.add(indexVar);
					}
				}
				if( !error ) {
					int i, k;
					Expression ref, comp;
					ForLoop ULoop, LLoop;
					for( i=0; i<nestLevel; i++ ) {
						ref = permuteList.get(i);
						comp = null;
						for( k=i; k<nestLevel; k++ ) {
							comp = loopIndices.get(k);
							if( ref.equals(comp) ) {
								break;
							} else {
								comp = null;
							}
						}
						if( comp == null ) {
							outermostLoop = null;
							break;
						} else {
							if( i == k ) {
								if( i==0 ) {
									outermostLoop = targetLoop;
								}
							} else {
								ULoop = loopList.get(i);
								LLoop = loopList.get(k);
								if( permuteAnnotations ) {
									swapLoop(ULoop, LLoop);
								} else {
									cetus.transforms.LoopInterchange.swapLoop(ULoop, LLoop);
								}
								ref = (Expression)loopIndices.set(i, comp);
								loopIndices.set(k, ref);
								loopList.set(i, LLoop);
								loopList.set(k, ULoop);
								if( i==0 ) {
									if( permuteAnnotations ) {
										outermostLoop = LLoop;
									} else {
										outermostLoop = ULoop;
									}
								}
							}
						}
					}
				}
			}
    	}
    	return outermostLoop;
    }
    
	static public void removeUnusedProcedures(Program prog) {
		String mainEntryFunc = null;
		String value = Driver.getOptionValue("SetAccEntryFunction");
		if( (value != null) && !value.equals("1") ) {
			mainEntryFunc = value;
		}
		List<Procedure> cprocList = IRTools.getProcedureList(prog);
		List<SomeExpression> someExps = IRTools.getExpressionsOfType(prog, SomeExpression.class);
		String main_name = null;
		if( cprocList != null ) {
			for( Procedure tProc : cprocList ) {
				String name = tProc.getName().toString();

				/* f2c code uses MAIN__ */
				if ( ((mainEntryFunc != null) && name.equals(mainEntryFunc)) || 
						((mainEntryFunc == null) && (name.equals("main") || name.equals("MAIN__"))) ) {
					main_name = name;
					break;
				}
			}
		}
		if( main_name == null ) {
			//No entry function is found, and thus it's not possible to check whether a procedure
			//is used or not; skip this transformation!
			return;
		}
		boolean mayContainUnusedOnes = true;
		while ( mayContainUnusedOnes ) {
			List<Procedure> procList = IRTools.getProcedureList(prog);
			List<FunctionCall> funcCallList = IRTools.getFunctionCalls(prog);
			HashSet<String> funcCallSet = new HashSet<String>();
			HashSet<String> deletedSet = new HashSet<String>();
			for( FunctionCall fCall : funcCallList ) {
				funcCallSet.add(fCall.getName().toString());
			}
			for( Procedure proc : procList ) {
				String pName = proc.getSymbolName();
				if ( pName.equals(main_name) ) {
					// Skip main procedure.
					continue;
				}
				if( !pName.contains("dev__") && !pName.contains("_clnd") ) {
					//[DEBUG] Current implementation does not detect functions used via function pointers.
					//Therefore, we will remove unused functions only if they are created as by-product of 
					//O2G translation (by KernelCallingProcCloning() and devProcCloning()).
					continue;
				}
				if( !funcCallSet.contains(pName) ) {
					boolean notUsed = true;
					if( someExps != null ) {
						for( SomeExpression sExp: someExps ) {
							if( sExp.toString().contains(pName) ) {
								notUsed = false;
								break;
							}
						}
					}
					if( notUsed ) {
						//Check whether the procedure is used in pragma annotations.
						//[DEBUG] For simple checking, use string comparison, but the overhead
						//may be huge; more optimization is necessary.
						for( Procedure ttProc : procList ) {
							String ttname = ttProc.getName().toString();
							if( !ttname.equals(pName) ) {
								CompoundStatement ttBody = ttProc.getBody();
								if( ttBody.toString().contains(pName) ) {
									notUsed = false;
									break;
								}
							}
						}
					}
					if( notUsed ) {
						//Procedure is never used in this program.
						TranslationUnit tu = (TranslationUnit)proc.getParent();
						//Delete the unused procedure.
						/////////////////////////////////////////////////////////////////////////////
						// FIXME: When a procedure, proc, is added to a TranslationUnit, tu,       //
						// tu.containsSymbol(proc) returns false, but tu.containsDeclaration(proc) //
						// returns true.                                                           //
						/////////////////////////////////////////////////////////////////////////////
						//if( tu.containsSymbol(proc) ) 
						if( tu.containsDeclaration(proc) ) {
							tu.removeChild(proc);
							deletedSet.add(pName);
						} else {
							PrintTools.println("\n[WARNING in removeUnusedProcedures()] Can't delete procedure, " + pName + ".\n", 0);
						}
						// Check whether corresponding ProcedureDeclarator exists and delete it. 
						FlatIterator Fiter = new FlatIterator(prog);
						while (Fiter.hasNext())
						{
							TranslationUnit cTu = (TranslationUnit)Fiter.next();
							List<Traversable> children = cTu.getChildren();
							Traversable child = null;
							do {
								child = null;
								for( Traversable t : children ) {
									if( t instanceof VariableDeclaration ) {
										Declarator declr = ((VariableDeclaration)t).getDeclarator(0);
										if( declr instanceof ProcedureDeclarator ) {
											if( ((ProcedureDeclarator)declr).getSymbolName().equals(pName) ) {
												child = t;
												break;
											}
										}
									}
								}
								if( child != null ) {
									//cTu.removeChild(child);
									children.remove(child);
									child.setParent(null);
								}
							} while (child != null);
						}
					}
				}
			}
			if( !deletedSet.isEmpty() ) {
				mayContainUnusedOnes = true;
				PrintTools.println("[INFO in removeUnusedProcedures()] list of deleted procedures: " +
						PrintTools.collectionToString(deletedSet, ", "), 2);
			} else {
				mayContainUnusedOnes = false;
			}
		}
	}
	
	static private String[] dummyVars = {"gpuNumThreads", "gpuNumBlocks", "gpuBytes",
	"gpuNumBlocks1", "gpuNumBlocks2", "totolNumThreads", "gpuGmemSize", "gpuSmemSize",
	"totalNumThreads"};
	
	static public void removeUnusedSymbols(Program prog) {
		Set<TranslationUnit> skipTrList = new HashSet<TranslationUnit>();
		List<Procedure> procList = IRTools.getProcedureList(prog);
		for( Procedure proc : procList ) {
			List returnTypes = proc.getTypeSpecifiers();
			if( returnTypes.contains(CUDASpecifier.CUDA_GLOBAL) 
				|| returnTypes.contains(CUDASpecifier.CUDA_DEVICE) ) {
				skipTrList.add(IRTools.getParentTranslationUnit(proc));
			}
			CompoundStatement pBody = proc.getBody();
			List<OmpAnnotation> ompAnnots = AnalysisTools.collectPragmas(pBody, OmpAnnotation.class);
			boolean checkOmpAnnots = false;
			if( (ompAnnots != null) && (!ompAnnots.isEmpty()) ) {
				checkOmpAnnots = true;
			}
			Set<Symbol> accessedSymbols = AnalysisTools.getAccessedVariables(pBody, true);
			//Set<Symbol> declaredSymbols = SymbolTools.getVariableSymbols(pBody);
			Set<Symbol> declaredSymbols = SymbolTools.getLocalSymbols(pBody);
			List<SomeExpression> someExps = IRTools.getExpressionsOfType(pBody, SomeExpression.class);
			HashSet<Symbol> unusedSymbolSet = new HashSet<Symbol>();
			//Symbols used in kernel function configuration are not visible to IR tree.
			//Therefore, they should be checked manually.
			List<FunctionCall> funcCalls = IRTools.getFunctionCalls(pBody);
			if( funcCalls != null ) {
				for( FunctionCall fCall : funcCalls ) {
					if( fCall instanceof KernelFunctionCall ) {
						List<Traversable> kConfList = (List<Traversable>)((KernelFunctionCall)fCall).getConfArguments();
						for( Traversable tConf : kConfList ) {
							Set<Symbol> tSyms = AnalysisTools.getAccessedVariables(tConf, true);
							if( tSyms != null ) {
								accessedSymbols.addAll(tSyms);
							}
						}
					}
				}
			}
			
			for( Symbol sym : declaredSymbols ) {
				if( (sym instanceof Declarator) && (AnalysisTools.isTypedefName((Declarator)sym)) ) {
					continue; //Skip typedef name.
				}
				String sname = sym.getSymbolName();
				CompoundStatement cParent = null;
				if( sname.startsWith("dimBlock") || sname.startsWith("dimGrid") ||
					sname.startsWith("_gtid") || sname.startsWith("_bid") ||
					sname.startsWith("row_temp_") || sname.startsWith("lred__") ||
					sname.endsWith("__extended") || //sname.startsWith("gpu__") ||
					sname.startsWith("red__") || sname.startsWith("param__") ||
					sname.startsWith("pitch__") || sname.startsWith("sh__") ) {
					continue;
				}
				if( !accessedSymbols.contains(sym) ) {
					String symName = sym.getSymbolName();
					boolean notUsed = true;
					if( someExps != null ) {
						for( SomeExpression sExp: someExps ) {
							if( sExp.toString().contains(symName) ) {
								notUsed = false;
								break;
							}
						}
					}
					if( notUsed ) {
						if( sym instanceof VariableDeclarator ) {
							VariableDeclarator declr = (VariableDeclarator)sym;
							if( !AnalysisTools.isClassMember(declr) ) {
								Declaration decl = declr.getDeclaration();
								cParent = (CompoundStatement)decl.getParent().getParent();
								unusedSymbolSet.add(sym);
								//pBody.removeChild(decl.getParent());
								cParent.removeChild(decl.getParent());
							}
						}
						if( checkOmpAnnots ) {
							//Remove unused symbols from OpenMP clauses.
							for( OmpAnnotation oAnnot : ompAnnots ) {
								for( String key: oAnnot.keySet() ) {
									Object obj = oAnnot.get(key);
									if( (obj instanceof String) && ((String)obj).equals(symName) ) {
										oAnnot.remove(key);
									} else if( obj instanceof Set ) {
										Set<String> sSet = (Set<String>)obj;
										sSet.remove(symName);
										if( sSet.isEmpty() ) {
											oAnnot.remove(key);
										}
									}
								}
							}
						}
					}
				}
			}
			if( !unusedSymbolSet.isEmpty() ) {
				PrintTools.println(" Declarations removed from a procedure, " + 
						proc.getSymbolName() + ": " + AnalysisTools.symbolsToString(unusedSymbolSet, ", "), 2);
			}
		}
		
		//DEBUG: removing unused dummy variables are temporarily disabled, since it may cause
		//problems when one file is manually included in another file.
/*		List<String> dummyVarList = Arrays.asList(dummyVars);
		for ( Traversable tt : prog.getChildren() )
		{
			HashSet<Symbol> unusedSymbolSet = new HashSet<Symbol>();
			TranslationUnit tu = (TranslationUnit)tt;
			String iFileName = tu.getInputFilename();
			int dot = iFileName.lastIndexOf(".h");
			if( dot >= 0 ) {
				continue;
			} else if( skipTrList.contains(tu)) {
				continue;
			}
			Set<Symbol> accessedSymbols = SymbolTools.getAccessedSymbols(tu);
			if( accessedSymbols == null ) {
				continue;
			}
			for(String var : dummyVarList) {
				Declaration symDecl = SymbolTools.findSymbol(tu, var);
				if( ( symDecl != null ) && ( symDecl instanceof VariableDeclaration ) ) {
					VariableDeclarator sym = (VariableDeclarator)((VariableDeclaration)symDecl).getDeclarator(0);
					if( !accessedSymbols.contains(sym)) {
						unusedSymbolSet.add(sym);
						tu.removeChild(symDecl);
					}
				}
			}
			if( !unusedSymbolSet.isEmpty() ) {
				PrintTools.println(" Declarations removed from a translation unit, " + 
						iFileName + ": " + AnalysisTools.symbolsToString(unusedSymbolSet, ", "), 2);
			}
		}*/
	}
	
	/**
	 * Get a temporary scalar variable that can be used as temporary data holder. 
	 * The name of the variable is decided by the type and the trailer value,
	 * and if the variable with the given name exists
	 * in a region, where, this function returns the existing variable.
	 * Otherwise, this function create a new variable with the given name.
	 * This function differs from Tools.getTemp() in two ways; first if the 
	 * temporary variable exits in the region, this function returns existing
	 * one, but Tools.getTemp() creates another new one. 
	 * Second, if the temporary variable does not exist in the region, this 
	 * function creates the new variable, but Tools.getTemp() searches parents
	 * of the region and creates the new variable only if none of parents contains
	 * the temporary variable.
	 * 
	 * @param where code region from where temporary variable is searched or 
	 *        created. 
	 * @param type type of the new variable
	 * @param baseName base name used to create/search a variable name
	 * @param trailer integer trailer that is used to create/search a variable name if baseName is null
	 * @return
	 */
	public static Identifier getTempScalar(Traversable where, List<Specifier> typeSpecs, String baseName, long trailer) {
	    Traversable t = where;
	    while ( !(t instanceof SymbolTable) )
	      t = t.getParent();
	    // Traverse to the parent of a loop statement
	    if (t instanceof ForLoop || t instanceof DoLoop || t instanceof WhileLoop) {
	      t = t.getParent();
	      while ( !(t instanceof SymbolTable) )
	        t = t.getParent();
	    }
	    SymbolTable st = (SymbolTable)t;
	    String name;
	    if( baseName == null ) {
	    	String typeString = "";
	    	for( Specifier spec : typeSpecs ) {
	    		typeString += spec.toString();
	    	}
	    	String header = "_ti_" + typeString;
	    	name = header+"_"+trailer;
	    } else {
	    	name = baseName;
	    }
	    Identifier ret = null;
	    ///////////////////////////////////////////////////////////////////////////
	    // SymbolTable.findSymbol(IDExpression name) can not be used here, since //
	    // it will search parent tables too.                                     //
	    ///////////////////////////////////////////////////////////////////////////
	    Set<String> symNames = AnalysisTools.symbolsToStringSet(st.getSymbols());
	    if( symNames.contains(name) ) {
	    	VariableDeclaration decl = (VariableDeclaration)st.findSymbol(new NameID(name));
	    	ret = new Identifier((VariableDeclarator)decl.getDeclarator(0));
	    } else {
	    	//ret = SymbolTools.getTemp(t, Specifier.INT, header);
	    	///////////////////////////////////////////////////////////////////
	    	//SymbolTools.getTemp() may cause a problem if parent symbol tables    //
	    	//contain a variable whose name is the same as the one of ret.   //
	    	//To avoid this problem, a new temp variable is created directly //
	    	//here without using SymbolTools.getTemp().                           //
	    	///////////////////////////////////////////////////////////////////
	    	VariableDeclarator declarator = new VariableDeclarator(new NameID(name));
	        VariableDeclaration decl = new VariableDeclaration(typeSpecs, declarator);
	        st.addDeclaration(decl);
	    	ret = new Identifier(declarator);
	    }
	    return ret;
	}
	
	/**
	 * Get a temporary long integer variable that can be used as a loop index variable 
	 * or other temporary data holder. The name of the variable is decided by
	 * using the trailer value, and if the variable with the given name exists
	 * in a region, where, this function returns the existing variable.
	 * Otherwise, this function create a new variable with the given name.
	 * This function differs from Tools.getTemp() in two ways; first if the 
	 * temporary variable exits in the region, this function returns existing
	 * one, but Tools.getTemp() creates another new one. 
	 * Second, if the temporary variable does not exist in the region, this 
	 * function creates the new variable, but Tools.getTemp() searches parents
	 * of the region and creates the new variable only if none of parents contains
	 * the temporary variable.
	 * 
	 * @param where code region from where temporary variable is searched or 
	 *        created. 
	 * @param trailer integer trailer that is used to create/search a variable name
	 * @return
	 */
	public static Identifier getTempIndex(Traversable where, long trailer) {
	    Traversable t = where;
	    while ( !(t instanceof SymbolTable) )
	      t = t.getParent();
	    // Traverse to the parent of a loop statement
	    if (t instanceof ForLoop || t instanceof DoLoop || t instanceof WhileLoop) {
	      t = t.getParent();
	      while ( !(t instanceof SymbolTable) )
	        t = t.getParent();
	    }
	    SymbolTable st = (SymbolTable)t;
	    String header = "_ti_100";
	    String name = header+"_"+trailer;
	    Identifier ret = null;
	    ///////////////////////////////////////////////////////////////////////////
	    // SymbolTable.findSymbol(IDExpression name) can not be used here, since //
	    // it will search parent tables too.                                     //
	    ///////////////////////////////////////////////////////////////////////////
	    Set<String> symNames = AnalysisTools.symbolsToStringSet(st.getSymbols());
	    if( symNames.contains(name) ) {
	    	VariableDeclaration decl = (VariableDeclaration)st.findSymbol(new NameID(name));
	    	ret = new Identifier((VariableDeclarator)decl.getDeclarator(0));
	    } else {
	    	//ret = SymbolTools.getTemp(t, Specifier.INT, header);
	    	///////////////////////////////////////////////////////////////////
	    	//SymbolTools.getTemp() may cause a problem if parent symbol tables    //
	    	//contain a variable whose name is the same as the one of ret.   //
	    	//To avoid this problem, a new temp variable is created directly //
	    	//here without using SymbolTools.getTemp().                           //
	    	///////////////////////////////////////////////////////////////////
	    	VariableDeclarator declarator = new VariableDeclarator(new NameID(name));
	        //VariableDeclaration decl = new VariableDeclaration(Specifier.LONG, declarator);
	        VariableDeclaration decl = new VariableDeclaration(Specifier.INT, declarator);
	        st.addDeclaration(decl);
	    	ret = new Identifier(declarator);
	    }
	    return ret;
	}
	
	/**
	 * Get a temporary pointer variable with types of {@code typeSpecs} that can be used as
	 * a temporary data pointer. The name of the variable is decided by  the types, {@code typeSpecs},
	 * and the trailer value, and if the variable with the given name exists
	 * in a region, where, this function returns the existing variable.
	 * Otherwise, this function create a new variable with the given name.
	 * 
	 * @param where code region from where temporary variable is searched or 
	 *        created. 
	 * @param typeSpecs, types of the new pointer variable
	 * @param trailer integer trailer that is used to create/search a variable name
	 * @return
	 */
	public static Identifier getPointerTempIndex(Traversable where, List<Specifier> typeSpecs, long trailer) {
	    Traversable t = where;
	    while ( !(t instanceof SymbolTable) )
	      t = t.getParent();
	    // Traverse to the parent of a loop statement
	    if (t instanceof ForLoop || t instanceof DoLoop || t instanceof WhileLoop) {
	      t = t.getParent();
	      while ( !(t instanceof SymbolTable) )
	        t = t.getParent();
	    }
	    SymbolTable st = (SymbolTable)t;
	    String typeString = "";
	    for( Specifier spec : typeSpecs ) {
	    	typeString += spec.toString();
	    }
	    //remove all white spaces.
	    typeString.replaceAll("\\s","");
	    String header = "pt__";
	    String name = header+"_"+typeString+"_"+trailer;
	    Identifier ret = null;
	    ///////////////////////////////////////////////////////////////////////////
	    // SymbolTable.findSymbol(IDExpression name) can not be used here, since //
	    // it will search parent tables too.                                     //
	    ///////////////////////////////////////////////////////////////////////////
	    Set<String> symNames = AnalysisTools.symbolsToStringSet(st.getSymbols());
	    if( symNames.contains(name) ) {
	    	VariableDeclaration decl = (VariableDeclaration)st.findSymbol(new NameID(name));
	    	ret = new Identifier((VariableDeclarator)decl.getDeclarator(0));
	    } else {
	    	//ret = SymbolTools.getTemp(t, Specifier.INT, header);
	    	///////////////////////////////////////////////////////////////////
	    	//SymbolTools.getTemp() may cause a problem if parent symbol tables    //
	    	//contain a variable whose name is the same as the one of ret.   //
	    	//To avoid this problem, a new temp variable is created directly //
	    	//here without using SymbolTools.getTemp().                           //
	    	///////////////////////////////////////////////////////////////////
	    	VariableDeclarator declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, new NameID(name));
	        VariableDeclaration decl = new VariableDeclaration(typeSpecs, declarator);
	        st.addDeclaration(decl);
	    	ret = new Identifier(declarator);
	    }
	    return ret;
	}

	/**
	 * Get a new temporary long integer variable, which has not been created 
	 * by getTempIndex() method.
	 * 
	 * @param where
	 * @return
	 */
	public static Identifier getNewTempIndex(Traversable where) {
	    Traversable t = where;
	    while ( !(t instanceof SymbolTable) )
	      t = t.getParent();
	    // Traverse to the parent of a loop statement
	    if (t instanceof ForLoop || t instanceof DoLoop || t instanceof WhileLoop) {
	      t = t.getParent();
	      while ( !(t instanceof SymbolTable) )
	        t = t.getParent();
	    }
	    SymbolTable st = (SymbolTable)t;
	    String header = "_ti_100";
	    int trailer = 0;
		Identifier ret = null;
	   	Set<String> symNames = AnalysisTools.symbolsToStringSet(st.getSymbols());
	    while ( true ) {
	    	String name = header+"_"+trailer;
	    	if( symNames.contains(name) ) {
	    		trailer++;
	    	} else {
	    		//ret = SymbolTools.getTemp(t, Specifier.INT, header);
	    		///////////////////////////////////////////////////////////////////
	    		//SymbolTools.getTemp() may cause a problem if parent symbol tables    //
	    		//contain a variable whose name is the same as the one of ret.   //
	    		//To avoid this problem, a new temp variable is created directly //
	    		//here without using SymbolTools.getTemp().                            //
	    		///////////////////////////////////////////////////////////////////
	    		VariableDeclarator declarator = new VariableDeclarator(new NameID(name));
	    		//VariableDeclaration decl = new VariableDeclaration(Specifier.LONG, declarator);
	    		VariableDeclaration decl = new VariableDeclaration(Specifier.INT, declarator);
	    		st.addDeclaration(decl);
	    		ret = new Identifier(declarator);
	    		break;
	    	}
	    }
	    return ret;
	}
	
	/**
	 * Clone the input symbol with a new name, {@code name}, and declare the new symbol in the specified scope, where, if
	 * it does not exist. If existing, return the identifier for the existing symbol.
	 * 
	 * @param where
	 * @param inSym
	 * @param name
	 * @param removeSpecs
	 * @param addSpecs
	 * @param removeLeftMostDim true to set leftmost dimension to be unspecified
	 * @return
	 */
	public static Identifier declareClonedVariable(Traversable where, Symbol inSym, String name, List<Specifier> removeSpecs,
			List<Specifier> addSpecs, boolean removeLeftMostDim, boolean addRestrictQualifier) {
		NameID id = new NameID(name);
		Identifier newID = null;
		List specs = null;
		List orgArraySpecs = null;
		List arraySpecs = new LinkedList();
		SymbolTable st;
		if (where instanceof SymbolTable) {
			st = (SymbolTable)where;
		} else {
			st = IRTools.getAncestorOfType(where, SymbolTable.class);
		}
		if (st instanceof Loop) {
			st = IRTools.getAncestorOfType(st, SymbolTable.class);
		}
		Set<String> symNames = AnalysisTools.symbolsToStringSet(st.getSymbols());
		if( symNames.contains(name) ) {
			VariableDeclaration decl = (VariableDeclaration)st.findSymbol(new NameID(name));
			newID = new Identifier((Symbol)decl.getDeclarator(0));
		} else {
			if( inSym instanceof VariableDeclarator ) {
				VariableDeclarator varSym = (VariableDeclarator)inSym;
				specs = varSym.getTypeSpecifiers();
				if( removeSpecs != null ) {
					specs.removeAll(removeSpecs);
				}
				if( addSpecs != null ) {
					specs.addAll(addSpecs);
				}
				// Separate declarator/declaration specifiers.
				List declaration_specs = new ArrayList(specs.size());
				List declarator_specs = new ArrayList(specs.size());
				for (int i = 0; i < specs.size(); i++) {
					Object spec = specs.get(i);
					if (spec instanceof PointerSpecifier) {
						if( addRestrictQualifier ) {
							if( spec.equals(PointerSpecifier.UNQUALIFIED) ) {
								declarator_specs.add(PointerSpecifier.RESTRICT);
							} else if( spec.equals(PointerSpecifier.CONST) ) {
								declarator_specs.add(PointerSpecifier.CONST_RESTRICT);
							} else if( spec.equals(PointerSpecifier.CONST_VOLATILE) ) {
								declarator_specs.add(PointerSpecifier.CONST_RESTRICT_VOLATILE);
							} else {
								declarator_specs.add(spec);
							}
						} else {
							declarator_specs.add(spec);
						}
					} else {
						declaration_specs.add(spec);
					}
				}
				orgArraySpecs = varSym.getArraySpecifiers();
				if(  (orgArraySpecs != null) && (!orgArraySpecs.isEmpty()) ) {
					for( Object obj : orgArraySpecs ) {
						if( obj instanceof ArraySpecifier ) {
							ArraySpecifier oldAS = (ArraySpecifier)obj;
							int numDims = oldAS.getNumDimensions();
							List<Expression> dimensions = new LinkedList<Expression>();
							if( removeLeftMostDim ) {
								dimensions.add(null);
							} else {
								dimensions.add(oldAS.getDimension(0).clone());
							}
							for( int i=1; i<numDims; i++ ) {
								dimensions.add(oldAS.getDimension(i).clone());
							}
							ArraySpecifier newAS = new ArraySpecifier(dimensions);
							arraySpecs.add(newAS);
						} else {
							arraySpecs.add(obj);
						}
					}
				}
				VariableDeclarator Declr = null;
				if( declarator_specs.isEmpty() ) {
					if( arraySpecs == null || arraySpecs.isEmpty() ) {
						Declr = new VariableDeclarator(id);
					} else {
						Declr = new VariableDeclarator(id, arraySpecs);
					}
				} else if ( arraySpecs == null || arraySpecs.isEmpty() ) {
					Declr = new VariableDeclarator(declarator_specs, id);
				} else {
					Declr = new VariableDeclarator(declarator_specs, id, arraySpecs);
				}

				Declaration decls = new VariableDeclaration(declaration_specs, Declr);
				st.addDeclaration(decls);
				newID = new Identifier(Declr);
			} else if( inSym instanceof NestedDeclarator ) {
				NestedDeclarator nestedSym = (NestedDeclarator)inSym;
				Declarator childDeclarator = nestedSym.getDeclarator();
				List childDeclr_specs = null;
				List childArray_specs = null;
				if( childDeclarator instanceof VariableDeclarator ) {
					VariableDeclarator varSym = (VariableDeclarator)childDeclarator;
					childDeclr_specs = varSym.getSpecifiers();
					childArray_specs = varSym.getArraySpecifiers();
				} else {
					Tools.exit("[ERROR in TransformTools.declareClonedVariable()] nested declarator whose child declarator is also " +
							"nested declarator is not supported yet; exit!\n" +
							"Symbol: " + inSym + "\n");
				}
				VariableDeclarator childDeclr = null;
				if (childDeclr_specs.isEmpty()) {
					if (childArray_specs == null || childArray_specs.isEmpty()) {
						childDeclr = new VariableDeclarator(id);
					} else {
						childDeclr = new VariableDeclarator(id, childArray_specs);
					}
				} else if (childArray_specs == null || childArray_specs.isEmpty()) {
					childDeclr = new VariableDeclarator(childDeclr_specs, id);
				} else {
					childDeclr = new VariableDeclarator(childDeclr_specs, id, childArray_specs);
				}
				specs = nestedSym.getTypeSpecifiers();
				if( removeSpecs != null ) {
					specs.removeAll(removeSpecs);
				}
				if( addSpecs != null ) {
					specs.addAll(addSpecs);
				}
				// Separate declarator/declaration specifiers.
				List declaration_specs = new ArrayList(specs.size());
				List declarator_specs = new ArrayList(specs.size());
				for (int i = 0; i < specs.size(); i++) {
					Object spec = specs.get(i);
					if (spec instanceof PointerSpecifier) {
						declarator_specs.add(spec);
					} else {
						declaration_specs.add(spec);
					}
				}
				arraySpecs = nestedSym.getArraySpecifiers();
				NestedDeclarator nestedDeclr = null;
				if( declarator_specs.isEmpty() ) {
					if( arraySpecs == null || arraySpecs.isEmpty() ) {
						nestedDeclr = new NestedDeclarator(childDeclr);
					} else {
						nestedDeclr = new NestedDeclarator(declarator_specs, childDeclr, null, arraySpecs);
					}
				} else if ( arraySpecs == null || arraySpecs.isEmpty() ) {
					nestedDeclr = new NestedDeclarator(declarator_specs, childDeclr, null);
				} else {
					nestedDeclr = new NestedDeclarator(declarator_specs, childDeclr, null, arraySpecs);
				}

				Declaration decls = new VariableDeclaration(declaration_specs, nestedDeclr);
				st.addDeclaration(decls);
				newID = new Identifier(nestedDeclr);
			}
		}
		return newID;
	}
	
	/**
	 * Clone the input symbol with a new name, {@code name}, and declare the new symbol in the specified scope, where, if
	 * it does not exist. If existing, return the identifier for the existing symbol.
	 * 
	 * @param where
	 * @param inSym
	 * @param name
	 * @param removeSpecs
	 * @return
	 */
	public static Identifier declareClonedArrayVariable(Traversable where, SubArray sArray, String name, List<Specifier> removeSpecs,
			List<Specifier> addSpecs ) {
		NameID id = new NameID(name);
		Identifier newID = null;
		List specs = null;
		List arraySpecs = null;
		SymbolTable st;
		if (where instanceof SymbolTable) {
			st = (SymbolTable)where;
		} else {
			st = IRTools.getAncestorOfType(where, SymbolTable.class);
		}
		if (st instanceof Loop) {
			st = IRTools.getAncestorOfType(st, SymbolTable.class);
		}
		Set<String> symNames = AnalysisTools.symbolsToStringSet(st.getSymbols());
		if( symNames.contains(name) ) {
			VariableDeclaration decl = (VariableDeclaration)st.findSymbol(new NameID(name));
			newID = new Identifier((Symbol)decl.getDeclarator(0));
		} else {
			Symbol inSym = SymbolTools.getSymbolOf(sArray.getArrayName());
			while( inSym instanceof AccessSymbol ) {
				inSym = ((AccessSymbol)inSym).getMemberSymbol();
			}
			List<Expression> startList = new ArrayList<Expression>();
			List<Expression> lengthList = new ArrayList<Expression>();
			if( !AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, false, (Annotatable)where) ) {
				Tools.exit("[ERROR in TransformTools.declareClonedArrayVariable() array dimension information is missing" +
						"for the following symbol: " + sArray.getArrayName() + "\n");
			}
			if( inSym instanceof VariableDeclarator ) {
				VariableDeclarator varSym = (VariableDeclarator)inSym;
				specs = varSym.getTypeSpecifiers();
				if( removeSpecs != null ) {
					specs.removeAll(removeSpecs);
				}
				if( addSpecs != null ) {
					specs.addAll(addSpecs);
				}
				// Separate declarator/declaration specifiers.
				List declaration_specs = new ArrayList(specs.size());
				List declarator_specs = new ArrayList(specs.size());
				for (int i = 0; i < specs.size(); i++) {
					Object spec = specs.get(i);
					if (spec instanceof PointerSpecifier) {
						declarator_specs.add(spec);
					} else {
						declaration_specs.add(spec);
					}
				}
				arraySpecs = new ArrayList<Specifier>();
				if( !lengthList.isEmpty() ) {
					ArraySpecifier aspec = new ArraySpecifier(lengthList);
					arraySpecs.add(aspec);
				}
				VariableDeclarator Declr = null;
				if( declarator_specs.isEmpty() ) {
					if( arraySpecs == null || arraySpecs.isEmpty() ) {
						Declr = new VariableDeclarator(id);
					} else {
						Declr = new VariableDeclarator(id, arraySpecs);
					}
				} else if ( arraySpecs == null || arraySpecs.isEmpty() ) {
					Declr = new VariableDeclarator(declarator_specs, id);
				} else {
					Declr = new VariableDeclarator(declarator_specs, id, arraySpecs);
				}

				Declaration decls = new VariableDeclaration(declaration_specs, Declr);
				st.addDeclaration(decls);
				newID = new Identifier(Declr);
			} else if( inSym instanceof NestedDeclarator ) {
				NestedDeclarator nestedSym = (NestedDeclarator)inSym;
				specs = nestedSym.getTypeSpecifiers();
				if( removeSpecs != null ) {
					specs.removeAll(removeSpecs);
				}
				if( addSpecs != null ) {
					specs.addAll(addSpecs);
				}
				// Separate declarator/declaration specifiers.
				List declaration_specs = new ArrayList(specs.size());
				List declarator_specs = new ArrayList(specs.size());
				for (int i = 0; i < specs.size(); i++) {
					Object spec = specs.get(i);
					if (spec instanceof PointerSpecifier) {
						declarator_specs.add(spec);
					} else {
						declaration_specs.add(spec);
					}
				}
				arraySpecs = new ArrayList<Specifier>();
				if( !lengthList.isEmpty() ) {
					ArraySpecifier aspec = new ArraySpecifier(lengthList);
					arraySpecs.add(aspec);
				}
				VariableDeclarator Declr = null;
				if( declarator_specs.isEmpty() ) {
					if( arraySpecs == null || arraySpecs.isEmpty() ) {
						Declr = new VariableDeclarator(id);
					} else {
						Declr = new VariableDeclarator(id, arraySpecs);
					}
				} else if ( arraySpecs == null || arraySpecs.isEmpty() ) {
					Declr = new VariableDeclarator(declarator_specs, id);
				} else {
					Declr = new VariableDeclarator(declarator_specs, id, arraySpecs);
				}

				Declaration decls = new VariableDeclaration(declaration_specs, Declr);
				st.addDeclaration(decls);
				newID = new Identifier(Declr);
			}
		}
		return newID;
	}
	
	/**
	 * Get a GPU variable with a name, {@code name}, in the input symbol table, {@code targetSymbolTable}.
	 * If it does not exist, a new GPU symbol with that name is created.
	 * 
	 * @param clonedspecs type specifiers for the GPU symbol
	 * @param name name of the GPU symbol
	 * @param targetSymbolTable symbol scope where the GPU symbol is defined.
	 * @param main_TrUnt Main translation unit where acc_init() is called.
	 * @param OpenACCHeaderEndMap 
	 * @return
	 */
	public static VariableDeclaration getGPUVariable(String name, SymbolTable targetSymbolTable, List<Specifier> typeSpecs,
			TranslationUnit main_TrUnt, Map<TranslationUnit, Declaration> OpenACCHeaderEndMap, Expression initExp) {
		VariableDeclaration gpu_decl = null;
		Set<Symbol> symSet = targetSymbolTable.getSymbols();
		Symbol gpu_sym = AnalysisTools.findsSymbol(symSet, name);
		if( gpu_sym != null ) {
			gpu_decl = (VariableDeclaration)gpu_sym.getDeclaration();
		} else {
			// Create a new GPU device variable.
			// The type of the device symbol should be a pointer type 
			VariableDeclarator gpu_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
					new NameID(name));
			if( initExp != null ) {
				gpu_declarator.setInitializer(new Initializer(initExp.clone()));
			}
			else
			{
				gpu_declarator.setInitializer(new Initializer(new NameID("NULL")));
			}
			List<Specifier> clonedspecs = new ChainedList<Specifier>();
			clonedspecs.addAll(typeSpecs);
			gpu_decl = new VariableDeclaration(clonedspecs, 
					gpu_declarator);
			TranslationUnit parentTrUnt = null;
			if( targetSymbolTable instanceof TranslationUnit ) {
				parentTrUnt = (TranslationUnit)targetSymbolTable;
			} else {
				Traversable tt = targetSymbolTable;
				while( tt != null ) {
					if( tt instanceof TranslationUnit ) {
						parentTrUnt = ((TranslationUnit)tt);
						break;
					} else {
						tt = tt.getParent();
					}
				}
			}
			if( targetSymbolTable instanceof TranslationUnit ) {
				symSet = main_TrUnt.getSymbols();
				gpu_sym = AnalysisTools.findsSymbol(symSet, name);
				if( gpu_sym == null ) {
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
						parentTrUnt.addDeclarationAfter(tLastCudaDecl, gpu_decl);
						OpenACCHeaderEndMap.put(parentTrUnt, gpu_decl);
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
					parentTrUnt.addDeclarationAfter(tLastCudaDecl, gpu_decl);
					OpenACCHeaderEndMap.put(parentTrUnt, gpu_decl);
				}
			} else {
				targetSymbolTable.addDeclaration(gpu_decl);
			}
		}
		return gpu_decl;
	}

	public static ForLoop stripmining(ForLoop ploop, Expression stripSize, long suffix, CompoundStatement targetRegion) {
		Procedure cProc = IRTools.getParentProcedure(ploop);
		////////////////////////////////////////////////////////////////
		// Original loop :                                            //
		//     for( k = LB; k <= UB; k++ ) { }                        //
		// Stripmined loop:                                           //
		//     for( i = 0; i < stripSize; i++ ) {                     //
		//          for( k = i+LB; k <= UB; k += stripSize ) { }      //
		//     }                                                      //
		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////
		// Check the increment of the input loop is 1 or not. //
		////////////////////////////////////////////////////////
		Expression incrExp = LoopTools.getIncrementExpression(ploop);
		if( incrExp instanceof IntegerLiteral ) {
			if( ((IntegerLiteral)incrExp).getValue() != 1 ) {
				Tools.exit("[ERROR in ACC2GPUTranslator.worksharingLoopUnrolling()] Stripmining is not applicable " +
						"to a worksharing loop whose increment is not 1; exit!\nEnclosing Procedure: " + 
						cProc.getSymbolName() + "\nFor loop: \n" + ploop + "\n");
			}
		} else {
			Tools.exit("[ERROR in ACC2GPUTranslator.worksharingLoopUnrolling()] Stripmining is not applicable " +
					"to a worksharing loop whose increment is not 1; exit!\nEnclosing Procedure: " + 
					cProc.getSymbolName() + "\nFor loop: \n" + ploop + "\n");
		}
		CompoundStatement forBody = new CompoundStatement();
		Identifier index = null;
		if( targetRegion != null ) {
			index = TransformTools.getTempIndex(targetRegion, suffix);
		} else {
			index = TransformTools.getTempIndex(forBody, suffix);
		}
		Expression expr1 = new AssignmentExpression(index, AssignmentOperator.NORMAL,
				new IntegerLiteral(0));
		Statement initStmt = new ExpressionStatement(expr1);
		expr1 = new BinaryExpression((Identifier)index.clone(), BinaryOperator.COMPARE_LT,
				stripSize.clone());
		Expression expr2 = new UnaryExpression(
				UnaryOperator.POST_INCREMENT, (Identifier)index.clone());
		ForLoop wLoop = new ForLoop(initStmt, expr1, expr2, forBody);
		// Swap the new loop (wLoop) with the old loop (ploop).
		wLoop.swapWith(ploop);
		expr1 = new BinaryExpression((Identifier)index.clone(), BinaryOperator.ADD,
				LoopTools.getLowerBoundExpression(ploop));
		Expression oldindex = LoopTools.getIndexVariable(ploop);
		expr2 = new AssignmentExpression((Expression)oldindex.clone(), AssignmentOperator.NORMAL,
				expr1);
		initStmt = new ExpressionStatement(expr2);
		ploop.getInitialStatement().swapWith(initStmt);
		/*					expr1 = new BinaryExpression((Expression)oldindex.clone(), BinaryOperator.COMPARE_LE,
							LoopTools.getUpperBoundExpression(ploop));
					ploop.getCondition().swapWith(expr1);*/
		expr2 = new AssignmentExpression((Expression)oldindex.clone(), AssignmentOperator.ADD,
				stripSize.clone());
		ploop.getStep().swapWith(expr2);
		forBody.addStatement(ploop);
		///////////////////////////////////////////////
		// Move all Annotations of ploop into wLoop. //
		///////////////////////////////////////////////
		List<Annotation> annot_list = ploop.getAnnotations();
		for( Annotation tAnnot : annot_list ) {
			wLoop.annotate(tAnnot);
		}
		ploop.removeAnnotations();
		if( wLoop.containsAnnotation(ACCAnnotation.class, "gang") && !wLoop.containsAnnotation(ACCAnnotation.class, "worker")) {
			//If the original ploop was a pure gang loop and had a private clause, the new index variable should be included too.
			//[DEBUG] disabled since it will be handled by later priavizeTransformation pass. 
/*			ACCAnnotation privAnnot = wLoop.getAnnotation(ACCAnnotation.class, "private");
			if( privAnnot !=null ) {
				Set<SubArray> privASet = (Set<SubArray>)privAnnot.get("private");
				privASet.add(AnalysisTools.createSubArray(index.getSymbol(), true, null));
				privAnnot = wLoop.getAnnotation(ACCAnnotation.class, "accprivate");
				if( privAnnot != null ) {
					Set<Symbol> privSSet = (Set<Symbol>)privAnnot.get("accprivate");
					privSSet.add(index.getSymbol());
				}
			}*/
			//If the original ploop was a pure gang loop, add "innergang" internal clause to the inner loop 
			//to be used in a later privatization pass.
			ACCAnnotation iAnnot = new ACCAnnotation("internal", "_directive");
			iAnnot.put("innergang", "_clause");
			iAnnot.setSkipPrint(true);
			ploop.annotate(iAnnot);
		}
		return wLoop;
	}
	
	/**
	 * Create a reduction assignment expression for the given reduction operator.
	 * This function is used to perform both in-block partial reduction and across-
	 * block final reduction.
	 * [CAUTION] the partial results of a subtraction reduction are added to form the 
	 * final value.
	 * 
	 * @param RedExp expression of reduction variable/array
	 * @param redOp reduction operator
	 * @param Rexp right-hand-side expression
	 * @return reduction assignment expression
	 */
	public static AssignmentExpression RedExpression(Expression RedExp, ReductionOperator redOp,
			Expression Rexp) {
		AssignmentExpression assignExp = null;
		if( redOp.equals(ReductionOperator.ADD) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.ADD,
					Rexp);
		}else if( redOp.equals(ReductionOperator.BITWISE_INCLUSIVE_OR) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.BITWISE_INCLUSIVE_OR,
					Rexp);
		}else if( redOp.equals(ReductionOperator.BITWISE_EXCLUSIVE_OR) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.BITWISE_EXCLUSIVE_OR,
					Rexp);
		}else if( redOp.equals(ReductionOperator.MULTIPLY) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.MULTIPLY,
					Rexp);
		}else if( redOp.equals(ReductionOperator.BITWISE_AND) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.BITWISE_AND,
					Rexp);
		}else if( redOp.equals(ReductionOperator.LOGICAL_AND) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.NORMAL,
					new BinaryExpression((Expression)RedExp.clone(), redOp.getBinaryOperator(), Rexp));
		}else if( redOp.equals(ReductionOperator.LOGICAL_OR) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.NORMAL,
					new BinaryExpression((Expression)RedExp.clone(), redOp.getBinaryOperator(), Rexp));
		}else if( redOp.equals(ReductionOperator.MIN) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.NORMAL, 
					new MinMaxExpression(true, RedExp.clone(), Rexp));
		}else if( redOp.equals(ReductionOperator.MAX) ) {
			assignExp = new AssignmentExpression( RedExp, AssignmentOperator.NORMAL, 
					new MinMaxExpression(false, RedExp.clone(), Rexp));
		}
		return assignExp;
			
	}
	
	/**
	 * Find appropriate initialization value for a given reduction operator
	 * and variable type.
	 * @param redOp reductioin operator
	 * @param specList list containing type specifiers of the reduction variable
	 * @return initialization value for the reduction variable
	 */
	public static Expression getRInitValue(ReductionOperator redOp, List specList) {
		///////////////////////////////////////////////////////
		// Operator		Initialization value                 //
		///////////////////////////////////////////////////////
		//	+			0
		//	*			1
		//	&			~0
		//	|			0
		//	^			0
		//	&&			1
		//	||			0
		//  min         largest
		//  max         least
		///////////////////////////////////////////////////////
		Expression initValue = null;
		if( redOp.equals(ReductionOperator.ADD) ) {
			if(specList.contains(Specifier.FLOAT) || specList.contains(Specifier.DOUBLE)) {
				initValue = new FloatLiteral(0.0f, "F");
			} else {
				initValue = new IntegerLiteral(0);
			}
		} else if( redOp.equals(ReductionOperator.BITWISE_INCLUSIVE_OR)
				|| redOp.equals(ReductionOperator.BITWISE_EXCLUSIVE_OR)
				|| redOp.equals(ReductionOperator.LOGICAL_OR) ) {
			initValue = new IntegerLiteral(0);
		} else if( redOp.equals(ReductionOperator.MULTIPLY) ) {
			if(specList.contains(Specifier.FLOAT) || specList.contains(Specifier.DOUBLE)) {
				initValue = new FloatLiteral(1.0f, "F");
			} else {
				initValue = new IntegerLiteral(1);
			}
		} else if( redOp.equals(ReductionOperator.LOGICAL_AND) ) {
			initValue = new IntegerLiteral(1);
		} else if( redOp.equals(ReductionOperator.BITWISE_AND) ) {
			initValue = new UnaryExpression(UnaryOperator.BITWISE_COMPLEMENT, 
					new IntegerLiteral(0));
		} else if( redOp.equals(ReductionOperator.MIN) ) {
			if(specList.contains(Specifier.FLOAT)) {
				initValue = new NameID("FLT_MAX");
			} else if(specList.contains(Specifier.DOUBLE)) {
				initValue = new NameID("DBL_MAX");
			} else if(specList.contains(Specifier.LONG)) {
				initValue = new NameID("LONG_MAX");
			} else {
				initValue = new NameID("INT_MAX");
			}
		} else if( redOp.equals(ReductionOperator.MAX) ) {
			if(specList.contains(Specifier.FLOAT)) {
				initValue = new NameID("FLT_MIN");
			} else if(specList.contains(Specifier.DOUBLE)) {
				initValue = new NameID("DBL_MIN");
			} else if(specList.contains(Specifier.LONG)) {
				initValue = new NameID("LONG_MIN");
			} else {
				initValue = new NameID("INT_MIN");
			}
		}
		return initValue;
	}
	
	
	/**
	 * Build a string name of an AccessExpression, which can be used as a valid C variable name.
	 * 
	 * @param accExp
	 * @return
	 */
	public static String buildAccessSymbolName( AccessSymbol accSym ) {
		Symbol base = accSym.getBaseSymbol();
		Symbol member = accSym.getMemberSymbol();
		StringBuilder sb = new StringBuilder(32);
		if( base instanceof AccessSymbol ) {
			sb.append(buildAccessSymbolName((AccessSymbol)base));
		} else if( base instanceof DerefSymbol ) {
			while( base instanceof DerefSymbol ) {
				base = ((DerefSymbol)base).getRefSymbol();
			}
			if( base instanceof AccessSymbol ) {
				sb.append(buildAccessSymbolName((AccessSymbol)base));
			} else {
				sb.append(base.getSymbolName());
			}
		} else {
			sb.append(base.getSymbolName());
		}
		sb.append("__");
		if( member instanceof AccessSymbol ) {
			sb.append(buildAccessSymbolName((AccessSymbol)member));
		} else {
			sb.append(member.getSymbolName());
		}
		return sb.toString();
	}
	
	/**
	 * Build a string name of an AccessExpression, which can be used as a valid C variable name.
	 * 
	 * @param accExp
	 * @return
	 */
	public static String buildAccessExpressionName( AccessExpression accExp ) {
		AccessSymbol accSym = new AccessSymbol(accExp);
		return buildAccessSymbolName(accSym);
	}
	
    /**
    * Replaces all instances of expression <var>x</var> on the IR tree
    * beneath <var>t</var> by <i>clones of</i> expression <var>y</var>.
    * Skips the immediate right hand side of member access expressions.
    * This method differs from IRTools.replaceAll() in that this does not
    * replace declarator ID. If input expression <var>x</var> is 
    * an Identifier and * input Traversable <var>t</var> contains the 
    * declarator for the input identifier <var>x</var>, and if <var>t</var> 
    * also contains the same identifier in other IR tree, 
    * getting the identifier's name will cause infinite loops.
    *
    * @param t The location at which to start the search.
    * @param x The expression to be replaced.
    * @param y The expression to substitute.
    */
    public static void replaceAll(Traversable t, Expression x, Expression y) {
        List<Expression> matches = IRTools.findExpressions(t, x);
        for (int i = 0; i < matches.size(); i++) {
            Expression match = matches.get(i);
            Traversable parent = match.getParent();
            if (parent instanceof AccessExpression &&
                    match == ((AccessExpression)parent).getRHS()) {
                /* don't replace these */
            } else if ( parent instanceof Declarator) {
                /* don't replace NameID inside of symbol declarator */
            } else {
                match.swapWith(y.clone());
            }
        }
        /*
        BreadthFirstIterator iter = new BreadthFirstIterator(t);
        for (;;) {
            Expression o = null;
            try {
                o = (Expression) iter.next(x.getClass());
            }
            catch(NoSuchElementException e) {
                break;
            }
            if (o.equals(x)) {
                if (o.getParent()instanceof AccessExpression
                        && ((AccessExpression) o.getParent()).getRHS() == o) {
                    // don't replace these
                } else {
                    if (o.getParent() == null) {
                        System.err.println("[ERROR] this " + o.toString() +
                                           " should be on the tree");
                    }
                    Expression copy = y.clone();
                    o.swapWith(copy);
                    if (copy.getParent() == null) {
                        System.err.println("[ERROR] " + y.toString() +
                                           " didn't get put on tree properly");
                    }
                }
            }
        }
        */
    }
	
    /**
    * Replaces all instances of expression, whose symbol is an AccessSymbol <var>accSym</var>, on the IR tree
    * beneath <var>t</var> by <i>clones of</i> expression <var>y</var>.
    * Skips the immediate right hand side of member access expressions.
    *
    * @param t The location at which to start the search.
    * @param accSym The access symbol whose derived expressions will be replaced.
    * @param y The expression to substitute.
    */
    public static void replaceAccessExpressions(Traversable t, AccessSymbol accSym, Expression y) {
        List<Expression> matches = new ArrayList<Expression>(4);
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        while (iter.hasNext()) {
            Expression child = iter.next();
            if (SymbolTools.getSymbolOf(child).equals(accSym)) {
                matches.add(child);
            }
        }
        for (int i = 0; i < matches.size(); i++) {
            Expression match = matches.get(i);
            Traversable parent = match.getParent();
            if (parent instanceof AccessExpression &&
                    match == ((AccessExpression)parent).getRHS()) {
                /* don't replace these */
            } else {
            	Expression tExp = match;
            	while( tExp instanceof AccessExpression ) {
            		tExp = ((AccessExpression)tExp).getRHS();
            	}
            	Expression newExp = y.clone();
            	if( tExp instanceof ArrayAccess ) {
            		List<Expression> indices = ((ArrayAccess)tExp).getIndices();
            		List<Expression> newIndices = new ArrayList<Expression>(indices.size());
            		for( Expression index : indices ) {
            			newIndices.add(index.clone());
            		}
            		newExp = new ArrayAccess(y.clone(), newIndices);
            	}
                match.swapWith(newExp);
            }
        }
    }
    
    /**
     * Generate an array copy statement.
     * Ex: 
	 *     for(i=0; i<SIZE1; i++) {                     
	 *	        for(k=0; k<SIZE2; k++) {                 
	 *	             lpriv_x[_bid][i][k] = lfpriv_x[i][k];
	 *	        }                                         
	 *	   }                                              
     * 
     * @param index_vars
     * @param lengthList
     * @param LHS
     * @param RHS
     * @return
     */
    public static Statement genArrayCopyLoop(List<Identifier> index_vars, List<Expression> lengthList, 
    		Expression LHS, Expression RHS) {
    	Statement estmt = null;
    	Identifier index_var = null;
    	Expression assignex = null;
    	Statement loop_init = null;
    	Expression condition = null;
    	Expression step = null;
    	CompoundStatement loop_body = null;
    	ForLoop innerLoop = null;
    	int dimsize = lengthList.size();
    	if( dimsize == 0 ) {
    		estmt = new ExpressionStatement( new AssignmentExpression(LHS.clone(),
    						AssignmentOperator.NORMAL, RHS.clone()));
    	} else {
    		for( int i=dimsize-1; i>=0; i-- ) {
    			index_var = index_vars.get(i);
    			assignex = new AssignmentExpression((Identifier)index_var.clone(),
    					AssignmentOperator.NORMAL, new IntegerLiteral(0));
    			loop_init = new ExpressionStatement(assignex);
    			condition = new BinaryExpression(index_var.clone(),
    					BinaryOperator.COMPARE_LT, lengthList.get(i).clone());
    			step = new UnaryExpression(UnaryOperator.POST_INCREMENT, 
    					(Identifier)index_var.clone());
    			loop_body = new CompoundStatement();
    			if( i == (dimsize-1) ) {
    				assignex = new AssignmentExpression(LHS.clone(),
    						AssignmentOperator.NORMAL, RHS.clone());
    				loop_body.addStatement(new ExpressionStatement(assignex));
    			} else {
    				loop_body.addStatement(innerLoop);
    			}
    			innerLoop = new ForLoop(loop_init, condition, step, loop_body);
    		}
    		estmt = innerLoop;
    	}
    	return estmt;
    }
    
    /**
     * Generate a loop performing reduction.
     * Ex: 
	 *     for(i=0; i<SIZE1; i++) {                     
	 *	        for(k=0; k<SIZE2; k++) {                 
	 *	             lpriv_x[i][k][_tid] += lpriv_x[i][k][_tid + s];
	 *	        }                                         
	 *	   }                                              
     * 
     * @param index_vars
     * @param lengthList
     * @param LHS
     * @param RHS
     * @param redOp
     * @return
     */
    public static Statement genReductionLoop(List<Identifier> index_vars, List<Expression> lengthList, 
    		Expression LHS, Expression RHS, ReductionOperator redOp) {
    	Statement estmt = null;
    	Identifier index_var = null;
    	Expression assignex = null;
    	Statement loop_init = null;
    	Expression condition = null;
    	Expression step = null;
    	CompoundStatement loop_body = null;
    	ForLoop innerLoop = null;
    	int dimsize = lengthList.size();
    	if( dimsize == 0 ) {
    		estmt = new ExpressionStatement( RedExpression(LHS.clone(), redOp, RHS.clone()));
    	} else {
    		for( int i=dimsize-1; i>=0; i-- ) {
    			index_var = index_vars.get(i);
    			assignex = new AssignmentExpression((Identifier)index_var.clone(),
    					AssignmentOperator.NORMAL, new IntegerLiteral(0));
    			loop_init = new ExpressionStatement(assignex);
    			condition = new BinaryExpression(index_var.clone(),
    					BinaryOperator.COMPARE_LT, lengthList.get(i).clone());
    			step = new UnaryExpression(UnaryOperator.POST_INCREMENT, 
    					(Identifier)index_var.clone());
    			loop_body = new CompoundStatement();
    			if( i == (dimsize-1) ) {
    				assignex = RedExpression(LHS.clone(), redOp, RHS.clone());
    				loop_body.addStatement(new ExpressionStatement(assignex));
    			} else {
    				loop_body.addStatement(innerLoop);
    			}
    			innerLoop = new ForLoop(loop_init, condition, step, loop_body);
    		}
    		estmt = innerLoop;
    	}
    	return estmt;
    }
    
    /**
     * Create result-compare statements and insert them after {@code refStmt} if it is not null.
     * 
     * @param cProc 
     * @param refStmt
     * @param hostVar
     * @param compVar
     * @param pitchID
     * @param lengthList
     * @param specs
     * @param cAnnot
     * @param EPSILON
     */
    public static List<Statement> genResultCompareCodes(Procedure cProc, Statement refStmt, Expression hostVar, Expression compVar, 
    		Identifier pitchID, List<Expression> lengthList, List<Specifier> specs, ACCAnnotation cAnnot,
    		FloatLiteral EPSILON, boolean initCompVarAddress, FloatLiteral minCheckValue) {
    	CompoundStatement cBody = cProc.getBody();
    	Annotatable at = cAnnot.getAnnotatable();
    	ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "refname");
    	String refname = null;
    	if( iAnnot != null ) {
    		refname = iAnnot.get("refname");
    	}
    	////////////////////////////////////////////////////////////////////////////////////////////////////
    	//If current compute region contains resilience repeat clause, ftout__a is used instead of a for  //
    	//host-side reference output value.                                                               //
    	////////////////////////////////////////////////////////////////////////////////////////////////////
		//Check repeat clause in resilience region.
		boolean containsRepeatClause = false;
		ARCAnnotation tAnnot = at.getAnnotation(ARCAnnotation.class, "resilience");
		if( (tAnnot != null) && tAnnot.containsKey("repeat") ) {
			Expression ftcond = tAnnot.get("ftcond");
			if( (ftcond ==null) || !(ftcond instanceof IntegerLiteral) 
					|| (((IntegerLiteral)ftcond).getValue() != 0) ) {
				containsRepeatClause = true;
			}
		}
		Expression refHostVar = hostVar;
		Expression totalNumErrors = null;
		if( containsRepeatClause ) {
			StringBuilder str = new StringBuilder(80);
			str.append("ftout__");
			if( hostVar instanceof AccessExpression ) {
				str.append(TransformTools.buildAccessExpressionName((AccessExpression)hostVar));
			} else {
				str.append(hostVar.toString());
			}
			Set<Symbol> symSet = cBody.getSymbols();
			Symbol ftout_sym = AnalysisTools.findsSymbol(symSet, str.toString());
			if( ftout_sym != null ) {
				refHostVar = new Identifier(ftout_sym);
			}
			Symbol tNESym = AnalysisTools.findsSymbol(symSet, "_ti_totalfaults");
			if( tNESym != null ) {
				totalNumErrors = new Identifier(tNESym);
			}
		}
		
    	List<Statement> Stmts = new ArrayList<Statement>();
    	// Example code:
    	// HI_get_temphost_address(&hostVar,, (void **)compVar, 1);
    	// double comp__var = (double)*compVar;
    	// double ref__var = (double)hostVar;
    	// double diff__var = fabs(comp__var - ref__var);
    	// if( ((ref__var != 0.0) && ((diff__var/fabs(ref__var)) > EPSILON)) || ((ref__var == 0.0) && (fabs(diff__var) > EPSILON)) ) { 
    	//     printf("[Kernel Verification Error] Values of variable a differ:\nCPU output: %lf\nGPU output: %lf
    	//     OpenACC Annotation: ...\nEnclosing Procedure: cProc\n", hostVar, comp__var); 
    	//     printf("Exit the kernel verification test!\n");
    	//     acc_shutdown(acc_device_nvidia);
    	//     exit(1);
    	// }
    	//
    	// foundDiff = 0;
    	// HI_get_temphost_address(&hostVar,, (void **)compVar, 1);
    	// double ref__var;
    	// double comp__var;
    	// double diff__var;
    	// for( temp_i = 0; temp_i < SIZE1; temp_i++ ) {
    	//     if( foundDiff == 1 ) { break; }
    	//     for( temp_k = 0; temp_k < SIZE2; temp_k++ ) {
    	//         ref__var = (double)hostVar[temp_i][temp_k];
    	//         comp__var = (double)compVar[temp_i*SIZE2+temp_k];
    	//         diff__var = fabs(comp__var - ref__var);
    	//         if( ((ref__var != 0.0) && ((diff__var/fabs(ref__var)) > EPSILON)) || ((ref__var == 0.0) && (fabs(diff__var) > EPSILON)) ) { 
    	//                 printf("[Kernel Verification Error] Values of variable a[%d][%d] differ:\nCPU output: %lf\nGPU output: %lf
    	//                 OpenACC Annotation: ...\nEnclosing Procedure: cProc\n", temp_i, temp_k, ref__var, comp__var); 
    	//                 foundDiff = 1;
    	//                 break;
    	//         } 
    	//     }
    	// }
    	// if( foundDiff == 1 ) {
    	//     printf("Exit the kernel verification test\n");
    	//     acc_shutdown(acc_device_nvidia);
    	//     exit(1);
    	// }
    	
    	// int foundDiff;
    	List<Specifier> tSpecs = new ArrayList<Specifier>(1);
    	tSpecs.add(Specifier.INT);
    	Identifier foundDiff = getTempScalar(cBody, tSpecs, null, 0);
    	// double comp__var;
    	tSpecs = new ArrayList<Specifier>(1);
    	tSpecs.add(Specifier.DOUBLE);
    	Identifier comp__var = getTempScalar(cBody, tSpecs, null, 1);
    	// double ref__var;
    	tSpecs = new ArrayList<Specifier>(1);
    	tSpecs.add(Specifier.DOUBLE);
    	Identifier ref__var = getTempScalar(cBody, tSpecs, null, 2);
    	// double diff__var;
    	tSpecs = new ArrayList<Specifier>(1);
    	tSpecs.add(Specifier.DOUBLE);
    	Identifier diff__var = getTempScalar(cBody, tSpecs, null, 3);
    	Literal ZERO = new FloatLiteral(0.0);
    	int dimsize = lengthList.size();
    	if( dimsize == 0 ) { //scalar variable
    		if( initCompVarAddress ) {
    			// HI_get_temphost_address(&hostVar,, (void **)&compVar, 1);
    			FunctionCall fCall = new FunctionCall(new NameID("HI_get_temphost_address"));
    			fCall.addArgument(new UnaryExpression(UnaryOperator.ADDRESS_OF, hostVar.clone()));
    			tSpecs = new ArrayList<Specifier>();
    			tSpecs.add(Specifier.VOID);
    			tSpecs.add(PointerSpecifier.UNQUALIFIED);
    			tSpecs.add(PointerSpecifier.UNQUALIFIED);
    			Typecast tcast = new Typecast(tSpecs, new UnaryExpression(UnaryOperator.ADDRESS_OF, compVar.clone()));
    			fCall.addArgument(tcast);
    			fCall.addArgument(new IntegerLiteral(1));
    			Stmts.add(new ExpressionStatement(fCall));
    		}
    		//comp_var = (double)(*compVar);
    		tSpecs = new ArrayList<Specifier>(1);
    		tSpecs.add(Specifier.DOUBLE);
    		Typecast tcast = new Typecast(tSpecs, new UnaryExpression(UnaryOperator.DEREFERENCE, compVar.clone()));
    		Stmts.add(new ExpressionStatement(new AssignmentExpression(comp__var.clone(), AssignmentOperator.NORMAL,
    				tcast)));
    		//ref__var = (double)hostVar;
    		tSpecs = new ArrayList<Specifier>(1);
    		tSpecs.add(Specifier.DOUBLE);
    		tcast = new Typecast(tSpecs, refHostVar.clone());
    		Stmts.add(new ExpressionStatement(new AssignmentExpression(ref__var.clone(), AssignmentOperator.NORMAL,
    				tcast)));
    		//diff__var = fabs(comp__var - ref__var);
    		FunctionCall fCall = new FunctionCall(new NameID("fabs"));
    		fCall.addArgument(new BinaryExpression(comp__var.clone(), BinaryOperator.SUBTRACT, ref__var.clone()));
    		Stmts.add(new ExpressionStatement(new AssignmentExpression(diff__var.clone(), AssignmentOperator.NORMAL, 
    				fCall)));
    	    // if( ((ref__var != 0.0) && ((diff__var/fabs(ref__var)) > EPSILON)) || ((ref__var == 0.0) && (fabs(diff__var) > EPSILON)) ) { 
    		//     printf("[Kernel Verification Error] Values of variable a differ:\nCPU output: %lf\nGPU output: %lf\n
    		//     OpenACC Annotation: ...\nEnclosing Procedure: cProc\n", ref__var, comp__var); 
    		//     printf("Exit the kernel verification test!\n");
    		//     acc_shutdown(acc_device_nvidia);
    		//     exit(1);
    		// } 
    		Expression condExp1 = new BinaryExpression(new BinaryExpression(ref__var.clone(), BinaryOperator.COMPARE_NE, ZERO.clone()),
    				BinaryOperator.LOGICAL_AND,
    				new BinaryExpression(new BinaryExpression(diff__var.clone(),BinaryOperator.DIVIDE,
    						new FunctionCall(new NameID("fabs"),ref__var.clone())), 
    						BinaryOperator.COMPARE_GT, EPSILON.clone()));
 /*   		Expression condExp2 = new BinaryExpression(new BinaryExpression(comp__var.clone(), BinaryOperator.COMPARE_NE, ZERO.clone()),
    				BinaryOperator.LOGICAL_AND,
    				new BinaryExpression(new BinaryExpression(diff__var.clone(),BinaryOperator.DIVIDE,comp__var.clone()), 
    						BinaryOperator.COMPARE_GT, EPSILON.clone()));*/
//    		Expression condExp2 = new BinaryExpression(diff__var.clone(), BinaryOperator.COMPARE_GT, EPSILON.clone());
    		Expression condExp2 = new BinaryExpression(new BinaryExpression(ref__var.clone(), BinaryOperator.COMPARE_EQ, ZERO.clone()),
    				BinaryOperator.LOGICAL_AND,
    				new BinaryExpression(new FunctionCall(new NameID("fabs"), diff__var.clone()), BinaryOperator.COMPARE_GT, EPSILON.clone()));
    		CompoundStatement ifBody = new CompoundStatement();
    		fCall = new FunctionCall(new NameID("printf"));
    		StringLiteral str =  null;
    		if( refname == null ) {
    			str = new StringLiteral("[Kernel Verification Error] Values of variable " + hostVar.toString() + 
    					" differ:\\nCPU output: %E\\nGPU output: %E\\n" +
    					"OpenACC Annotation: " + cAnnot + "\\nEnclosing Procedure: " + cProc.getSymbolName() + "\\n" +
    							"(Acceptable margin of errors = " + EPSILON + ")\\n");
    			if( minCheckValue != null ) {
    				str = new StringLiteral(str.getValue() + "(Values smaller than " + minCheckValue + " are ignored.)\\n");
    			}
    		} else {
    			str = new StringLiteral("[Kernel Verification Error] Values of variable " + hostVar.toString() + 
    					" differ:\\nCPU output: %E\\nGPU output: %E\\n" +
    					"GPU kernel name: " + refname + "\\n" +
    							"(Acceptable margin of errors = " + EPSILON + ")\\n");
    			if( minCheckValue != null ) {
    				str = new StringLiteral(str.getValue() + "(Values smaller than " + minCheckValue + " are ignored.)\\n");
    			}
    		}
    		fCall.addArgument(str);
    		fCall.addArgument(ref__var.clone());
    		fCall.addArgument(comp__var.clone());
    		ifBody.addStatement(new ExpressionStatement(fCall));
    		if( totalNumErrors == null ) {
    			fCall = new FunctionCall(new NameID("printf"));
    			str = new StringLiteral("Exit the kernel verification test!\\n");
    			fCall.addArgument(str);
    			ifBody.addStatement(new ExpressionStatement(fCall));
    			fCall = new FunctionCall(new NameID("acc_shutdown"));
    			fCall.addArgument(new NameID("acc_device_nvidia"));
    			ifBody.addStatement(new ExpressionStatement(fCall));
    			fCall = new FunctionCall(new NameID("exit"));
    			fCall.addArgument(new IntegerLiteral(1));
    			ifBody.addStatement(new ExpressionStatement(fCall));
    		} else {
    			Statement expStmt = new ExpressionStatement(new AssignmentExpression(totalNumErrors.clone(), AssignmentOperator.ADD,
    					new IntegerLiteral(1)));
    			ifBody.addStatement(expStmt);
    		}
    		Statement tIfStmt = new IfStatement(new BinaryExpression(condExp1, BinaryOperator.LOGICAL_OR, condExp2), ifBody);
    		//If minCheckExp is set, error-checking will be done only if both CPU and GPU values are greater than or 
    		//equal to the minimum value.
    		if( minCheckValue == null ) {
    			Stmts.add(tIfStmt);
    		} else {
    			Expression minCheckExp = new BinaryExpression( 
    					new BinaryExpression(ref__var.clone(), BinaryOperator.COMPARE_GE, minCheckValue.clone()), 
    					BinaryOperator.LOGICAL_AND,
    					new BinaryExpression(comp__var.clone(), BinaryOperator.COMPARE_GE, minCheckValue.clone())); 
    			Stmts.add(new IfStatement(minCheckExp, tIfStmt));
    		}
    	} else {
    		// foundDiff = 0;
    		Stmts.add(new ExpressionStatement(new AssignmentExpression(foundDiff.clone(), AssignmentOperator.NORMAL,
    				new IntegerLiteral(0))));
    		FunctionCall fCall = null;
    		Typecast tcast = null;
    		if( initCompVarAddress ) {
    			// HI_get_temphost_address(hostVar,, (void **)&compVar, 1);
    			fCall = new FunctionCall(new NameID("HI_get_temphost_address"));
    			fCall.addArgument(hostVar.clone());
    			tSpecs = new ArrayList<Specifier>();
    			tSpecs.add(Specifier.VOID);
    			tSpecs.add(PointerSpecifier.UNQUALIFIED);
    			tSpecs.add(PointerSpecifier.UNQUALIFIED);
    			tcast = new Typecast(tSpecs, new UnaryExpression(UnaryOperator.ADDRESS_OF, compVar.clone()));
    			fCall.addArgument(tcast);
    			fCall.addArgument(new IntegerLiteral(1));
    			Stmts.add(new ExpressionStatement(fCall));
    		}
    		// for( temp_i = 0; temp_i < SIZE1; temp_i++ ) {
    		//     if( foundDiff == 1 ) { break; }
    		//     for( temp_k = 0; temp_k < SIZE2; temp_k++ ) {
    		//         ref__var = (double)hostVar[temp_i][temp_k];
    		//         comp__var = (double)compVar[temp_i*SIZE2+temp_k];
    		//         diff__var = fabs(comp__var - ref__var);
    	    //         if( ((ref__var != 0.0) && ((diff__var/fabs(ref__var)) > EPSILON)) || ((ref__var == 0.0) && (fabs(diff__var) > EPSILON)) ) { 
    		//                 printf("[Kernel Verification Error] Values of variable a[%d][%d] differ:\nCPU output: %lf\nGPU output: %lf
    		//                 OpenACC Annotation: ...\nEnclosing Procedure: cProc\n", temp_i, temp_k, ref__var, comp__var); 
    		//                 foundDiff = 1;
    		//                 break;
    		//         } 
    		//     }
    		// }
    		// if( foundDiff == 1 ) {
    		//     printf("Exit the kernel verification test\n");
    		//     acc_shutdown(acc_device_nvidia);
    		//     exit(1);
    		// }
    		
    		// Create temp indices.
    		List<Identifier> index_vars = new LinkedList<Identifier>();
    		for( int i=0; i<dimsize; i++ ) {
    			index_vars.add(TransformTools.getTempIndex(cBody, tempIndexBase+i));
    		}
    		// Create a nested loop to compare each element
    		ForLoop innerLoop = null;
    		for( int i=dimsize-1; i>=0; i-- ) {
    			Identifier index_var = index_vars.get(i);
    			Expression assignex = new AssignmentExpression((Identifier)index_var.clone(),
    					AssignmentOperator.NORMAL, new IntegerLiteral(0));
    			Statement loop_init = new ExpressionStatement(assignex);
    			Expression condition = new BinaryExpression(index_var.clone(),
    					BinaryOperator.COMPARE_LT, lengthList.get(i).clone());
    			Expression step = new UnaryExpression(UnaryOperator.POST_INCREMENT, 
    					(Identifier)index_var.clone());
    			CompoundStatement loop_body = new CompoundStatement();
    			if( i == (dimsize-1) ) { //innermost loop
    				// comp__var = (double)compVar[temp_i*SIZE2+temp_k];
    				Expression indexEx = null;
					for( int k=0; k<dimsize; k++ ) {
						Expression tExp = null;
						if( k+1 < dimsize ) {
							tExp = lengthList.get(k+1).clone();
							for( int m=k+2; m<dimsize; m++ ) {
								tExp = new BinaryExpression(tExp, BinaryOperator.MULTIPLY, lengthList.get(m).clone());
							} 
							tExp = new BinaryExpression(index_vars.get(k).clone(), BinaryOperator.MULTIPLY, tExp); 
						} else {
							tExp = index_vars.get(k).clone();
						}
						if( indexEx == null ) {
							indexEx = tExp;
						} else {
							indexEx = new BinaryExpression(indexEx, BinaryOperator.ADD, tExp);
						}
					}
					ArrayAccess aAccess = new ArrayAccess(compVar.clone(), indexEx);
    				tSpecs = new ArrayList<Specifier>(1);
    				tSpecs.add(Specifier.DOUBLE);
    				tcast = new Typecast(tSpecs, aAccess);
    				Statement tStmt = new ExpressionStatement(new AssignmentExpression(comp__var.clone(), AssignmentOperator.NORMAL,
    						tcast));
    				loop_body.addStatement(tStmt);
    				// ref__var = (double)hostVar[temp_i][temp_k];
    				if( refHostVar.equals(hostVar) ) {
    					List<Expression> indices1 = new LinkedList<Expression>();
    					for( int k=0; k<dimsize; k++ ) {
    						indices1.add((Expression)index_vars.get(k).clone());
    					}
    					tSpecs = new ArrayList<Specifier>(1);
    					tSpecs.add(Specifier.DOUBLE);
    					if( refHostVar instanceof AccessExpression ) {
    						AccessExpression accExp = (AccessExpression)refHostVar;
    						Expression lhs = accExp.getLHS().clone();
    						Expression rhs = new ArrayAccess(accExp.getRHS().clone(), indices1);
    						accExp = new AccessExpression(lhs, (AccessOperator)accExp.getOperator(), rhs);
    						tcast = new Typecast(tSpecs, accExp);
    					} else {
    						aAccess = new ArrayAccess(refHostVar.clone(), indices1);
    						tcast = new Typecast(tSpecs, aAccess);
    					}
    					tStmt = new ExpressionStatement(new AssignmentExpression(ref__var.clone(), AssignmentOperator.NORMAL,
    							tcast));
    					loop_body.addStatement(tStmt);
    				} else {
    					// ref__var = (double)ftoutVar[temp_i*SIZE2+temp_k];
    					indexEx = null;
    					for( int k=0; k<dimsize; k++ ) {
    						Expression tExp = null;
    						if( k+1 < dimsize ) {
    							tExp = lengthList.get(k+1).clone();
    							for( int m=k+2; m<dimsize; m++ ) {
    								tExp = new BinaryExpression(tExp, BinaryOperator.MULTIPLY, lengthList.get(m).clone());
    							} 
    							tExp = new BinaryExpression(index_vars.get(k).clone(), BinaryOperator.MULTIPLY, tExp); 
    						} else {
    							tExp = index_vars.get(k).clone();
    						}
    						if( indexEx == null ) {
    							indexEx = tExp;
    						} else {
    							indexEx = new BinaryExpression(indexEx, BinaryOperator.ADD, tExp);
    						}
    					}
    					aAccess = new ArrayAccess(refHostVar.clone(), indexEx);
    					tSpecs = new ArrayList<Specifier>(1);
    					tSpecs.add(Specifier.DOUBLE);
    					tcast = new Typecast(tSpecs, aAccess);
    					tStmt = new ExpressionStatement(new AssignmentExpression(ref__var.clone(), AssignmentOperator.NORMAL,
    							tcast));
    					loop_body.addStatement(tStmt);
    				}
    				// diff__var = fabs(comp__var - ref__var);
    				fCall = new FunctionCall(new NameID("fabs"));
    				fCall.addArgument(new BinaryExpression(comp__var.clone(), BinaryOperator.SUBTRACT, ref__var.clone()));
    				tStmt = new ExpressionStatement(new AssignmentExpression(diff__var.clone(), AssignmentOperator.NORMAL, 
    						fCall));
    				loop_body.addStatement(tStmt);
    	            // if( ((ref__var != 0.0) && ((diff__var/fabs(ref__var)) > EPSILON)) || ((ref__var == 0.0) && (fabs(diff__var) > EPSILON)) ) { 
    				//     printf("[Kernel Verification Error] Values of variable a[%d][%d] differ:\nCPU output: %lf\nGPU output: %lf\n
    				//     OpenACC Annotation: ...\nEnclosing Procedure: cProc\n", temp_i, temp_k, ref__var, comp__var); 
    				//     foundDiff = 1;
    				//     break;
    				// } 
    				Expression condExp1 = new BinaryExpression(new BinaryExpression(ref__var.clone(), BinaryOperator.COMPARE_NE, ZERO.clone()),
    						BinaryOperator.LOGICAL_AND,
    						new BinaryExpression(new BinaryExpression(diff__var.clone(),BinaryOperator.DIVIDE,
    								new FunctionCall(new NameID("fabs"),ref__var.clone())), 
    								BinaryOperator.COMPARE_GT, EPSILON.clone()));
    				//Expression condExp2 = new BinaryExpression(diff__var.clone(), BinaryOperator.COMPARE_GT, EPSILON.clone());
    				Expression condExp2 = new BinaryExpression(new BinaryExpression(ref__var.clone(), BinaryOperator.COMPARE_EQ, ZERO.clone()),
    						BinaryOperator.LOGICAL_AND,
    						new BinaryExpression(new FunctionCall(new NameID("fabs"), diff__var.clone()), BinaryOperator.COMPARE_GT, EPSILON.clone()));
    				CompoundStatement ifBody = new CompoundStatement();
    				fCall = new FunctionCall(new NameID("printf"));
    				StringBuilder stmp = new StringBuilder(hostVar.toString());
    				for( int k=0; k<dimsize; k++ ) {
    					stmp.append("[%d]");
    				}
    				StringLiteral str =  null;
    				if( refname == null ) {
    					str = new StringLiteral("[Kernel Verification Error] Values of variable " + 
    							stmp.toString() + " differ:\\nCPU output: %E\\nGPU output: %E\\n" +
    							"OpenACC Annotation: " + cAnnot + "\\nEnclosing Procedure: " + cProc.getSymbolName() + "\\n" + 
    							"(Acceptable margin of errors = " + EPSILON + ")\\n");
    					if( minCheckValue != null ) {
    						str = new StringLiteral(str.getValue() + "(Values smaller than " + minCheckValue + " are ignored.)\\n");
    					}
    				} else {
    					str = new StringLiteral("[Kernel Verification Error] Values of variable " + 
    							stmp.toString() + " differ:\\nCPU output: %E\\nGPU output: %E\\n" +
    							"GPU kernel name: " + refname + "\\n" +
    							"(Acceptable margin of errors = " + EPSILON + ")\\n");
    					if( minCheckValue != null ) {
    						str = new StringLiteral(str.getValue() + "(Values smaller than " + minCheckValue + " are ignored.)\\n");
    					}
    				}
    				fCall.addArgument(str);
    				for( int k=0; k<dimsize; k++ ) {
    					fCall.addArgument(index_vars.get(k).clone());
    				}
    				fCall.addArgument(ref__var.clone());
    				fCall.addArgument(comp__var.clone());
    				ifBody.addStatement(new ExpressionStatement(fCall));
    				if( totalNumErrors == null ) {
    					ifBody.addStatement(new ExpressionStatement(
    							new AssignmentExpression(foundDiff.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(1))));
    					ifBody.addStatement(new BreakStatement());
    				} else {
    					ifBody.addStatement(new ExpressionStatement(
    							new AssignmentExpression(foundDiff.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(1))));
    					Statement expStmt = new ExpressionStatement(new AssignmentExpression(totalNumErrors.clone(), AssignmentOperator.ADD,
    							new IntegerLiteral(1)));
    					ifBody.addStatement(expStmt);
    				}
    				tStmt = new IfStatement(new BinaryExpression(condExp1, BinaryOperator.LOGICAL_OR, condExp2), ifBody);
    				//loop_body.addStatement(tStmt);
    				//If minCheckExp is set, error-checking will be done only if both CPU and GPU values are greater than or 
    				//equal to the minimum value.
    				if( minCheckValue == null ) {
    					loop_body.addStatement(tStmt);
    				} else {
    					Expression minCheckExp = new BinaryExpression( 
    							new BinaryExpression(ref__var.clone(), BinaryOperator.COMPARE_GE, minCheckValue.clone()), 
    							BinaryOperator.LOGICAL_AND,
    							new BinaryExpression(comp__var.clone(), BinaryOperator.COMPARE_GE, minCheckValue.clone())); 
    					loop_body.addStatement(new IfStatement(minCheckExp, tStmt));
    				}
    			} else {
    				if( totalNumErrors == null ) {
    					// if( foundDiff == 1 ) { break; }
    					Expression condExp1 = new BinaryExpression(foundDiff.clone(), BinaryOperator.COMPARE_EQ, new IntegerLiteral(1));
    					Statement tStmt = new IfStatement(condExp1, new BreakStatement());
    					loop_body.addStatement(tStmt);
    				}
    				loop_body.addStatement(innerLoop);
    			}
    			innerLoop = new ForLoop(loop_init, condition, step, loop_body);
    		}
    		Stmts.add(innerLoop);
    		// if( foundDiff == 1 ) {
    		//     printf("Exit the kernel verification test\n");
    		//     acc_shutdown(acc_device_nvidia);
    		//     exit(1);
    		// }
    		if( totalNumErrors == null ) {
    			Expression condExp1 = new BinaryExpression(foundDiff.clone(), BinaryOperator.COMPARE_EQ, new IntegerLiteral(1));
    			CompoundStatement ifBody = new CompoundStatement();
    			fCall = new FunctionCall(new NameID("printf"));
    			StringLiteral str = new StringLiteral("Exit the kernel verification test!\\n");
    			fCall.addArgument(str);
    			ifBody.addStatement(new ExpressionStatement(fCall));
    			fCall = new FunctionCall(new NameID("acc_shutdown"));
    			fCall.addArgument(new NameID("acc_device_nvidia"));
    			ifBody.addStatement(new ExpressionStatement(fCall));
    			fCall = new FunctionCall(new NameID("exit"));
    			fCall.addArgument(new IntegerLiteral(1));
    			ifBody.addStatement(new ExpressionStatement(fCall));
    			Stmts.add(new IfStatement(condExp1, ifBody));
    		}
    	}
    	if( refStmt != null ) {
    		CompoundStatement cStmt = (CompoundStatement)refStmt.getParent();
    		int listSize = Stmts.size();
    		for( int m = listSize-1; m>=0; m-- ) {
    			cStmt.addStatementAfter(refStmt, Stmts.get(m));
    		}
    	}
    	return Stmts;
    }
    
	/**
	 * Add a statement before the ref_stmt in the parent CompoundStatement.
	 * This method can be used to insert declaration statement before the ref_stmt,
	 * which is not allowed in CompoundStatement.addStatementBefore() method.
	 * 
	 * @param parent parent CompoundStatement containing the ref_stmt as a child
	 * @param ref_stmt reference statement
	 * @param new_stmt new statement to be added
	 */
	public static void addStatementBefore(CompoundStatement parent, Statement ref_stmt, Statement new_stmt) {
		List<Traversable> children = parent.getChildren();
		int index = Tools.indexByReference(children, ref_stmt);
		if (index == -1)
			throw new IllegalArgumentException();
		if (new_stmt.getParent() != null)
			throw new NotAnOrphanException();
		children.add(index, new_stmt);
		new_stmt.setParent(parent);
		if( new_stmt instanceof DeclarationStatement ) {
			Declaration decl = ((DeclarationStatement)new_stmt).getDeclaration();
			SymbolTools.addSymbols(parent, decl);
		}
	}

	/**
	 * Add a statement after the ref_stmt in the parent CompoundStatement.
	 * This method can be used to insert declaration statement after the ref_stmt,
	 * which is not allowed in CompoundStatement.addStatementAfter() method.
	 * 
	 * @param parent parent CompoundStatement containing the ref_stmt as a child
	 * @param ref_stmt reference statement
	 * @param new_stmt new statement to be added
	 */
	public static void addStatementAfter(CompoundStatement parent, Statement ref_stmt, Statement new_stmt) {
		List<Traversable> children = parent.getChildren();
		int index = Tools.indexByReference(children, ref_stmt);
		if (index == -1)
			throw new IllegalArgumentException();
		if (new_stmt.getParent() != null)
			throw new NotAnOrphanException();
		children.add(index+1, new_stmt);
		new_stmt.setParent(parent);
		if( new_stmt instanceof DeclarationStatement ) {
			Declaration decl = ((DeclarationStatement)new_stmt).getDeclaration();
			SymbolTools.addSymbols(parent, decl);
		}
	}
	
	/**
	 * Remove a child from a parent; this method is used to delete ProcedureDeclaration
	 * when both Procedure and ProcedureDeclaration need to be deleted. TranslationUnit
	 * symbol table contains only one entry for both, and thus TranslationUnit.removeChild()
	 * complains an error when trying to delete both of them. 
	 * 
	 * 
	 * @param parent parent traversable containing the child
	 * @param child child traversable to be removed
	 */
	public static void removeChild(Traversable parent, Traversable child)
	{
		List<Traversable> children = parent.getChildren();
		int index = Tools.indexByReference(children, child);

		if (index == -1)
			throw new NotAChildException();

		child.setParent(null);
		children.remove(index);
	}
	
	/**
	 * In C, both 0 and (void *)0 are legel for NULL pointer, but in C++, only 0 is legal.
	 * Therefore, in C++, "int *pt = (void *)0;" is illegal since C++ does not allow
	 * implicit type conversion. To fix this, "(void *)0" should be changed to "0" if
	 * "(void *)0" was a value of NULL macro.
	 * @param t
	 */
	public static void NULLPointerCorrection(Traversable t) {
		DFIterator<Typecast> iter =
			new DFIterator<Typecast>(t, Typecast.class);
		while (iter.hasNext()) {
			Typecast tc = iter.next();
			if( tc != null ) {
				//[FIXME] current Cetus parser parses pointers in a typecast expression (e.g., (void *)0) 
				//incorrectly as VariableDeclarator, and thus generic List should be used insted of List<Specifier>.
				//====> the above bug is fixed, but still it is possible to have Declarator in a spec list.
				//List<Specifier> specs = tc.getSpecifiers();
				List specs = tc.getSpecifiers();
				Expression tExp = tc.getExpression();
				boolean isNULLPointer = false;
				//[FIXME] current Cetus parser parses pointers in a typecast expression (e.g., (void *)0) 
				//incorrectly as VariableDeclarator.
				//==> Fixed.
				isNULLPointer = tExp.equals(new IntegerLiteral(0)) && (specs.size() == 2) && 
			specs.contains(Specifier.VOID) && specs.contains(PointerSpecifier.UNQUALIFIED);
/*				isNULLPointer = tExp.equals(new IntegerLiteral(0)) && (specs.size() == 2) && 
				specs.contains(Specifier.VOID) && specs.get(1).toString().contains("*");*/
				//Debugging print.
/*				if( specs.size() > 1 ) {
					System.out.println("Typecast expression: " + tc);
					Object tS = specs.get(0);
					System.out.println("class type of the first specifier " + tS + " : " + tS.getClass());
					tS = specs.get(1);
					System.out.println("class type of the second specifier " + tS + " : " + tS.getClass());
				}*/
				if( isNULLPointer ) {
					Traversable pt = tc.getParent();
					boolean swapTypecast = false;
					if( (pt instanceof BinaryExpression) ) {
						BinaryExpression bExp = (BinaryExpression)pt;
						BinaryOperator bOp = bExp.getOperator();
						if( bOp.equals(BinaryOperator.COMPARE_EQ) || bOp.equals(BinaryOperator.COMPARE_NE) ||
								bOp.equals(AssignmentOperator.NORMAL) ) {
							Expression tExp2 = bExp.getLHS();
							if( tc.equals(tExp2) ) {
								tExp2 = bExp.getRHS();
							}
							while ( (tExp2 instanceof AccessExpression) || (tExp2 instanceof ArrayAccess) ) {
								if( tExp2 instanceof ArrayAccess ) {
									tExp2 = ((ArrayAccess)tExp2).getArrayName();
								}
								if ( tExp2 instanceof AccessExpression ) {
									tExp2 = ((AccessExpression)tExp2).getRHS();
								}
							}
							DFIterator<Identifier> iter2 = new DFIterator<Identifier>(tExp2, Identifier.class);
							int cnt = 0;
							Identifier tid = null;
							while( iter2.hasNext() ) {
								tid = iter2.next();
								if(++cnt > 1) {
									break;
								}
							}
							if( (cnt == 1) && (tid != null) ) {
								Symbol sym = tid.getSymbol();
								if( !sym.getTypeSpecifiers().contains(Specifier.VOID) ) {
									swapTypecast = true;
								}
							}
						}
					} else if( pt instanceof ConditionalExpression ) {
						ConditionalExpression cE = (ConditionalExpression)pt;
						if( tc.equals(cE.getTrueExpression()) ) {
							swapTypecast = true;
						} else if( tc.equals(cE.getFalseExpression()) ) {
							swapTypecast = true;
						}
					} else if( pt instanceof Initializer ) {
						pt = pt.getParent();
						List tspecs = null;
						if( pt instanceof VariableDeclarator ) {
							tspecs = ((VariableDeclarator)pt).getTypeSpecifiers();
							if( !tspecs.contains(Specifier.VOID) ) {
								swapTypecast = true;
							}
						} else if( pt instanceof NestedDeclarator ) {
							tspecs = ((NestedDeclarator)pt).getTypeSpecifiers();
							if( !tspecs.contains(Specifier.VOID) ) {
								swapTypecast = true;
							}
						} else if( pt instanceof ProcedureDeclarator ) {
							swapTypecast = true;
						}
					} else if( pt instanceof FunctionCall ) {
						swapTypecast = true;
					}
					
					if( swapTypecast ) {
						//This Typecast (tc) is the value of NULL macro.
						//System.out.println("found NULL pointer expression: ");
						tc.swapWith(new IntegerLiteral(0));

					}
				}
			}
		}
	}
	
	//Add initial value to the return variable for the main function if cetus.Transforms.NormalizeReturn()
	//is executed.
	public static void initializeMainReturnVariable(Program prog) {
		List<Procedure> procList = IRTools.getProcedureList(prog);
		for( Procedure proc : procList ) {
			String pName = proc.getSymbolName();
			if( pName.equals("main") || pName.equals("MAIN__") ) {
				CompoundStatement cBody = proc.getBody();
				Declaration retSymDecl = SymbolTools.findSymbol(cBody, "_ret_val_0");
				if( retSymDecl != null ) {
					for( Traversable declr : retSymDecl.getChildren() ) {
						if( declr instanceof VariableDeclarator ) {
							VariableDeclarator vdeclr = (VariableDeclarator)declr;
							if( vdeclr.getSymbolName().equals("_ret_val_0") ) {
								vdeclr.setInitializer(new Initializer(new IntegerLiteral(0)));
								break;
							}
						}
					}
				}
				break;
			}
		}
	}
	
	/**
	 * Add a new extern procedure declaration for the input Procedure, new_proc, if its declaration
	 * does not exist in the input translation unit, trUnt.
	 * 
	 * @param new_proc
	 * @param trUnt
	 * @return
	 */
	public static void addExternProcedureDeclaration(Procedure c_proc, TranslationUnit cTu) {
		VariableDeclaration newProcDecl = null;
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
			Traversable parent = procDeclr.getParent();
			VariableDeclaration procDecl = null;
			if( parent instanceof VariableDeclaration ) {
				//Found function declaration.
				procDecl = (VariableDeclaration)parent;
				if( procDeclr.getID().equals(c_proc.getName()) ) {
					//Found function declaration for the input procedure, c_proc.
					newProcDecl = procDecl;
					break;
				}
			}
		}
		if( newProcDecl == null ) {
			//Procedure declaration for c_proc does not exist.
			//Create a new function declaration.
			List<Specifier> returnTypes = new ArrayList<Specifier>(2);
			returnTypes.add(Specifier.EXTERN);
			returnTypes.addAll(c_proc.getReturnType());
			newProcDecl = 
				new VariableDeclaration(returnTypes, c_proc.getDeclarator().clone());
			//Insert the new function declaration.
			Declaration firstDecl = cTu.getFirstDeclaration();
			if( firstDecl != null ) {
				//System.out.println("firstDecl found: " + firstDecl);
				cTu.addDeclarationBefore(firstDecl, newProcDecl);
			} else {
				cTu.addDeclaration(newProcDecl);
			}
		}
	}
	
	public static VariableDeclaration addExternVariableDeclaration(VariableDeclaration vDecl, TranslationUnit cTu) {
		Declarator inSym = vDecl.getDeclarator(0);
		VariableDeclaration decls = null;
		List specs = null;
		List orgArraySpecs = null;
		List arraySpecs = new LinkedList();
		if( inSym instanceof VariableDeclarator ) {
			VariableDeclarator varSym = (VariableDeclarator)inSym;
			NameID id = new NameID(varSym.getSymbolName());
			specs = varSym.getTypeSpecifiers();
			// Separate declarator/declaration specifiers.
			List declaration_specs = new ArrayList(specs.size());
			if( !specs.contains(Specifier.EXTERN) ) {
				declaration_specs.add(Specifier.EXTERN);
			}
			List declarator_specs = new ArrayList(specs.size());
			for (int i = 0; i < specs.size(); i++) {
				Object spec = specs.get(i);
				if (spec instanceof PointerSpecifier) {
					declarator_specs.add(spec);
				} else {
					declaration_specs.add(spec);
				}
			}
			orgArraySpecs = varSym.getArraySpecifiers();
			if(  (orgArraySpecs != null) && (!orgArraySpecs.isEmpty()) ) {
				for( Object obj : orgArraySpecs ) {
					if( obj instanceof ArraySpecifier ) {
						ArraySpecifier oldAS = (ArraySpecifier)obj;
						int numDims = oldAS.getNumDimensions();
						List<Expression> dimensions = new LinkedList<Expression>();
						for( int i=0; i<numDims; i++ ) {
							dimensions.add(oldAS.getDimension(i).clone());
						}
						ArraySpecifier newAS = new ArraySpecifier(dimensions);
						arraySpecs.add(newAS);
					} else {
						arraySpecs.add(obj);
					}
				}
			}
			VariableDeclarator Declr = null;
			if( declarator_specs.isEmpty() ) {
				if( arraySpecs == null || arraySpecs.isEmpty() ) {
					Declr = new VariableDeclarator(id);
				} else {
					Declr = new VariableDeclarator(id, arraySpecs);
				}
			} else if ( arraySpecs == null || arraySpecs.isEmpty() ) {
				Declr = new VariableDeclarator(declarator_specs, id);
			} else {
				Declr = new VariableDeclarator(declarator_specs, id, arraySpecs);
			}

			decls = new VariableDeclaration(declaration_specs, Declr);
			Declaration firstDecl = cTu.getFirstDeclaration();
			if( firstDecl != null ) {
				//System.out.println("firstDecl found: " + firstDecl);
				cTu.addDeclarationBefore(firstDecl, decls);
			} else {
				cTu.addDeclaration(decls);
			}
		} else if( inSym instanceof NestedDeclarator ) {
			NestedDeclarator nestedSym = (NestedDeclarator)inSym;
			NameID id = new NameID(nestedSym.getSymbolName());
			Declarator childDeclarator = nestedSym.getDeclarator();
			List childDeclr_specs = null;
			List childArray_specs = null;
			if( childDeclarator instanceof VariableDeclarator ) {
				VariableDeclarator varSym = (VariableDeclarator)childDeclarator;
				childDeclr_specs = varSym.getSpecifiers();
				childArray_specs = varSym.getArraySpecifiers();
			} else {
				Tools.exit("[ERROR in TransformTools.declareClonedVariable()] nested declarator whose child declarator is also " +
						"nested declarator is not supported yet; exit!\n" +
						"Symbol: " + inSym + "\n");
			}
			VariableDeclarator childDeclr = null;
			if (childDeclr_specs.isEmpty()) {
				if (childArray_specs == null || childArray_specs.isEmpty()) {
					childDeclr = new VariableDeclarator(id);
				} else {
					childDeclr = new VariableDeclarator(id, childArray_specs);
				}
			} else if (childArray_specs == null || childArray_specs.isEmpty()) {
				childDeclr = new VariableDeclarator(childDeclr_specs, id);
			} else {
				childDeclr = new VariableDeclarator(childDeclr_specs, id, childArray_specs);
			}
			specs = nestedSym.getTypeSpecifiers();
			// Separate declarator/declaration specifiers.
			List declaration_specs = new ArrayList(specs.size());
			if( !specs.contains(Specifier.EXTERN) ) {
				declaration_specs.add(Specifier.EXTERN);
			}
			List declarator_specs = new ArrayList(specs.size());
			for (int i = 0; i < specs.size(); i++) {
				Object spec = specs.get(i);
				if (spec instanceof PointerSpecifier) {
					declarator_specs.add(spec);
				} else {
					declaration_specs.add(spec);
				}
			}
			arraySpecs = nestedSym.getArraySpecifiers();
			NestedDeclarator nestedDeclr = null;
			if( declarator_specs.isEmpty() ) {
				if( arraySpecs == null || arraySpecs.isEmpty() ) {
					nestedDeclr = new NestedDeclarator(childDeclr);
				} else {
					nestedDeclr = new NestedDeclarator(declarator_specs, childDeclr, null, arraySpecs);
				}
			} else if ( arraySpecs == null || arraySpecs.isEmpty() ) {
				nestedDeclr = new NestedDeclarator(declarator_specs, childDeclr, null);
			} else {
				nestedDeclr = new NestedDeclarator(declarator_specs, childDeclr, null, arraySpecs);
			}

			decls = new VariableDeclaration(declaration_specs, nestedDeclr);
			Declaration firstDecl = cTu.getFirstDeclaration();
			if( firstDecl != null ) {
				//System.out.println("firstDecl found: " + firstDecl);
				cTu.addDeclarationBefore(firstDecl, decls);
			} else {
				cTu.addDeclaration(decls);
			}
		}
		return decls;
	}

}
