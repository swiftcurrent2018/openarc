package openacc.analysis;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.TreeMap;
import java.util.LinkedList;

import openacc.hir.CUDASpecifier;
import openacc.hir.CudaStdLibrary;

import cetus.exec.Driver;
import cetus.hir.*;
import openacc.hir.*;
import cetus.analysis.LoopTools;


/**
 * <b>AnalysisTools</b> provides tools that perform various analyses for OpenACC-to-GPU translation.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public abstract class AnalysisTools {
	/**
	 * Java doesn't allow a class to be both abstract and final,
	 * so this private constructor prevents any derivations.
	 */
	private AnalysisTools()
	{
	}

	static private final String[] predefinedCVars = {"stdin", "stdout",
	"stderr"};

	static List predefinedCVarList = new LinkedList(Arrays.asList(predefinedCVars));

	/**
	 * check whether the input variable is a member of a class.
	 * 
	 * @param varSym input variable symbol
	 * @return true if input variable is a member of a class
	 */
	public static boolean isClassMember(Declarator varSym) {
		Traversable t = varSym.getParent();
		boolean foundParentClass = false;
		while( t != null ) {
			if( t instanceof ClassDeclaration ) {
				foundParentClass = true;
				break;
			} else {
				t = t.getParent();
			}
		}
		return foundParentClass;
	}
	
	/**
	 * check whether the input variable is a typedef name or not.
	 * 
	 * @param varSym input variable symbol
	 * @return true if input variable is a typedef name.
	 */
	public static boolean isTypedefName(Declarator varSym) {
		Traversable t = varSym.getParent();
		boolean foundTypedef = false;
		if( varSym instanceof VariableDeclarator ) {
			if( ((VariableDeclarator)varSym).getTypeSpecifiers().contains(Specifier.TYPEDEF) ) {
				foundTypedef = true;
			}
		} else if( varSym instanceof NestedDeclarator ) {
			if( ((NestedDeclarator)varSym).getTypeSpecifiers().contains(Specifier.TYPEDEF) ) {
				foundTypedef = true;
			}
		}
		return foundTypedef;
	}
	
	/**
	 * Return a set of symbols that are initialized in their declaration statements within the traversable tt.
	 * 
	 * @param tt
	 * @return
	 */
	static public Set<Symbol> getInitializedSymbols( Traversable tt) {
		Set<Symbol> ret = new HashSet<Symbol>();
        Set<Symbol> tSymSet = null;
        DFIterator<SymbolTable> iter =
                new DFIterator<SymbolTable>(tt, SymbolTable.class);
        while (iter.hasNext()) {
            SymbolTable o = iter.next();
            tSymSet = SymbolTools.getVariableSymbols(o);
            for( Symbol tSym : tSymSet ) {
            	if( tSym instanceof VariableDeclarator ) {
            		VariableDeclarator tDeclr = (VariableDeclarator)tSym;
            		Initializer tInit = tDeclr.getInitializer();
            		if( (tInit != null) && (!tInit.getChildren().isEmpty()) ) {
            			ret.add(tSym);
            		}
            	}
            }
        }
		return ret;
	}

	/**
	 * Return a set of static symbols from the input symbol set, iset
	 * 
	 * @param iset input symbol set
	 * @return a set of static symbols
	 */
	static public HashSet<Symbol> getStaticVariables(Set<Symbol> iset)
	{
		HashSet<Symbol> ret = new HashSet<Symbol> ();
		for (Symbol sym : iset)
		{
			if(SymbolTools.containsSpecifier(sym, Specifier.STATIC)) {
				ret.add(sym);
			}
		}
		return ret;
	}
	
	/**
	 * Returns the set of global symbols accessed in the input Traversable object, st.
	 *
	 * @param st input Traversable object to be searched.
	 * @return the set of accessed global symbols, visible in the current scope.
	 */
	public static Set<Symbol> getAccessedGlobalSymbols(Traversable st, Map<String, Symbol> visibleGSymMap) {
		Map<String, Symbol> gSymMap;
		Set<Symbol> rSet = new HashSet<Symbol>();
		Set<Symbol> aSet = AnalysisTools.getAccessedVariables(st, true);
		Set<Symbol> lSet = SymbolTools.getLocalSymbols(st);
		// Remove procedure symbols from aSet. //
		Set<Symbol> tSet = new HashSet<Symbol>();
		for( Symbol sm : aSet ) {
			if( (sm instanceof Procedure) || (sm instanceof ProcedureDeclarator) ) {
				tSet.add(sm);
			}
		}
		aSet.removeAll(tSet);
		// Remove local symbols from aSet.
		aSet.removeAll(lSet);
		// Find global symbols visible in the current scope
		// (distinguish extern variable from the original one.)
		if( visibleGSymMap == null ) {
			tSet = SymbolTools.getGlobalSymbols(st);
			gSymMap = new HashMap<String, Symbol>();
			for( Symbol gS : tSet ) {
				gSymMap.put(gS.getSymbolName(), gS);
			}
		} else {
			gSymMap = visibleGSymMap;
		}
		for( Symbol sym : aSet ) {
			if( SymbolTools.isGlobal(sym) ) {
				if( gSymMap.containsKey(sym.getSymbolName()) ) {
					rSet.add(gSymMap.get(sym.getSymbolName()));
				} else {
					Procedure tProc = IRTools.getParentProcedure(st);
					String tProcName = "Unknown";
					if( tProc != null ) {
						tProcName = tProc.getSymbolName();
					}
					PrintTools.println("\n[WARNING] a global symbol (" + sym.getSymbolName() + ") accessed in a procedure, " +
							tProcName + ", is not visible in the enclosing compute region; this may result in incorrect translation.\n", 0);
				}
			}
		}
		return rSet;
	}
	
	/**
	 * Returns the set of global symbols accessed in the input Traversable object, st.
	 * If st contains function calls, each function is recursively checked.
	 *
	 * @param st input Traversable object to be searched interprocedurally.
	 * @return the set of accessed global symbols, visible in the current scope.
	 */
	public static Set<Symbol> getIpAccessedGlobalSymbols(Traversable st, Map<String, Symbol> visibleGSymMap, Set<Procedure> accessedProcs) {
		if( accessedProcs == null ) {
			accessedProcs = new HashSet<Procedure>();
		}
		Map<String, Symbol> gSymMap;
		Set<Symbol> rSet = new HashSet<Symbol>();
		Set<Symbol> aSet = AnalysisTools.getAccessedVariables(st, true);
		Set<Symbol> lSet = SymbolTools.getLocalSymbols(st);
		// Remove procedure symbols from aSet. //
		Set<Symbol> tSet = new HashSet<Symbol>();
		for( Symbol sm : aSet ) {
			if( (sm instanceof Procedure) || (sm instanceof ProcedureDeclarator) ) {
				tSet.add(sm);
			}
		}
		aSet.removeAll(tSet);
		// Remove local symbols from aSet.
		aSet.removeAll(lSet);
		// Find global symbols visible in the current scope
		// (distinguish extern variable from the original one.)
		if( visibleGSymMap == null ) {
			tSet = SymbolTools.getGlobalSymbols(st);
			gSymMap = new HashMap<String, Symbol>();
			for( Symbol gS : tSet ) {
				gSymMap.put(gS.getSymbolName(), gS);
			}
		} else {
			gSymMap = visibleGSymMap;
		}
		for( Symbol sym : aSet ) {
			if( SymbolTools.isGlobal(sym) ) {
				if( gSymMap.containsKey(sym.getSymbolName()) ) {
					rSet.add(gSymMap.get(sym.getSymbolName()));
				} else {
					//[DEBUG] even if this global symbol is not visible, add this so that it can be externed later
					//to be accesed by the enclosing compute region.
					rSet.add(sym);
					Procedure tProc = IRTools.getParentProcedure(st);
					String tProcName = "Unknown";
					if( tProc != null ) {
						tProcName = tProc.getSymbolName();
					}
					PrintTools.println("\n[WARNING] a global symbol (" + sym.getSymbolName() + ") accessed in a procedure, " +
							tProcName + ", is not visible in the enclosing compute region; this may result in incorrect translation.\n", 0);
				}
			}
		}
		List<FunctionCall> calledFuncs = IRTools.getFunctionCalls(st);
		for( FunctionCall call : calledFuncs ) {
			Procedure called_procedure = call.getProcedure();
			if( (called_procedure != null) && !accessedProcs.contains(called_procedure) ) {
				accessedProcs.add(called_procedure);
				CompoundStatement body = called_procedure.getBody();
				Set<Symbol> procAccessedSymbols = getIpAccessedGlobalSymbols(body, gSymMap, accessedProcs);
				rSet.addAll(procAccessedSymbols);
			}
		}
		return rSet;
	}


	/**
	 * returns a set of all accessed symbols except Procedure symbols and 
	 * member symbols of an enumeration. if {@code IRSymbolOnly} is true, only IR symbols are 
	 * included.
	 */
	public static Set<Symbol> getAccessedVariables(Traversable st, boolean IRSymbolOnly)
	{
		Set<Symbol> set = SymbolTools.getAccessedSymbols(st);
		HashSet<Symbol> ret = new HashSet<Symbol> ();
		for (Symbol symbol : set)
		{
			if( symbol.getDeclaration() instanceof Enumeration ) {
				continue; //Skip member symbols of an enumeration.
			}
			if( symbol instanceof VariableDeclarator ) {
				if( IRSymbolOnly ) {
					if( !isClassMember((VariableDeclarator)symbol) ) {
						ret.add(symbol);
					}
				} else {
					ret.add(symbol);
				}
			} else if( symbol instanceof AccessSymbol ) { 
				//DEBUG: this case will not be executed since set does not include this case.
				if( IRSymbolOnly ) {
					Symbol base = ((AccessSymbol)symbol).getIRSymbol();
					if( base != null ) {
						ret.add(base);
					}
				} else {
					ret.add(symbol);
				}
			} else if( symbol instanceof NestedDeclarator ){
				//Add it if it is a pointer to a chunk of arrays (ex: int (*B)[SIZE];)
				if( !((NestedDeclarator)symbol).isProcedure() ) {
					ret.add(symbol);
				}
			} else if( symbol instanceof DerefSymbol ) {
				//FIXME: how to handle this?
				PrintTools.println("\n[WARNING] AnalysisTools.getAccessedVariables() will ignore the following symbol, " +
						symbol + ".\n", 0);
			}
		}
		return ret;
	}
	
	/**
	 * For each symbol in input set, iset,
	 *     if it is an AccessSymbol, base symbol is added to output set
	 *     else if it is not a class member, the symbol itself is added to the output set.
	 * 
	 * @param iset
	 * @return output set of base symbols
	 */
	static public HashSet<Symbol> getIRSymbols(Set<Symbol> iset) {
		HashSet<Symbol> ret = new HashSet<Symbol>();
		for( Symbol sym : iset ) {
			if( sym instanceof VariableDeclarator ) {
				if( !isClassMember((VariableDeclarator)sym) ) {
					ret.add(sym);
				}
			} else if( sym instanceof AccessSymbol ) {
				Symbol base = ((AccessSymbol)sym).getIRSymbol();
				if( base != null ) {
					ret.add(base);
				}
			} else if( sym != null ){
				ret.add(sym);
			}
		}
		return ret;
	}
	
    /**
     * Get loop index variables.
     * This method differs from LoopTools.getIndexVariables in that this method can
     * handle some non-canonical for-loops as well as canonical loops.
     */
    public static List<Expression> getIndexVariables(Loop loop) {
        List<Expression> indexVarList = new ArrayList<Expression>(2);
        Expression indexVar = null;
        // Handle for loops here
        if (loop instanceof ForLoop) {
            // determine the name of the index variable
            ForLoop for_loop = (ForLoop)loop;
            Expression step_expr = for_loop.getStep();
            if (step_expr instanceof AssignmentExpression) {
                indexVar = ((AssignmentExpression)step_expr).getLHS().clone();
                indexVarList.add(indexVar);
            } else if (step_expr instanceof UnaryExpression) {
                UnaryExpression uexpr = (UnaryExpression)step_expr;
                indexVar = uexpr.getExpression().clone();
                indexVarList.add(indexVar);
            } else if( step_expr instanceof CommaExpression ) {
            	Set<Traversable> step_expr_set = new HashSet<Traversable>();
            	step_expr_set.addAll((List<Traversable>)step_expr.getChildren());
            	for( Traversable tExp : step_expr_set ) {
            		if (tExp instanceof AssignmentExpression) {
            			indexVar = ((AssignmentExpression)tExp).getLHS().clone();
            			indexVarList.add(indexVar);
            		} else if (tExp instanceof UnaryExpression) {
            			UnaryExpression uexpr = (UnaryExpression)tExp;
            			indexVar = uexpr.getExpression().clone();
            			indexVarList.add(indexVar);
            		}	
            	}
            }
        }
        // Handle other loop types
        else {
        }
        return indexVarList;
    }
	
	/**
	 * collect loop index variables for worksharing loops (gang/worker/vector loops) and
	 * other loops nested in the worksharing loops.
	 * 
	 * OpenACC V2.0 chapter 2.6.1 says "The loop variable in a C for statement or Fortran do statement that is associated 
	 * with a loop directive is predetermined to be private to each thread that will execute each iteration of the loop."
	 * Therefore, all index variables related to the OpenACC loop directive should be private.
	 */
	public static HashSet<Symbol> getWorkSharingLoopIndexVarSet(Traversable tr)
	{
		HashSet<Symbol> ret = new HashSet<Symbol> ();
		DFIterator<Annotatable> iter = new DFIterator<Annotatable>(tr, Annotatable.class);
		while(iter.hasNext())
		{
			Annotatable at = iter.next();
/*			boolean workLoopFound = false;
			for( String clause : ACCAnnotation.worksharingClauses ) {
				if(at.containsAnnotation(ACCAnnotation.class, clause)) {
					workLoopFound = true;
					break;
				}
			}
			if(workLoopFound) {*/
			if(at.containsAnnotation(ACCAnnotation.class, "loop") || at.containsAnnotation(ACCAnnotation.class, "innergang")) {
				ForLoop loop = (ForLoop)at;
				List<Expression> ivar_exprSet = getIndexVariables(loop);
				for( Expression ivar_expr : ivar_exprSet ) {
					Symbol ivar = SymbolTools.getSymbolOf(ivar_expr);
					if (ivar==null)
						Tools.exit("[getLoopIndexVariables] Cannot find symbol:" + ivar_expr.toString());
					else
						ret.add(ivar);
				}
				ret.addAll(getLoopIndexVarSet(loop.getBody()));
			}
		}
		return ret;
	}
	
	/**
	 * collect loop index variables within the input traversable object {@code tr}
	 */
	public static	HashSet<Symbol> getLoopIndexVarSet(Traversable tr)
	{
		HashSet<Symbol> ret = new HashSet<Symbol> ();
		DFIterator<ForLoop> iter = new DFIterator<ForLoop>(tr, ForLoop.class);
		while(iter.hasNext())
		{
			List<Expression> ivar_exprSet = getIndexVariables(iter.next());
			for( Expression ivar_expr : ivar_exprSet ) {
				if( ivar_expr != null ) {
					Symbol ivar = SymbolTools.getSymbolOf(ivar_expr);
					if (ivar==null)
						Tools.exit("[getLoopIndexVariables] Cannot find symbol:" + ivar_expr.toString());
					else
						ret.add(ivar);
				}
			}
		}
		return ret;
	}

	/**
	 * Check whether a CUDA kernel function calls C standard library functions 
	 * that are not supported by CUDA runtime systems.
	 * If so, CUDA compiler will fail if they are not inlinable.
	 * 
	 * @param prog
	 */
	public static void checkKernelFunctions(Program prog, String targetArch) {
		List<Procedure> procList = IRTools.getProcedureList(prog);
		List<Procedure> kernelProcs = new LinkedList<Procedure>();
		for( Procedure proc : procList ) {
			List return_type = proc.getReturnType();
			if( return_type.contains(CUDASpecifier.CUDA_DEVICE) 
					|| return_type.contains(CUDASpecifier.CUDA_GLOBAL) ) {
				kernelProcs.add(proc);
			}
		}
		for( Procedure kProc : kernelProcs ) {
			List<FunctionCall> fCalls = IRTools.getFunctionCalls(kProc);
			for( FunctionCall fCall : fCalls ) {
				if( StandardLibrary.contains(fCall) ) {
					if( targetArch.equals("CUDA") ) {
						if( !CudaStdLibrary.contains(fCall) ) {
							PrintTools.println("\n[WARNING] C standard library function ("+fCall.getName()+
									") is called in a kernel function,"+kProc.getName()+
									", but not supported by CUDA runtime system V4.0; " +
									"it may cause compilation error if not inlinable.\n", 0);
						}
					} else if( targetArch.equals("OPENCL") ) {
						if( !OpenCLStdLibrary.contains(fCall) ) {
							PrintTools.println("\n[WARNING] C standard library function ("+fCall.getName()+
									") is called in a kernel function,"+kProc.getName()+
									", but not supported by OpenCL runtime system V1.1; " +
									"it may cause compilation error if not inlinable.\n", 0);
						}
					}
				}
			}
		}
	}
	
	/**
	 * Generate an access expression of a form, a.b, from an AccessSymbol.
	 * 
	 * @param accSym
	 * @return
	 */
	public static AccessExpression accessSymbolToExpression( AccessSymbol accSym, List<Expression> indices ) {
		Symbol base = accSym.getBaseSymbol();
		Symbol member = accSym.getMemberSymbol();
        Expression lhs = null, rhs = null;
        if (base instanceof DerefSymbol) {
            lhs = ((DerefSymbol)base).toExpression();
        } else if (base instanceof AccessSymbol) {
            lhs = accessSymbolToExpression((AccessSymbol)base, indices);
        } else if (base instanceof Identifier) {
            lhs = new Identifier(base);
        } else {
            PrintTools.printlnStatus(0,
                    "\n[WARNING] Unexpected access expression type\n");
            return null;
        }
        if( (indices == null) || indices.isEmpty() ) {
        	rhs = new Identifier(member);
        } else {
        	rhs = new ArrayAccess(new Identifier(member), indices);
        }
        return new AccessExpression(lhs, AccessOperator.MEMBER_ACCESS, rhs);
	}

	/**
	 * Converts a collection of symbols to a string of symbol names with the given separator.
	 *
	 * @param symbols the collection of Symbols to be converted.
	 * @param separator the separating string.
	 * @return the converted string, which is a list of symbol names
	 */
	public static String symbolsToString(Collection<Symbol> symbols, String separator)
	{
		if ( symbols == null || symbols.size() == 0 )
			return "";

		StringBuilder str = new StringBuilder(80);

		Iterator<Symbol> iter = symbols.iterator();
		if ( iter.hasNext() )
		{
			str.append(iter.next().getSymbolName());
			while ( iter.hasNext() ) {
				str.append(separator+iter.next().getSymbolName());
			}
		}

		return str.toString();
	}

	/**
	 * Converts a collection of symbols to a set of strings of symbol names.
	 *
	 * @param symbols the collection of Symbols to be converted.
	 * @return a set of strings, which contains symbol names
	 */
	public static Set<String> symbolsToStringSet(Collection<Symbol> symbols)
	{
		HashSet<String> strSet = new HashSet<String>();
		if ( symbols == null || symbols.size() == 0 )
			return strSet;


		Iterator<Symbol> iter = symbols.iterator();
		if ( iter.hasNext() )
		{
			strSet.add(iter.next().getSymbolName());
			while ( iter.hasNext() ) {
				strSet.add(iter.next().getSymbolName());
			}
		}

		return strSet;
	}
	
	/**
	 * Converts a collection of expressions to a set of strings.
	 *
	 * @param exprs the collection of expressions to be converted.
	 * @return a set of strings
	 */
	public static Set<String> expressionsToStringSet(Collection<Expression> exprs)
	{
		HashSet<String> strSet = new HashSet<String>();
		if ( exprs == null || exprs.size() == 0 )
			return strSet;


		Iterator<Expression> iter = exprs.iterator();
		if ( iter.hasNext() )
		{
			strSet.add(iter.next().toString());
			while ( iter.hasNext() ) {
				strSet.add(iter.next().toString());
			}
		}

		return strSet;
	}
	
	/**
	 * Convert subarray into a symbol.
	 * 
	 * @param subArr SubArray
	 * @param returnIRSymbols if true, returned symbol will be IR symbol.
	 * @return symbol for the {@code subArr}
	 */
	public static Symbol subarrayToSymbol(SubArray subArr, boolean returnIRSymbols) {
		Symbol sym = SymbolTools.getSymbolOf(subArr.getArrayName());
		if( sym == null ) {
			PrintTools.println("\n[WARNING in AnalysisTools.subarrayToSymbol()]: cannot find the symbol for the subarray, " +
					subArr  + ".\n", 0);
		} 
		if( returnIRSymbols ) {
			if( sym instanceof PseudoSymbol ) {
				sym = ((PseudoSymbol)sym).getIRSymbol();
			}
		}
		return sym;
	}
	
	/**
	 * Convert subarrays into a set of symbols.
	 * 
	 * @param subarrays collection of {@code subarrays}
	 * @param returnIRSymbols if true, returned symbols will be IR symbols.
	 * @return a set of symbols for the {@code subarrays}
	 */
	public static Set<Symbol> subarraysToSymbols(Collection<SubArray> subarrays, boolean returnIRSymbols) {
		Set<Symbol> symbols = new HashSet<Symbol>();
		for( SubArray subArr : subarrays ) {
			Symbol sym = subarrayToSymbol(subArr, returnIRSymbols);
			symbols.add(sym);
		}
		return symbols;
	}
	
	/**
	 * Find a subarray of the input symbol, {@code inSym} among the input subarrays, {@code subarrays}.
	 * 
	 * @param subarrays collection of {@code subarrays}
	 * @param inSym symbol to search for
	 * @return subarray of the input symbol, {@code inSym}
	 */
	public static SubArray subarrayOfSymbol(Collection<SubArray> subarrays, Symbol inSym) {
		SubArray sArry = null;
		for( SubArray subArr : subarrays ) {
			Symbol sym = SymbolTools.getSymbolOf(subArr.getArrayName());
			if( sym == null ) {
				PrintTools.println("\n[WARNING in AnalysisTools.subarraysOfSymbol()]: cannot find the symbol for the subarray, " +
						subArr  + ".\n", 0);
			} 
			if( sym instanceof PseudoSymbol ) {
				sym = ((PseudoSymbol)sym).getIRSymbol();
			}
			if( inSym.equals(sym) ) {
				sArry = subArr;
				break;
			}
		}
		return sArry;
	}
	
	
	/**
	 * Find a subarray for the input symbol <var>inSym</var> by searching 
	 * data clauses in a given ACCAnnotation <var>aAnnot</var>
	 * 
	 * @param aAnnot
	 * @param inSym
	 * @param IRSymbolOnly
	 * @return
	 */
	public static SubArray findSubArrayInDataClauses(ACCAnnotation aAnnot, Symbol inSym, boolean IRSymbolOnly) {
		SubArray rSubArray = null;
		if( (inSym != null) && (aAnnot != null) ) {
			for( String dClause : ACCAnnotation.dataClauses ) {
				Set<SubArray> dataSet = (Set<SubArray>)aAnnot.get(dClause);
				if( dataSet != null ) {
					for( SubArray tSub : dataSet ) {
						Symbol subSym = subarrayToSymbol(tSub, IRSymbolOnly);
						if( inSym.equals(subSym) ) {
							rSubArray = tSub;
							break;
						}
					}
				}
				if( rSubArray != null ) {
					break;
				}
			}
		}
		return rSubArray;
	}
	
	/**
	 * Find a subarray for the input symbol <var>inSym</var> by searching data clauses 
	 * in data regions enclosing the input traversable <var>at</var>
	 * If multiple subarrays exist, the first one is returned. 
	 * 
	 * @param inSym
	 * @param IRSymbolOnly
	 * @param at
	 * @param containsDimensionInfo returns only if the subarray contains dimension info.
	 * @return
	 */
	public static SubArray findSubArrayInEnclosingDataClauses(Symbol inSym, boolean IRSymbolOnly, 
			Traversable at, boolean containsDimensionInfo) {
		SubArray rSubArray = null;
		List<ACCAnnotation> aAnnotList = null;
		if( (inSym != null) && (at != null) ) {
			while (at != null) { 
				if( at instanceof Annotatable ) {
					aAnnotList = ((Annotatable)at).getAnnotations(ACCAnnotation.class);
					if( aAnnotList != null ) {
						for( ACCAnnotation aAnnot : aAnnotList ) {
							for( String dClause : ACCAnnotation.dataClauses ) {
								Set<SubArray> dataSet = (Set<SubArray>)aAnnot.get(dClause);
								if( dataSet != null ) {
									for( SubArray tSub : dataSet ) {
										Symbol subSym = subarrayToSymbol(tSub, IRSymbolOnly);
										if( inSym.equals(subSym) ) {
											if( containsDimensionInfo ) {
												if( tSub.getArrayDimension() >= 0 ) {
													rSubArray = tSub;
													break;
												}
											} else {
												rSubArray = tSub;
												break;
											}
										}
									}
								}
								if( rSubArray != null ) {
									break;
								}
							}
							if( rSubArray != null ) {
								break;
							}
						}
					}
				}
				if( rSubArray != null ) {
					break;
				}
				at = at.getParent();
			}
		}
		return rSubArray;
	}
	
	/**
	 * Find a subarray for the input symbol string <var>name</var> by searching 
	 * data clauses in a given ACCAnnotation <var>aAnnot</var>
	 * 
	 * @param aAnnot
	 * @param name
	 * @param IRSymbolOnly
	 * @return
	 */
	public static SubArray findSubArrayInDataClauses(ACCAnnotation aAnnot, String name, boolean IRSymbolOnly) {
		SubArray rSubArray = null;
		if( (name != null) && (aAnnot != null) ) {
			for( String dClause : ACCAnnotation.dataClauses ) {
				Set<SubArray> dataSet = (Set<SubArray>)aAnnot.get(dClause);
				if( dataSet != null ) {
					for( SubArray tSub : dataSet ) {
						Symbol subSym = subarrayToSymbol(tSub, IRSymbolOnly);
						if( name.equals(subSym.getSymbolName()) ) {
							rSubArray = tSub;
							break;
						}
					}
				}
				if( rSubArray != null ) {
					break;
				}
			}
		}
		return rSubArray;
	}
	
	/**
	 * Find a list of data clause and subarray for the input symbol <var>inSym</var> by searching 
	 * data clauses in a given ACCAnnotation <var>aAnnot</var>
	 * 
	 * @param aAnnot
	 * @param inSym
	 * @param IRSymbolOnly
	 * @return
	 */
	public static List findClauseNSubArrayInDataClauses(ACCAnnotation aAnnot, Symbol inSym, boolean IRSymbolOnly) {
		List retList = new ArrayList(2);
		String clause = null;
		SubArray rSubArray = null;
		if( (inSym != null) && (aAnnot != null) ) {
			for( String dClause : ACCAnnotation.dataClauses ) {
				Set<SubArray> dataSet = (Set<SubArray>)aAnnot.get(dClause);
				if( dataSet != null ) {
					for( SubArray tSub : dataSet ) {
						Symbol subSym = subarrayToSymbol(tSub, IRSymbolOnly);
						if( inSym.equals(subSym) ) {
							rSubArray = tSub;
							clause = dClause;
							break;
						}
					}
				}
				if( rSubArray != null ) {
					break;
				}
			}
		}
		if( rSubArray != null ) {
			retList.add(clause);
			retList.add(rSubArray);
		}
		return retList;
	}
	
	/**
	 * Find a list of data clause and subarray for the input symbol string <var>name</var> by searching 
	 * data clauses in a given ACCAnnotation <var>aAnnot</var>
	 * 
	 * @param aAnnot
	 * @param name
	 * @param IRSymbolOnly
	 * @return
	 */
	public static List findClauseNSubArrayInDataClauses(ACCAnnotation aAnnot, String name, boolean IRSymbolOnly) {
		List retList = new ArrayList(2);
		String clause = null;
		SubArray rSubArray = null;
		if( (name != null) && (aAnnot != null) ) {
			for( String dClause : ACCAnnotation.dataClauses ) {
				Set<SubArray> dataSet = (Set<SubArray>)aAnnot.get(dClause);
				if( dataSet != null ) {
					for( SubArray tSub : dataSet ) {
						Symbol subSym = subarrayToSymbol(tSub, IRSymbolOnly);
						if( name.equals(subSym.getSymbolName()) ) {
							rSubArray = tSub;
							break;
						}
					}
				}
				if( rSubArray != null ) {
					break;
				}
			}
		}
		if( rSubArray != null ) {
			retList.add(clause);
			retList.add(rSubArray);
		}
		return retList;
	}
	
	/**
	 * Check whether input CompoundStatement contains compute regions only.
	 * 
	 * @param cStmt
	 * @param allowACCAnnotations if true, Stane-alone OpenACC annotations are allowed in the input {\@code cStmt}
	 * @return true if input {\@code cStmt} contains only compute regions.
	 */
	public static boolean containsComputeRegionsOnly( CompoundStatement cStmt, boolean allowACCAnnotations) {
		boolean containsKernelsOnly = true;
		FlatIterator iter = new FlatIterator((Traversable)cStmt);
		Statement o = null;
		while (iter.hasNext())
		{
			boolean skip = false;
			do
			{
				o = (Statement)iter.next(Statement.class);

				if (o instanceof AnnotationStatement) {
					List<Annotation> tAList = ((AnnotationStatement)o).getAnnotations();
					if( (tAList.size() == 1) && (tAList.get(0) instanceof CommentAnnotation) ) {
						skip = true;
					} else if( allowACCAnnotations && (tAList.get(0) instanceof ACCAnnotation) ) {
						skip = true;
					} else {
						skip = false;
					}
				} else {
					skip = false;
				}

			} while ((skip) && (iter.hasNext()));

			if( !skip && !o.containsAnnotation(ACCAnnotation.class, "kernels") && 
					!o.containsAnnotation(ACCAnnotation.class, "parallel") ) {
				containsKernelsOnly = false;
				//System.err.println("Non-compute region: " + o.toString());
				break;
			}
		}
		return containsKernelsOnly;
	}
	
	/**
	 * Return true if any annotatable within the traversable tree from <var>t</var> contains 
	 * <var>redSym</var> as reduction symbol.
	 * 
	 * @param t
	 * @param redSym
	 * @param IRSymbolOnly
	 * @return
	 */
	public static boolean containsReductionSymbol(
			Traversable t, Symbol redSym, boolean IRSymbolOnly) {
		boolean foundRedSym = false;
		DFIterator<Annotatable> iter =
			new DFIterator<Annotatable>(t, Annotatable.class);
		while (iter.hasNext()) {
			Annotatable at = iter.next();
			ACCAnnotation redAnnot = at.getAnnotation(ACCAnnotation.class, "reduction");
			if( redAnnot != null ) {
				Map<ReductionOperator, Set<SubArray>> redMap = redAnnot.get("reduction");
				for( ReductionOperator rOp : redMap.keySet() ) {
					Set<SubArray> redSet = redMap.get(rOp);
					Set<Symbol> redSymSet = subarraysToSymbols(redSet, IRSymbolOnly);
					if( redSymSet != null ) {
						if( redSymSet.contains(redSym) ) {
							foundRedSym = true;
							break;
						}
						
					}
				}
			}
			if( foundRedSym ) {
				break;
			}
		}
		return foundRedSym;
	}
	
	/**
	 * Return true if any annotatable within the traversable tree from <var>t</var> contains 
	 * <var>redSym</var> as reduction symbol.
	 * 
	 * @param t
	 * @param redSyms
	 * @param IRSymbolOnly
	 * @return
	 */
	public static boolean containsReductionSymbols(
			Traversable t, Set<Symbol> redSyms, boolean IRSymbolOnly) {
		boolean foundRedSym = false;
		DFIterator<Annotatable> iter =
			new DFIterator<Annotatable>(t, Annotatable.class);
		while (iter.hasNext()) {
			Annotatable at = iter.next();
			ACCAnnotation redAnnot = at.getAnnotation(ACCAnnotation.class, "reduction");
			if( redAnnot != null ) {
				Map<ReductionOperator, Set<SubArray>> redMap = redAnnot.get("reduction");
				Set<Symbol> redSymSet = new HashSet<Symbol>();
				for( ReductionOperator rOp : redMap.keySet() ) {
					Set<SubArray> redSet = redMap.get(rOp);
					redSymSet.addAll(subarraysToSymbols(redSet, IRSymbolOnly));
				}
				redSymSet.retainAll(redSyms);
				if( redSymSet.size() > 0 ) {
					foundRedSym = true;
					break;
				}
			}
		}
		return foundRedSym;
	}
	
	/**
	 * Returns true if pragma annotations of the given type exists
	 * in the annotatable objects within the traversable object
	 * {@code t}. 
	 *
	 * @param t the traversable object to be searched.
	 * @param pragma_cls the type of pragmas to be searched for.
	 * @return true if pragmas exist; otherwise, return false
	 */
	public static <T extends PragmaAnnotation> boolean containsPragma(
			Traversable t, Class<T> pragma_cls) {
		boolean foundPragma = false;
		DFIterator<Annotatable> iter =
			new DFIterator<Annotatable>(t, Annotatable.class);
		while (iter.hasNext()) {
			Annotatable at = iter.next();
			List<T> pragmas = at.getAnnotations(pragma_cls);
			if( (pragmas != null) && !pragmas.isEmpty() ) {
				foundPragma = true;
				break;
			}
		}
		return foundPragma;
	}

	/**
	 * Returns a list of pragma annotations of the given type 
	 * and are attached to annotatable objects within the traversable object
	 * {@code t}. For example, it can collect a list of OpenACC pragmas 
	 * within a specific procedure.
	 *
	 * @param t the traversable object to be searched.
	 * @param pragma_cls the type of pragmas to be searched for.
	 * @return the list of matching pragma annotations.
	 */
	public static <T extends PragmaAnnotation> List<T> collectPragmas(
			Traversable t, Class<T> pragma_cls) {
		List<T> ret = new ArrayList<T>();
		DFIterator<Annotatable> iter =
			new DFIterator<Annotatable>(t, Annotatable.class);
		while (iter.hasNext()) {
			Annotatable at = iter.next();
			List<T> pragmas = at.getAnnotations(pragma_cls);
			ret.addAll(pragmas);
		}
		return ret;
	}
	
    /**
    * Returns a list of pragma annotations that contain the specified string
    * keys and are attached to annotatable objects within the traversable object
    * {@code t}. For example, it can collect list of OpenMP pragmas having
    * a work-sharing directive {@code for} within a specific procedure.
	 * If {@code includeAll} is true, check whether all keys in the {@code searchKeys} are included,
	 * and otherwise, check whether any key in the {@code searchKeys} is included. 
    *
    * @param t the traversable object to be searched.
    * @param pragma_cls the type of pragmas to be searched for.
    * @param searchKeys the keywords to be searched for.
	 * @param includeAll if true, search pragmas containing all keywords; otherwise search pragma containing any keywords
	 * in the {@code searchKeys} set.
    * @return the list of matching pragma annotations.
    */
    public static <T extends PragmaAnnotation> List<T> collectPragmas(
            Traversable t, Class<T> pragma_cls, Set<String> searchKeys, boolean includeAll) {
        List<T> ret = new ArrayList<T>();
        DFIterator<Annotatable> iter =
                new DFIterator<Annotatable>(t, Annotatable.class);
        while (iter.hasNext()) {
            Annotatable at = iter.next();
            List<T> pragmas = at.getAnnotations(pragma_cls);
            for (int i = 0; i < pragmas.size(); i++) {
                T pragma = pragmas.get(i);
				boolean found = false;
				for( String key: searchKeys ) {
					if ( pragma.containsKey(key) ) {
						found = true;
					} else {
						found = false;
					}
					if( includeAll ) {
						if( !found ) {
							break;
						}
					} else if( found ) {
						break;
					}
				}
				if( found ) {
					ret.add(pragma);
				}
            }
        }
        return ret;
    }

	/**
	 * Returns true if a pragma annotation exists that contain the specified string key
	 * and are attached to annotatable objects within the traversable object
	 * {@code t} interprocedurally. 
	 * If functions are called within the traversable object (@code t), 
	 * the called functions are recursively searched.
	 *
	 * @param t the traversable object to be searched.
	 * @param pragma_cls the type of pragmas to be searched for.
	 * @param key the keyword to be searched for.
	 * @return true if a matching pragma annotation exists.
	 */
	public static <T extends PragmaAnnotation> boolean
	ipContainPragmas(Traversable t, Class<T> pragma_cls, String key, Set<Procedure> accessedProcs)
	{
		if( accessedProcs == null ) {
			accessedProcs = new HashSet<Procedure>();
		}
		DFIterator<Traversable> iter = new DFIterator<Traversable>(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Annotatable )
			{
				Annotatable at = (Annotatable)o;
				List<T> pragmas = at.getAnnotations(pragma_cls);
				if( pragmas != null ) {
					for ( T pragma : pragmas )
						if ( pragma.containsKey(key) ) {
							return true;
						}
				}
			} else if( o instanceof FunctionCall ) {
				FunctionCall funCall = (FunctionCall)o;
				if( !StandardLibrary.contains(funCall) ) {
					Procedure calledProc = funCall.getProcedure();
					if( (calledProc != null) && !accessedProcs.contains(calledProc) ) { 
						accessedProcs.add(calledProc);
						if( ipContainPragmas(calledProc, pragma_cls, key, accessedProcs) ) return true;
					}
				}
			}
		}
		return false;
	}
	
	/**
	 * Returns true if a pragma annotation exists that contain the specified string keys
	 * and are attached to annotatable objects within the traversable object
	 * {@code t} interprocedurally. 
	 * If {@code includeAll} is true, check whether all keys in the {@code searchKeys} are included,
	 * and otherwise, check whether any key in the {@code searchKeys} is included. 
	 * If functions are called within the traversable object (@code t), 
	 * the called functions are recursively searched.
	 *
	 * @param t the traversable object to be searched.
	 * @param pragma_cls the type of pragmas to be searched for.
	 * @param searchKeys the keywords to be searched for.
	 * @param includeAll if true, search pragmas containing all keywords; otherwise search pragma containing any keywords
	 * in the {@code searchKeys} set.
	 * @return true if a matching pragma annotation exists.
	 */
	public static <T extends PragmaAnnotation> boolean
	ipContainPragmas(Traversable t, Class<T> pragma_cls, Set<String> searchKeys, boolean includeAll, Set<Procedure> accessedProcs)
	{
		if( accessedProcs == null ) {
			accessedProcs = new HashSet<Procedure>();
		}
		DFIterator<Traversable> iter = new DFIterator<Traversable>(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Annotatable )
			{
				Annotatable at = (Annotatable)o;
				List<T> pragmas = at.getAnnotations(pragma_cls);
				if( pragmas != null ) {
					for ( T pragma : pragmas ) {
						boolean found = false;
						for( String key: searchKeys ) {
							if ( pragma.containsKey(key) ) {
								found = true;
							} else {
								found = false;
							}
							if( includeAll ) {
								if( !found ) {
									break;
								}
							} else if( found ) {
									break;
							}
						}
						if( found ) {
							return true;
						}
					}
				}
			} else if( o instanceof FunctionCall ) {
				FunctionCall funCall = (FunctionCall)o;
				if( !StandardLibrary.contains(funCall) ) {
					Procedure calledProc = funCall.getProcedure();
					if( (calledProc != null) && !accessedProcs.contains(calledProc) ) { 
						accessedProcs.add(calledProc);
						if( ipContainPragmas(calledProc, pragma_cls, searchKeys, includeAll, accessedProcs) ) return true;
					}
				}
			}
		}
		return false;
	}

	/**
	 * Returns a list of pragma annotations that contain the specified string keys
	 * and are attached to annotatable objects within the traversable object
	 * {@code t} interprocedurally. 
	 * If functions are called within the traversable object (@code t), 
	 * the called functions are recursively searched.
	 *
	 * @param t the traversable object to be searched.
	 * @param pragma_cls the type of pragmas to be searched for.
	 * @param key the keyword to be searched for.
	 * @return the list of matching pragma annotations.
	 */
	public static <T extends PragmaAnnotation> List<T>
	ipCollectPragmas(Traversable t, Class<T> pragma_cls, String key, Set<Procedure> accessedProcs)
	{
		if( accessedProcs == null ) {
			accessedProcs = new HashSet<Procedure>();
		}
		List<T> ret = new LinkedList<T>();

		DFIterator<Traversable> iter = new DFIterator<Traversable>(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Annotatable )
			{
				Annotatable at = (Annotatable)o;
				List<T> pragmas = at.getAnnotations(pragma_cls);
				if( pragmas != null ) {
					for ( T pragma : pragmas )
						if ( pragma.containsKey(key) )
							ret.add(pragma);
				}
			} else if( o instanceof FunctionCall ) {
				FunctionCall funCall = (FunctionCall)o;
				if( !StandardLibrary.contains(funCall) ) {
					Procedure calledProc = funCall.getProcedure();
					if( (calledProc != null) && !accessedProcs.contains(calledProc) ) { 
						accessedProcs.add(calledProc);
						ret.addAll(ipCollectPragmas(calledProc, pragma_cls, key, accessedProcs));
					}
				}
			}
		}
		return ret;
	}
	
	/**
	 * Returns a list of pragma annotations that contain the specified string keys
	 * and are attached to annotatable objects within the traversable object
	 * {@code t} interprocedurally. 
	 * If {@code includeAll} is true, check whether all keys in the {@code searchKeys} are included,
	 * and otherwise, check whether any key in the {@code searchKeys} is included. 
	 * If functions are called within the traversable object (@code t), 
	 * the called functions are recursively searched.
	 *
	 * @param t the traversable object to be searched.
	 * @param pragma_cls the type of pragmas to be searched for.
	 * @param searchKeys the keywords to be searched for.
	 * @param includeAll if true, search pragmas containing all keywords; otherwise search pragma containing any keywords
	 * in the {@code searchKeys} set.
	 * @return the list of matching pragma annotations.
	 */
	public static <T extends PragmaAnnotation> List<T>
	ipCollectPragmas(Traversable t, Class<T> pragma_cls, Set<String> searchKeys, boolean includeAll, Set<Procedure> accessedProcs)
	{
		if( accessedProcs == null ) {
			accessedProcs = new HashSet<Procedure>();
		}
		List<T> ret = new LinkedList<T>();

		DFIterator<Traversable> iter = new DFIterator<Traversable>(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Annotatable )
			{
				Annotatable at = (Annotatable)o;
				List<T> pragmas = at.getAnnotations(pragma_cls);
				if( pragmas != null ) {
					for ( T pragma : pragmas ) {
						boolean found = false;
						for( String key: searchKeys ) {
							if ( pragma.containsKey(key) ) {
								found = true;
							} else {
								found = false;
							}
							if( includeAll ) {
								if( !found ) {
									break;
								}
							} else if( found ) {
								break;
							}
						}
						if( found ) {
							ret.add(pragma);
						}
					}
				}
			} else if( o instanceof FunctionCall ) {
				FunctionCall funCall = (FunctionCall)o;
				if( !StandardLibrary.contains(funCall) ) {
					Procedure calledProc = funCall.getProcedure();
					if( (calledProc != null) && !accessedProcs.contains(calledProc) ) { 
						accessedProcs.add(calledProc);
						ret.addAll(ipCollectPragmas(calledProc, pragma_cls, searchKeys, includeAll, accessedProcs));
					}
				}
			}
		}
		return ret;
	}

	/**
	 * Returns the first pragma annotation that contains the specified string keys
	 * and are attached to annotatable objects in the parent of traversable object
	 * {@code t}. 
	 * If the function enclosing the traversable object (@code t) is called in
	 * other functions, the calling functions are recursively searched.
	 *
	 * @param input the traversable object to be searched.
	 * @param pragma_cls the type of pragmas to be searched for.
	 * @param key the keyword to be searched for.
	 * @param visitedFuncs TODO
	 * @return a matching pragma annotation found first.
	 */
	public static <T extends PragmaAnnotation> T
	ipFindFirstPragmaInParent(Traversable input, Class<T> pragma_cls, String key, List<FunctionCall> funcCallList, Set<String> visitedFuncs)
	{
		T ret = null;
		if( input == null ) {
			return ret;
		}
		if( visitedFuncs == null ) {
			visitedFuncs = new HashSet<String>();
		}
		Traversable t = input.getParent();
		while( t != null ) {
			if( t instanceof Annotatable ) {
				Annotatable at  = (Annotatable)t;
				T tAnnot = at.getAnnotation(pragma_cls, key);
				if( tAnnot != null ) {
					ret = tAnnot;
					break;
				} 
			}
			if( t instanceof Procedure ) {
				Procedure tProc = (Procedure)t;
				List<FunctionCall> fCallList = funcCallList; 
				if( fCallList == null ) {
					while( t != null ) {
						if( t instanceof Program ) {
							break;
						}
						t = t.getParent();
					}
					if( t instanceof Program ) {
						fCallList = IRTools.getFunctionCalls(t);
					}
				}
				if( fCallList != null ) {
					for( FunctionCall fCall : fCallList ) {
						if( fCall.getName().equals(tProc.getName()) ) {
							T tAnnot = fCall.getStatement().getAnnotation(pragma_cls, key);
							if( tAnnot != null ) {
								ret = tAnnot;
								break;
							} else {
								if( !visitedFuncs.contains(tProc.getSymbolName()) ) {
									visitedFuncs.add(tProc.getSymbolName());
									tAnnot = ipFindFirstPragmaInParent(fCall.getStatement(), pragma_cls, key, fCallList, visitedFuncs);
								}
								if( tAnnot != null ) {
									ret = tAnnot;
									break;
								}
							}
						}
					}
				}
				break;
			}
			t = t.getParent();
		}
		return ret;
	}
	
	/**
	 * Returns the first pragma annotation that contains the specified string keys
	 * and are attached to annotatable objects in the parent of traversable object
	 * {@code t}. 
	 * If the function enclosing the traversable object (@code t) is called in
	 * other functions, the calling functions are recursively searched.
	 * If {@code includeAll} is true, check whether all keys in the {@code searchKeys} are included,
	 * and otherwise, check whether any key in the {@code searchKeys} is included. 
	 *
	 * @param input the traversable object to be searched.
	 * @param pragma_cls the type of pragmas to be searched for.
	 * @param searchKeys the keywords to be searched for.
	 * @param includeAll if true, search pragmas containing all keywords; otherwise search pragma containing any keywords
	 * in the {@code searchKeys} set.
	 * @param visitedFuncs TODO
	 * @return a matching pragma annotation found first.
	 */
	public static <T extends PragmaAnnotation> T
	ipFindFirstPragmaInParent(Traversable input, Class<T> pragma_cls, Set<String> searchKeys, boolean includeAll,
			List<FunctionCall> funcCallList, Set<String> visitedFuncs)
	{
		T ret = null;
		if( input == null ) {
			return ret;
		}
		if( visitedFuncs == null ) {
			visitedFuncs = new HashSet<String>();
		}
		Traversable t = input.getParent();
		while( t != null ) {
			if( t instanceof Annotatable ) {
				Annotatable at  = (Annotatable)t;
				T tAnnot = null;
				boolean found = false;
				for( String key: searchKeys ) {
					tAnnot = at.getAnnotation(pragma_cls, key);
					if( tAnnot != null ) {
						found = true;
					} else {
						found = false;
					}
					if( includeAll ) {
						if( !found ) {
							break;
						}
					} else if( found ) {
						break;
					}
				}
				if( found ) {
					ret = tAnnot;
					break;
				}
			}
			if( t instanceof Procedure ) {
				Procedure tProc = (Procedure)t;
				List<FunctionCall> fCallList = funcCallList; 
				if( fCallList == null ) {
					while( t != null ) {
						if( t instanceof Program ) {
							break;
						}
						t = t.getParent();
					}
					if( t instanceof Program ) {
						fCallList = IRTools.getFunctionCalls(t);
					}
				}
				if( fCallList != null ) {
					for( FunctionCall fCall : fCallList ) {
						if( fCall.getName().equals(tProc.getName()) ) {
							Annotatable at  = (Annotatable)fCall.getStatement();
							T tAnnot = null;
							boolean found = false;
							for( String key: searchKeys ) {
								tAnnot = at.getAnnotation(pragma_cls, key);
								if( tAnnot != null ) {
									found = true;
								} else {
									found = false;
								}
								if( includeAll ) {
									if( !found ) {
										break;
									}
								} else if( found ) {
									break;
								}
							}
							if( found ) {
								ret = tAnnot;
								break;
							} else {
								if( !visitedFuncs.contains(tProc.getSymbolName()) ) {
									visitedFuncs.add(tProc.getSymbolName());
									tAnnot = ipFindFirstPragmaInParent(at, pragma_cls, searchKeys, includeAll, 
										fCallList, visitedFuncs);
								}
								if( tAnnot != null ) {
									ret = tAnnot;
									break;
								}
							}
						}
					}
				}
				break;
			}
			t = t.getParent();
		}
		return ret;
	}
	
    /**
    * Returns a list of FunctionCall expressions within the traversable object
    * interprocedurally.
    *
    * @param t the traversable object to be searched.
    * @return the list of function calls that appear in {@code t}.
    */
    public static Set<Procedure> ipGetCalledProcedures(Traversable t, Set<Procedure> accessedProcs) {
    	if( accessedProcs == null ) {
    		accessedProcs = new HashSet<Procedure>();
    	}
    	Set<Procedure> procList = new HashSet<Procedure>();
        if (t == null) {
            return null;
        }
        List<FunctionCall> fCallList = ((new DFIterator<FunctionCall>(t, FunctionCall.class)).getList());
        for( FunctionCall fCall : fCallList ) {
        	if( !StandardLibrary.contains(fCall) ) {
        		Procedure proc = fCall.getProcedure();
        		if( (proc != null) && !accessedProcs.contains(proc) ) {
        			accessedProcs.add(proc);
        			procList.add(proc);
        			Set<Procedure> ipList = ipGetCalledProcedures(proc.getBody(), accessedProcs);
        			if( ipList != null ) {
        				procList.addAll(ipList);
        			}
        		}
        	}
        }
        return procList;
    }
	
    /**
    * Returns the innermost pragma annotation that contain the specified string
    * keys and are attached to annotatable objects within the traversable object
    * {@code t}. For example, it can return the innermost worker loop annotation
    * within specific nested loops.
    * CAUTION: if multiple innermost pragmas exist, this method returns the last
    * found one; it does not compare nesting levels.
    *
    * @param t the traversable object to be searched.
    * @param pragma_cls the type of pragmas to be searched for.
    * @param searchKey the keywords to be searched for.
    * @return the innermost matching pragma annotation.
    */
    public static <T extends PragmaAnnotation> T findInnermostPragma(
            Traversable t, Class<T> pragma_cls, String searchKey ) {
        T ret = null;
        BreadthFirstIterator iter =
        		new BreadthFirstIterator(t);
        for (;;)
        {
        	Annotatable at = null;

        	try {
        		at = (Annotatable)iter.next(Annotatable.class);
        	} catch (NoSuchElementException e) {
        		break;
        	}
        	T tAnnot = at.getAnnotation(pragma_cls, searchKey);
        	if( tAnnot != null ) {
        		ret = tAnnot;
        	}
        }
        return ret;
    }
    
    /**
    * Returns the innermost pragma annotation that contain the specified string
    * keys and are attached to annotatable objects within the traversable object
    * {@code t}. For example, it can return the innermost worker loop annotation
    * within specific nested loops.
	* If {@code includeAll} is true, check whether all keys in the {@code searchKeys} are included,
	* and otherwise, check whether any key in the {@code searchKeys} is included. 
    * CAUTION: if multiple innermost pragmas exist, this method returns the last
    * found one; it does not compare nesting levels.
    *
    * @param t the traversable object to be searched.
    * @param pragma_cls the type of pragmas to be searched for.
    * @param searchKeys the keywords to be searched for.
	 * @param includeAll if true, search pragmas containing all keywords; otherwise search pragma containing any keywords
	 * in the {@code searchKeys} set.
    * @return the innermost matching pragma annotation.
    */
    public static <T extends PragmaAnnotation> T findInnermostPragma(
            Traversable t, Class<T> pragma_cls, Set<String> searchKeys, boolean includeAll) {
        T ret = null;
        BreadthFirstIterator iter =
        		new BreadthFirstIterator(t);
        for (;;)
        {
        	Annotatable at = null;

        	try {
        		at = (Annotatable)iter.next(Annotatable.class);
        	} catch (NoSuchElementException e) {
        		break;
        	}
        	List<T> pragmas = at.getAnnotations(pragma_cls);
        	for (int i = 0; i < pragmas.size(); i++) {
        		T pragma = pragmas.get(i);
        		boolean found = false;
        		for( String key: searchKeys ) {
        			if ( pragma.containsKey(key) ) {
        				found = true;
        			} else {
        				found = false;
        			}
        			if( includeAll ) {
        				if( !found ) {
        					break;
        				}
        			} else if( found ) {
        				break;
        			}
        		}
        		if( found ) {
        			ret = pragma;
        		}
        	}
        }
        return ret;
    }
    
    /**
    * Returns a list of outermost pragma annotations that contain the specified string
    * keys and are attached to annotatable objects within the traversable object
    * {@code t}. For example, it can return the outermost worker loop annotations
    * within specific nested loops.
    *
    * @param t the traversable object to be searched.
    * @param pragma_cls the type of pragmas to be searched for.
    * @param searchKey the keyword to be searched for.
    * @return the list of outermost matching pragma annotations.
    */
    public static <T extends PragmaAnnotation> List<T> collectOutermostPragmas(
    		Traversable t, Class<T> pragma_cls, String searchKey ) {
    	List<T> ret = new ArrayList<T>();
    	DFIterator<Annotatable> iter =
    		new DFIterator<Annotatable>(t, Annotatable.class);
    	while (iter.hasNext()) {
    		Annotatable at = iter.next();
    		T tAnnot = at.getAnnotation(pragma_cls, searchKey);
    		if( tAnnot != null ) {
    			Traversable tt = at.getParent();
    			boolean isOutermost = true; 
    			while( (tt != null) ) {
    				if( ((Annotatable)tt).containsAnnotation(pragma_cls, searchKey) ) {
    					isOutermost = false;
    					break;
    				}
    				tt = tt.getParent();
    			}
    			if( isOutermost ) {
    				ret.add(tAnnot);
    			}
    		}
    	}
    	return ret;
    }
	
	/**
	 * Remove pragma annotations of the given type 
	 * that are attached to annotatable objects within the traversable object
	 * {@code t}.
	 *
	 * @param t the traversable object to be searched.
	 * @param pragma_cls the type of pragmas to be removed.
	 */
	public static <T extends PragmaAnnotation> void removePragmas(
			Traversable t, Class<T> pragma_cls) {
		DFIterator<Annotatable> iter =
			new DFIterator<Annotatable>(t, Annotatable.class);
		while (iter.hasNext()) {
			Annotatable at = iter.next();
			at.removeAnnotations(pragma_cls);
		}
	}

	/**
	 * This method is an extended version of {@link cetus.analysis.LoopTools#isPerfectNest(Loop)}, which
	 * ignores comment statements in a loop body.
	 * FIXME: this may not handle cases where OpenACC cache directives exist.
	 * 
	 * @param outerloop input loop to be checked whether it is perfectly nested or not
	 * @param nestLevel a depth of nested loops to be checked ("nestLevel =< 1" means no loop nesting)
	 * @param nestedLoops a list of perfectly nested loops upto level {@code nestLevel}, which excludes the input {@code outerloop}
	 * @param commentList a list of comment statements included in the loop bodies of nested loops upto level {@code nestLevel} 
	 * @return true if the input loop {@code outerloop} is perfectly nested upto level {@code nestLevel}. "nestLevel =< 1" always return true.
	 */
	public static boolean extendedPerfectlyNestedLoopChecking(ForLoop outerloop, int nestLevel, 
			List<ForLoop> nestedLoops, List<AnnotationStatement> commentList) {
		boolean pnest = true;
		ForLoop currLoop = outerloop;
		int i = 1;
		while( i<nestLevel ) {
			//Find a nested loop.
			Statement fBody = currLoop.getBody();
			currLoop = null;
			FlatIterator iter = new FlatIterator((Traversable)fBody);
			Object o = null;
			while (iter.hasNext())
			{
				boolean skip = false;
				do
				{
					o = (Statement)iter.next(Statement.class);

					if (o instanceof AnnotationStatement) {
						List<Annotation> tAList = ((AnnotationStatement)o).getAnnotations();
						if( (tAList.size() == 1) && (tAList.get(0) instanceof CommentAnnotation) ) {
							skip = true;
							if( commentList != null ) {
								commentList.add((AnnotationStatement)o);
							}
						} else {
							skip = false;
						}
					} else {
						skip = false;
					}

				} while ((skip) && (iter.hasNext()));

				if (o instanceof ForLoop)
				{
					if( currLoop == null ) {
						currLoop = (ForLoop)o;
					} else {
						pnest = false;
						break;
					}
				}
				else {
					pnest = false;
					break;
				}
			}
			if( (!pnest) || (currLoop == null) ) {
				pnest = false;
				break;
			} else if( nestedLoops != null ){
				nestedLoops.add(currLoop);
			}
			i++;
		}
		return pnest;
	}

	/**
	 * Find all loops containing the clause {@code clause} and directly nested from the input loop {@code tloop}
	 * If the input loop {@code tloop} contains the clause {@code clause}, it is also included in the output list.
	 * If a collapse clause exists, the search continues upto the collapse level. 
	 * 
	 * @param tloop loop to be searched
	 * @param clause OpenACC clause to be searched
	 * @return a list of directly nested loops containing the clause {@code clause}
	 */
	public static List<ForLoop> findDirectlyNestedLoopsWithClause(ForLoop tloop, String clause) {
		boolean pnest = true;
		ForLoop currLoop = tloop;
		List<ForLoop> nestedLoops = new LinkedList<ForLoop>();

		if( currLoop == null ) { return nestedLoops; }
		else if( currLoop.containsAnnotation(ACCAnnotation.class, clause) ) {
			nestedLoops.add(currLoop);
		} else if( currLoop.containsAnnotation(ARCAnnotation.class, clause) ) {
			nestedLoops.add(currLoop);
		} else {
			return nestedLoops;
		}
		int collapseLevel = 1;
		while( pnest ) {
			//Find a perfectly nested loop.
			Statement fBody = currLoop.getBody();
			collapseLevel--;
			currLoop = null;
			FlatIterator iter = new FlatIterator((Traversable)fBody);
			Object o = null;
			while (iter.hasNext())
			{
				boolean skip = false;
				do
				{
					o = (Statement)iter.next(Statement.class);

					if (o instanceof AnnotationStatement) {
						List<Annotation> tAList = ((AnnotationStatement)o).getAnnotations();
						if( (tAList.size() == 1) && (tAList.get(0) instanceof CommentAnnotation) ) {
							skip = true;
						} else {
							skip = false;
							//System.out.println("Non-annotation statement" + (Statement)o);
						}
					} else {
						skip = false;
					}

				} while ((skip) && (iter.hasNext()));

				if (o instanceof ForLoop)
				{
					if( currLoop == null ) {
						currLoop = (ForLoop)o;
					} else {
						pnest = false;
						break;
					}
				}
				else {
					pnest = false;
					break;
				}
			}
			if( (!pnest) || (currLoop == null) ) {
				pnest = false;
				break;
			} else {
				ACCAnnotation tAnnot = currLoop.getAnnotation(ACCAnnotation.class, "collapse");
				if( tAnnot != null ) {
					int tVal = (int)((IntegerLiteral)tAnnot.get("collapse")).getValue();
					if( tVal > 1 ) {
						collapseLevel += tVal;
					} else {
						collapseLevel = 1;
					}
				}
				if( currLoop.containsAnnotation(ACCAnnotation.class, clause) ||
						currLoop.containsAnnotation(ARCAnnotation.class, clause) ){
					nestedLoops.add(currLoop);
					if( tAnnot == null ) {
						collapseLevel = 1;
					}
				} else if( collapseLevel <= 0 ) {
					pnest = false;
					break;
				}
			}
		}
		return nestedLoops;
	}
	
	/**
	 * Replace old IR symbol ({@code oldIRSym}) in {@code oldPSym} with new one ({@code newIRSym}).
	 * 
	 * @param oldPSym PseudoSymbol contains the old IR symbol ({@code oldIRSym})
	 * @param oldIRSym old IR symbol to be replaced
	 * @param newIRSym new IR symbol
	 * @return PseudoSymbol with the new IR symbol ({@code newIRSym})
	 */
	protected static PseudoSymbol replaceIRSymbol(PseudoSymbol oldPSym, Symbol oldIRSym, Symbol newIRSym) {
		PseudoSymbol newPSym = null;
		if( (oldPSym == null) || (oldIRSym == null) || (newIRSym == null) ) {
			return null;
		}
		if( oldPSym instanceof AccessSymbol ) {
			Symbol base = ((AccessSymbol) oldPSym).getBaseSymbol();
			Symbol member = ((AccessSymbol) oldPSym).getMemberSymbol();
			if( base.equals(oldIRSym) ) {
				newPSym = new AccessSymbol(newIRSym, member);
			} else if (base instanceof PseudoSymbol ) {
				Symbol newBase = replaceIRSymbol((PseudoSymbol)base, oldIRSym, newIRSym);
				newPSym = new AccessSymbol(newBase, member);
			} else {
				PrintTools.println("\n[WARNING in AnalysisTools.replaceIRSymbol()] cannot find old IR symbol to be replaced:\n" +
						"Old IR symbol: " + oldIRSym + "\nPseudo symbol: " + oldPSym + ".\n", 0);
			}
		} else if( oldPSym instanceof DerefSymbol ) {
			Symbol refSym = ((DerefSymbol)oldPSym).getRefSymbol();
			if( refSym.equals(oldIRSym) ) {
				newPSym = new DerefSymbol(newIRSym);
			} else if( refSym instanceof PseudoSymbol ) {
				Symbol newRefSym = replaceIRSymbol((PseudoSymbol)refSym, oldIRSym, newIRSym);
				newPSym = new DerefSymbol(newRefSym);
			} else {
				PrintTools.println("\n[WARNING in AnalysisTools.replaceIRSymbol()] cannot find old IR symbol to be replaced:\n" +
						"Old IR symbol: " + oldIRSym + "\nPseudo symbol: " + oldPSym + ".\n", 0);
			}
		}
		return newPSym;
	}

	/**
	 * Find the original symbol that the input extern symbol, inSym, refers to.
	 * If the symbol refers to predefined C variables (stdin, stdout, stderr), 
	 * return it as it is.
	 * 
	 * @param inSym input extern symbol
	 * @param prog program where the input symbol belongs.
	 * @return returns the original symbol that the input extern symbol refers to.
	 *         If the input symbol is not an extern symbol, return input symbol itself.
	 *         If no original symbol is found, return null.
	 */
	public static Symbol getOrgSymbolOfExternOne(Symbol inSym, Program prog) {
		Symbol orgSym = null;
		Symbol IRSym = null;
		if( inSym == null ) {
			return orgSym;
		}
		if( inSym instanceof PseudoSymbol ) {
			IRSym = ((PseudoSymbol)inSym).getIRSymbol();
		} else {
			IRSym = inSym;
		}
		if(!SymbolTools.containsSpecifier(IRSym, Specifier.EXTERN)) {
			return inSym;
		}
		String extSName = IRSym.getSymbolName();
		//Return predefined C variable as it is.
		if( predefinedCVarList.contains(extSName) ) {
			return inSym;
		}
		for ( Traversable tt : prog.getChildren() )
		{
			if( orgSym != null ) {
				break;
			}
			Set<Symbol> gSymbols = SymbolTools.getGlobalSymbols(tt);
			for( Symbol gSym : gSymbols ) {
				if(gSym.getSymbolName().equals(extSName)) {
					if(!SymbolTools.containsSpecifier(gSym, Specifier.EXTERN)) {
						orgSym = gSym;
						break;
					}
				}
			}
		}
		if( orgSym == null ) {
			return null;
		}
		if( inSym instanceof PseudoSymbol ) {
			orgSym = replaceIRSymbol((PseudoSymbol)inSym, IRSym, orgSym);
		}
		return orgSym;
	}
	
	static public class SymbolStatus {
		public static int ExternGlobal = 4;
		public static int Global = 3;
		public static int Parameter = 2;
		public static int Local = 1;
		public static int Invisible = 0;
		public static int MultipleArguments = -1;
		public static int ComplexArgument = -2;
		public static int NotSupported = -3;
		public static int NotSearchable = -4;
		public static int UnknownError = -5;
		public static boolean OrgSymbolFound(int status) {
			return (Invisible < status);
		}
		public static boolean SrcScalarSymbolFound(int status) {
			return (NotSearchable < status);
		}
		public static boolean isGlobal(int status) {
			return (Global <= status);
		}
	}

	/**
	 * Find the original symbol of the input symbol, inSym.
	 * If symbol scope, {@code symScope}, is given, and if the input symbol
	 * is a function parameter, the original symbol is searched only upto the
	 * symbol scope.
	 * CAVEAT: This method assumes that {@code inSym} is visible to the traversable {@code t}
	 * 
	 * If the input symbol is a global variable,
	 *     - return a list of the symbol and the parent TranslationUnit
	 * Else if the input symbol is a function parameter
	 *     If the parent function is the same as the {@code symScope}
	 *         - return a list of the symbol and the parent Procedure
	 *     Else if the input symbol is a value-passed parameter of the function
	 *         - If {@code symScope} == null
	 *             - return a list of the symbol and the parent Procedure
	 *         - Else
	 *             - return an empty list
	 *     Else if the argument for the parameter is unique, 
	 *         - call this method again with the argument to keep searching.
	 *     Else if the argument for the parameter is complex, 
	 *         - return a list of the input symbol only.
	 *     Else 
	 *         - return an empty list.
	 * Else if the input symbol is a (static) local variable 
	 *     If the symbol is visible in the {@code symScope} or if {@code symScope} is null
	 *         - return a list of the symbol and the parent Procedure
	 *     Else
	 *         - return an empty list
	 * 
	 * 
	 * @param inSym input symbol
	 * @param t traversable from which the original symbol is searched.
	 * @param noExtern set true to find the original symbol without extern specifier
	 * @param symScope procedure which sets the scope of the original symbol search
	 * @param symbolInfo a list of found symbol and enclosing function/translation unit
	 * @param fCallList a list of function calls in the program
	 * @return status of this search result
	 */
	public static int findOrgSymbol(Symbol inSym, Traversable t, boolean noExtern, Procedure symScope,
			List symbolInfo, List<FunctionCall> fCallList) {
		int symbolStatus = SymbolStatus.UnknownError;
		Symbol tSym = null;
		Symbol IRSym = null;
		Set<Symbol> symSet = null;
		Procedure p_proc = null;
		TranslationUnit t_unit = null;
		Program program = null;
		if( (inSym == null) || (t == null) ) {
			return SymbolStatus.NotSearchable;
		}
		// Find a parent Procedure.
		while (t != null) {
			if (t instanceof Procedure) break;
			t = t.getParent(); 
		}
		p_proc = (Procedure)t;
		// Find a parent TranslationUnit.
		while (t != null) {
			if (t instanceof TranslationUnit) break;
			t = t.getParent(); 
		}
		t_unit = (TranslationUnit)t;
		if( t_unit == null ) {
			return SymbolStatus.NotSearchable;
		}
		program = (Program)t_unit.getParent();
		if( inSym instanceof PseudoSymbol ) {
			IRSym = ((PseudoSymbol) inSym).getIRSymbol();
		} else {
			IRSym = inSym;
		}
		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		// If the input symbol is a global variable, return a list of the symbol and the parent TranslationUnit. //
		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		if( SymbolTools.isGlobal(inSym) ) {
			tSym = getOrgSymbolOfExternOne(inSym, program);
			if( noExtern ) {
				if( tSym == null ) {
					PrintTools.println("\n[WARNING in findOrgSymbol()] Can not find the original symbol" +
							" that the extern symbol (" + inSym + ") refers to.\n", 1); 
					tSym = inSym;
					symbolStatus = SymbolStatus.ExternGlobal;
				} else {
					symbolStatus = SymbolStatus.Global;
				}
			}else {
				if( (tSym == null) || (!tSym.equals(inSym)) ) {
					symbolStatus = SymbolStatus.ExternGlobal;
				} else {
					symbolStatus = SymbolStatus.Global;
				}
				tSym = inSym;
			}
			symbolInfo.add(tSym);
			symbolInfo.add(t_unit);
			return symbolStatus;
		} else if( SymbolTools.isFormal(IRSym) ) {
			//The input symbol is a formal function parameter.
			String pName = p_proc.getSymbolName();
			symSet = SymbolTools.getParameterSymbols(p_proc);
			if( symSet.contains(IRSym) ) {
				if( (symScope !=null) && (symScope.getSymbolName().equals(p_proc.getSymbolName())) ) {
					//Stop searching since this procedure is the same as {@code symScope}.
					symbolInfo.add(inSym);
					symbolInfo.add(p_proc);
					return SymbolStatus.Parameter;
				} else if( pName.equals("main") || pName.equals("MAIN__") ) {
					//We don't have to check a formal parameter of main function.
					if( symScope == null ) {
						symbolStatus = SymbolStatus.Parameter;
					} else {
						symbolStatus = SymbolStatus.Invisible;
					}
					symbolInfo.add(inSym);
					symbolInfo.add(p_proc);
					return symbolStatus;
				} else if( SymbolTools.isScalar(IRSym) && !SymbolTools.isPointer(IRSym) &&
						!SymbolTools.isArray(IRSym) && !(IRSym instanceof NestedDeclarator) ){
					//The parameter is a non-pointer, scalar variable; stop searching.
					if( symScope == null ) {
						symbolStatus = SymbolStatus.Parameter;
					} else {
						symbolStatus = SymbolStatus.Invisible;
					}
					symbolInfo.add(inSym);
					symbolInfo.add(p_proc);
					return symbolStatus;
				}else {
					//Find corresponding argument of this formal parameter.
					List<FunctionCall> funcCallList;
					if( fCallList == null ) {
						funcCallList = IRTools.getFunctionCalls(program);
					} else {
						funcCallList = fCallList;
					}
					// Find the caller procedure that called this procedure.
					List paramList = p_proc.getParameters();
					int list_size = paramList.size();
					if( list_size == 1 ) {
						Object obj = paramList.get(0);
						String paramS = obj.toString();
						// Remove any leading or trailing whitespace.
						paramS = paramS.trim();
						if( paramS.equals(Specifier.VOID.toString()) ) {
							list_size = 0;
						}
					}
					Procedure t_proc = null;
					Symbol currArgSym = null;
					Symbol prevArgSym = null;
					boolean foundArg = false;
					for( FunctionCall funcCall : funcCallList ) {
						if(p_proc.getName().equals(funcCall.getName())) {
							List argList = funcCall.getArguments();
							for( int i=0; i<list_size; i++ ) {
								////////////////////////////////////////////////////////////////////////
								// DEBUG: IRTools.containsSymbol() works only for searching           //
								// symbols in expression tree; it internally compare Identifier only, //
								// but VariableDeclarator contains NameID instead of Identifier.      //
								////////////////////////////////////////////////////////////////////////
								//if( IRTools.containsSymbol((Declaration)paramList.get(i), inSym)) 
								List declaredSyms = ((Declaration)paramList.get(i)).getDeclaredIDs();
								if( declaredSyms.contains(new Identifier(IRSym)) ) {
									// Found an actual argument for the inSym. 
									foundArg = true;
									t = funcCall.getStatement();
									while( (t != null) && !(t instanceof Procedure) ) {
										t = t.getParent();
									}
									t_proc = (Procedure)t;
									Expression exp = (Expression)argList.get(i);
									///////////////////////////////////////////////////////////////////
									// LIMIT: if the passed argument is complex, current translator  //
									// can not calculate the region accessed in the called function. //
									// Therefore, only the following expressions are handled for now.//
									//     - Simple pointer identifier (ex: a)                       //
									//     - Simple unary expression (ex: &b)                        //
									//     - Simple binary expression where only one symbol exists   //
									//       (ex: b + 1)                                             //
									///////////////////////////////////////////////////////////////////
									if( exp instanceof BinaryExpression ) {
										BinaryExpression biExp = (BinaryExpression)exp;
										Symbol lsym = SymbolTools.getSymbolOf(biExp.getLHS());
										Symbol rsym = SymbolTools.getSymbolOf(biExp.getRHS());
										if( (lsym == null) && (rsym == null) ) {
											currArgSym = null;
										} else if( (lsym != null) && (rsym == null) ) {
											currArgSym = lsym;
										} else if( (lsym == null) && (rsym != null) ) {
											currArgSym = rsym;
										} else {
											currArgSym = null;
										}
									} else {
										currArgSym = SymbolTools.getSymbolOf(exp);
									}
									if( currArgSym == null ) {
										symbolInfo.add(inSym);
										symbolInfo.add(p_proc);
										if( !(exp instanceof Literal) ) {
											PrintTools.println("\n[WARNING in findOrgSymbol()] argument (" + exp + 
													") passed for " + "the parameter symbol (" + IRSym + 
													") of a procedure, " + p_proc.getName() + ", has complex expression; " +
													"failed to analyze it.\n",0);
											return SymbolStatus.ComplexArgument;
										} else {
											if( symScope == null ) {
												symbolStatus = SymbolStatus.Parameter;
											} else {
												symbolStatus = SymbolStatus.Invisible;
											}
											return symbolStatus;
										}
									} else if( (prevArgSym != null) && (!currArgSym.equals(prevArgSym)) ) {
										Symbol pSym = getOrgSymbolOfExternOne(prevArgSym, program);
										Symbol cSym = getOrgSymbolOfExternOne(currArgSym, program);
										if( (pSym == null) || !pSym.equals(cSym) ) {
											// Multiple argument symbols are found.
											PrintTools.println("\n[WARNING in findOrgSymbol()] multiple argments exist " +
													"for the parameter symbol (" + IRSym + ") of procedure (" 
													+ p_proc.getSymbolName() + "); can't find the original symbol.\n", 1);
											symbolInfo.add(inSym);
											symbolInfo.add(p_proc);
											return SymbolStatus.MultipleArguments;
										}
									}
									prevArgSym = currArgSym;
								}
							}
						}
					} //end of funcCallList loop
					if( foundArg ) { //found a unique argument symbol for the current parameter symbol.
						symbolStatus = findOrgSymbol(currArgSym, t_proc, noExtern, symScope, symbolInfo, funcCallList);
						if( inSym instanceof PseudoSymbol ) {
							if( !symbolInfo.isEmpty() ) {
								Symbol newIRSym = (Symbol)symbolInfo.remove(0);
								inSym = replaceIRSymbol((PseudoSymbol)inSym, IRSym, newIRSym);
								symbolInfo.add(0, inSym);
							}
						}
						return symbolStatus;
					} else { //not corresponding argument symbol is found (implementation bug)
						PrintTools.println("\n[WARNING in findOrgSymbol()] Input symbol (" + IRSym + ") is a formal function parameter," +
								" but corresponding argument is not found in the current scope (enclosing procedure: "+ p_proc.getSymbolName() +
								").\n", 1); 
						symbolInfo.add(inSym);
						symbolInfo.add(p_proc);
						return SymbolStatus.UnknownError;
					}
				}
			} else {
				PrintTools.println("\n[WARNING in findOrgSymbol()] Input symbol (" + IRSym + ") is a formal function parameter," +
						" but it is not visible in the current scope (enclosing procedure: "+ p_proc.getSymbolName() +
						").\n", 1); 
				symbolInfo.add(inSym);
				symbolInfo.add(p_proc);
				return SymbolStatus.Invisible;
			}
		} else if (SymbolTools.isLocal(IRSym) ) {
			Traversable tt = (Traversable)IRSym;
			while ( (tt != null) && !(tt instanceof Procedure) ){
				tt = tt.getParent();
			}
			if( (tt instanceof Procedure) && ((Procedure)tt).getSymbolName().equals(p_proc.getSymbolName()) ) {
				if( (symScope == null) || p_proc.getSymbolName().equals(symScope.getSymbolName()) ) {
					symbolStatus = SymbolStatus.Local;
				} else {
					symbolStatus = SymbolStatus.Invisible;
				}
			} else {
				PrintTools.println("\n[WARNING in findOrgSymbol()] Input symbol (" + IRSym + ") is a local symbol," +
						" but it is not visible in the current scope (enclosing procedure: "+ p_proc.getSymbolName() +
						").\n", 1); 
				symbolStatus = SymbolStatus.Invisible;
			}
			symbolInfo.add(inSym);
			symbolInfo.add(p_proc);
			return symbolStatus;
		} else {
			PrintTools.println("\n[WARNING in findOrgSymbol()] Can not determine the type of the input symbol (" + inSym + 
					" in the procedure, " + p_proc.getSymbolName() + ");" +
					" return an empty list.\n", 1); 
			symbolInfo.add(inSym);
			symbolInfo.add(p_proc);
			return SymbolStatus.UnknownError;
		}
	}
	
	/**
	 * Find the original symbol of the input symbol, inSym.
	 * If symbol scope, {@code symScope}, is given, and if the input symbol
	 * is a function parameter, the original symbol is searched only upto the
	 * symbol scope.
	 * CAVEAT: This method assumes that {@code inSym} is visible to the traversable {@code t}
	 * 
	 * If the input symbol is a global variable,
	 *     - return a list of the symbol and the parent TranslationUnit
	 * Else if the input symbol is a function parameter
	 *     If the parent function is the same as the {@code symScope}
	 *         - return a list of the symbol and the parent Procedure
	 *     Else if the input symbol is a value-passed parameter of the function
	 *         - If {@code symScope} == null
	 *             - return a list of the symbol and the parent Procedure
	 *         - Else
	 *             - return an empty list
	 *     Else if the argument for the parameter is unique, 
	 *         - call this method again with the argument to keep searching.
	 *     Else if the argument for the parameter is complex, 
	 *         - return a list of the input symbol only.
	 *     Else 
	 *         - return an empty list.
	 * Else if the input symbol is a (static) local variable 
	 *     If the symbol is visible in the {@code symScope} or if {@code symScope} is null
	 *         - return a list of the symbol and the parent Procedure
	 *     Else
	 *         - return an empty list
	 * 
	 * 
	 * @param inSym input symbol
	 * @param t traversable from which the original symbol is searched.
	 * @param noExtern set true to find the original symbol without extern specifier
	 * @param symScope procedure which sets the scope of the original symbol search
	 * @param symbolInfo a list of found symbol and enclosing function/translation unit
	 * @param fCallList a list of function calls in the program
	 * @return status of this search result
	 */
	public static int findSrcScalarSymbol(Symbol inSym, Traversable t, boolean noExtern, Procedure symScope,
			List symbolInfo, List<FunctionCall> fCallList) {
		int symbolStatus = SymbolStatus.UnknownError;
		Symbol tSym = null;
		Symbol IRSym = null;
		Set<Symbol> symSet = null;
		Procedure p_proc = null;
		TranslationUnit t_unit = null;
		Program program = null;
		if( (inSym == null) || (t == null) ) {
			return SymbolStatus.NotSearchable;
		}
		// Find a parent Procedure.
		while (t != null) {
			if (t instanceof Procedure) break;
			t = t.getParent(); 
		}
		p_proc = (Procedure)t;
		// Find a parent TranslationUnit.
		while (t != null) {
			if (t instanceof TranslationUnit) break;
			t = t.getParent(); 
		}
		t_unit = (TranslationUnit)t;
		if( t_unit == null ) {
			return SymbolStatus.NotSearchable;
		}
		program = (Program)t_unit.getParent();
		if( inSym instanceof PseudoSymbol ) {
			IRSym = ((PseudoSymbol) inSym).getIRSymbol();
		} else {
			IRSym = inSym;
		}
		if( SymbolTools.isPointer(IRSym) || 
			SymbolTools.isArray(IRSym) || (IRSym instanceof NestedDeclarator) ){
			return SymbolStatus.NotSupported;
		}
		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		// If the input symbol is a global variable, return a list of the symbol and the parent TranslationUnit. //
		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		if( SymbolTools.isGlobal(inSym) ) {
			tSym = getOrgSymbolOfExternOne(inSym, program);
			if( noExtern ) {
				if( tSym == null ) {
					PrintTools.println("\n[WARNING in findSrcScalarSymbol()] Can not find the original symbol" +
							" that the extern symbol (" + inSym + ") refers to.\n", 1); 
					tSym = inSym;
					symbolStatus = SymbolStatus.ExternGlobal;
				} else {
					symbolStatus = SymbolStatus.Global;
				}
			}else {
				if( (tSym == null) || (!tSym.equals(inSym)) ) {
					symbolStatus = SymbolStatus.ExternGlobal;
				} else {
					symbolStatus = SymbolStatus.Global;
				}
				tSym = inSym;
			}
			symbolInfo.add(tSym);
			symbolInfo.add(t_unit);
			return symbolStatus;
		} else if( SymbolTools.isFormal(IRSym) ) {
			//The input symbol is a formal function parameter.
			String pName = p_proc.getSymbolName();
			symSet = SymbolTools.getParameterSymbols(p_proc);
			if( symSet.contains(IRSym) ) {
				if( (symScope !=null) && (symScope.getSymbolName().equals(p_proc.getSymbolName())) ) {
					//Stop searching since this procedure is the same as {@code symScope}.
					symbolInfo.add(inSym);
					symbolInfo.add(p_proc);
					return SymbolStatus.Parameter;
				} else if( pName.equals("main") || pName.equals("MAIN__") ) {
					//We don't have to check a formal parameter of main function.
					if( symScope == null ) {
						symbolStatus = SymbolStatus.Parameter;
					} else {
						symbolStatus = SymbolStatus.Invisible;
					}
					symbolInfo.add(inSym);
					symbolInfo.add(p_proc);
					return symbolStatus;
				}else {
					//Find corresponding argument of this formal parameter.
					List<FunctionCall> funcCallList;
					if( fCallList == null ) {
						funcCallList = IRTools.getFunctionCalls(program);
					} else {
						funcCallList = fCallList;
					}
					// Find the caller procedure that called this procedure.
					List paramList = p_proc.getParameters();
					int list_size = paramList.size();
					if( list_size == 1 ) {
						Object obj = paramList.get(0);
						String paramS = obj.toString();
						// Remove any leading or trailing whitespace.
						paramS = paramS.trim();
						if( paramS.equals(Specifier.VOID.toString()) ) {
							list_size = 0;
						}
					}
					Procedure t_proc = null;
					Symbol currArgSym = null;
					Symbol prevArgSym = null;
					boolean foundArg = false;
					for( FunctionCall funcCall : funcCallList ) {
						if(p_proc.getName().equals(funcCall.getName())) {
							List argList = funcCall.getArguments();
							for( int i=0; i<list_size; i++ ) {
								////////////////////////////////////////////////////////////////////////
								// DEBUG: IRTools.containsSymbol() works only for searching           //
								// symbols in expression tree; it internally compare Identifier only, //
								// but VariableDeclarator contains NameID instead of Identifier.      //
								////////////////////////////////////////////////////////////////////////
								//if( IRTools.containsSymbol((Declaration)paramList.get(i), inSym)) 
								List declaredSyms = ((Declaration)paramList.get(i)).getDeclaredIDs();
								if( declaredSyms.contains(new Identifier(IRSym)) ) {
									// Found an actual argument for the inSym. 
									foundArg = true;
									t = funcCall.getStatement();
									while( (t != null) && !(t instanceof Procedure) ) {
										t = t.getParent();
									}
									t_proc = (Procedure)t;
									Expression exp = (Expression)argList.get(i);
									////////////////////////////////////////////////////////////////////
									// We are interested in only simple scalar argument (Identifier). //
									////////////////////////////////////////////////////////////////////
									if( exp instanceof Identifier ) {
										currArgSym = SymbolTools.getSymbolOf(exp);
									}
									if( currArgSym == null ) {
										symbolInfo.add(inSym);
										symbolInfo.add(p_proc);
										if( !(exp instanceof Literal) ) {
											PrintTools.println("\n[INFO in findSrcScalarSymbol()] argument (" + exp + 
													") passed for " + "the parameter symbol (" + IRSym + 
													") of a procedure, " + p_proc.getName() + ", has complex expression; " +
													"failed to analyze it.\n",1);
											return SymbolStatus.ComplexArgument;
										} else {
											if( symScope == null ) {
												symbolStatus = SymbolStatus.Parameter;
											} else {
												symbolStatus = SymbolStatus.Invisible;
											}
											return symbolStatus;
										}
									} else if( (prevArgSym != null) && (!currArgSym.equals(prevArgSym)) ) {
										Symbol pSym = getOrgSymbolOfExternOne(prevArgSym, program);
										Symbol cSym = getOrgSymbolOfExternOne(currArgSym, program);
										if( (pSym == null) || !pSym.equals(cSym) ) {
											// Multiple argument symbols are found.
											PrintTools.println("\n[INFO in findSrcScalarSymbol()] multiple argments exist " +
													"for the parameter symbol (" + IRSym + ") of procedure (" 
													+ p_proc.getSymbolName() + "); can't find the original symbol.\n", 1);
											symbolInfo.add(inSym);
											symbolInfo.add(p_proc);
											return SymbolStatus.MultipleArguments;
										}
									}
									prevArgSym = currArgSym;
								}
							}
						}
					} //end of funcCallList loop
					if( foundArg ) { //found a unique argument symbol for the current parameter symbol.
						symbolStatus = findSrcScalarSymbol(currArgSym, t_proc, noExtern, symScope, symbolInfo, funcCallList);
						if( symbolStatus == SymbolStatus.NotSupported ) {
							symbolInfo.add(inSym);
							symbolInfo.add(p_proc);
							return symbolStatus;
						} else {
							if( inSym instanceof PseudoSymbol ) {
								if( !symbolInfo.isEmpty() ) {
									Symbol newIRSym = (Symbol)symbolInfo.remove(0);
									inSym = replaceIRSymbol((PseudoSymbol)inSym, IRSym, newIRSym);
									symbolInfo.add(0, inSym);
								}
							}
							return symbolStatus;
						}
					} else { //not corresponding argument symbol is found (implementation bug)
						PrintTools.println("\n[WARNING in findSrcScalarSymbol()] Input symbol (" + IRSym + ") is a formal function parameter," +
								" but corresponding argument is not found in the current scope (enclosing procedure: "+ p_proc.getSymbolName() +
								").\n", 1); 
						symbolInfo.add(inSym);
						symbolInfo.add(p_proc);
						return SymbolStatus.UnknownError;
					}
				}
			} else {
				PrintTools.println("\n[WARNING in findSrcScalarSymbol()] Input symbol (" + IRSym + ") is a formal function parameter," +
						" but it is not visible in the current scope (enclosing procedure: "+ p_proc.getSymbolName() +
						").\n", 1); 
				symbolInfo.add(inSym);
				symbolInfo.add(p_proc);
				return SymbolStatus.Invisible;
			}
		} else if (SymbolTools.isLocal(IRSym) ) {
			Traversable tt = (Traversable)IRSym;
			while ( (tt != null) && !(tt instanceof Procedure) ){
				tt = tt.getParent();
			}
			if( (tt instanceof Procedure) && ((Procedure)tt).getSymbolName().equals(p_proc.getSymbolName()) ) {
				if( (symScope == null) || p_proc.getSymbolName().equals(symScope.getSymbolName()) ) {
					symbolStatus = SymbolStatus.Local;
				} else {
					symbolStatus = SymbolStatus.Invisible;
				}
			} else {
				PrintTools.println("\n[WARNING in findSrcScalarSymbol()] Input symbol (" + IRSym + ") is a local symbol," +
						" but it is not visible in the current scope (enclosing procedure: "+ p_proc.getSymbolName() +
						").\n", 1); 
				symbolStatus = SymbolStatus.Invisible;
			}
			symbolInfo.add(inSym);
			symbolInfo.add(p_proc);
			return symbolStatus;
		} else {
			PrintTools.println("\n[WARNING in findSrcScalarSymbol()] Can not determine the type of the input symbol (" + inSym + 
					" in the procedure, " + p_proc.getSymbolName() + ");" +
					" return an empty list.\n", 1); 
			symbolInfo.add(inSym);
			symbolInfo.add(p_proc);
			return SymbolStatus.UnknownError;
		}
	}
	
	/**
	 * Searches the symbol set and returns the first symbol whose name is the specified string
	 * @param sset		Symbol set being searched
	 * @param symName	symbol name being searched for
	 * @return the first symbol amaong the symbol set whose name is the same as the specified string
	 */
	public static Symbol findsSymbol(Set<Symbol> sset, String symName)
	{
		if ( sset == null )
			return null;
	
		for( Symbol sym : sset ) {
			if( sym.getSymbolName().equals(symName) ) {
				return sym;
			}
		}
		
		return null;
	}

	public static SubArray createSubArray(Symbol inSym, boolean addArraySizeInfo, Expression exp) {
		SubArray rSArray;
		if( inSym == null ) {
			rSArray = null;
		} else if( inSym instanceof PseudoSymbol ) {
			if( exp == null ) {
				//DEBUG: PseudoSymbol (AccessSymbol or DerefSymbol) does not have enough information
				//       needed to create a SubArray. Therefore, subarray is not created in current implementation.
				PrintTools.println("\n[WARNING in AnalysisTools.createSubArray()] in current implementation, pseudosymbol, " + inSym +
						", is not converted to a subarray.\n", 1);
				rSArray = null;
			} else if( (inSym instanceof AccessSymbol) && (exp instanceof AccessExpression) ){
				Symbol member = ((AccessSymbol)inSym).getMemberSymbol();
				if( member instanceof NestedDeclarator ) {
					if( !((NestedDeclarator)member).isProcedure() ) {
						rSArray = new SubArray(exp, -1);
					} else {
						rSArray = null;
					}
				} else {
					boolean isArray = SymbolTools.isArray(member);
					boolean isPointer = SymbolTools.isPointer(member);
					if( !isArray && !isPointer ) {
						//scalar, non-pointer variable
						rSArray = new SubArray(exp, 0);
					} else if( isArray && !isPointer ) { //array variable
						rSArray = new SubArray(exp, -1);
						if( addArraySizeInfo ) {
							List aspecs = member.getArraySpecifiers();
							ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
							int dimsize = aspec.getNumDimensions();
							ArrayList<Expression> startIndices = new ArrayList<Expression>(dimsize);
							ArrayList<Expression> lengths = new ArrayList<Expression>(dimsize);
							for( int i=0; i<dimsize; i++ ) {
								startIndices.add(i, new IntegerLiteral(0));
								Expression length = aspec.getDimension(i);
								if( length == null ) {
									lengths.add(i, null);
								} else {
									lengths.add(i, length.clone());
								}
							}
							rSArray.setRange(startIndices, lengths);
						}
					} else {//pointer or pointer array
						rSArray = new SubArray(exp, -1);
					}
				}
			} else {
				PrintTools.println("\n[WARNING in AnalysisTools.createSubArray()] in current implementation, pseudosymbol, " + inSym +
						", is not converted to a subarray.\n", 1);
				rSArray = null;
			}
		} else if( inSym instanceof VariableDeclarator ) {
			VariableDeclarator vDeclr = (VariableDeclarator)inSym;
			Expression aName = new Identifier(inSym);
			boolean isArray = SymbolTools.isArray(inSym);
			boolean isPointer = SymbolTools.isPointer(inSym);
			if( !isArray && !isPointer ) {
				//scalar, non-pointer variable
				rSArray = new SubArray(aName, 0);
			} else if( isArray && !isPointer ) { //array variable
				rSArray = new SubArray(aName, -1);
				if( addArraySizeInfo ) {
					List aspecs = inSym.getArraySpecifiers();
					ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
					int dimsize = aspec.getNumDimensions();
					ArrayList<Expression> startIndices = new ArrayList<Expression>(dimsize);
					ArrayList<Expression> lengths = new ArrayList<Expression>(dimsize);
					for( int i=0; i<dimsize; i++ ) {
						startIndices.add(i, new IntegerLiteral(0));
						Expression length = aspec.getDimension(i);
						if( length == null ) {
							lengths.add(i, null);
						} else {
							lengths.add(i, length.clone());
						}
					}
					rSArray.setRange(startIndices, lengths);
				}
			} else {//pointer or pointer array
				rSArray = new SubArray(aName, -1);
			}
		} else if( inSym instanceof NestedDeclarator ) {
			NestedDeclarator nestSym = (NestedDeclarator)inSym;
			if( !nestSym.isProcedure() ) {
				rSArray = new SubArray(new Identifier(nestSym), -1);
			} else {
				rSArray = null;
			}
		} else {
			//Non-variable symbol
			rSArray = null;
		}
		return rSArray;
	}
	
	
	/**
	 * Convert ArrayAccess into SubArray.
	 * 
	 * @param aa
	 * @return subarray
	 */
	public static SubArray arrayAccessToSubArray(ArrayAccess aa) {
		SubArray subArr = null;
		if( aa != null ) {
			subArr = new SubArray(aa.getArrayName().clone());
			List<Expression> startList = new LinkedList<Expression>();
			List<Expression> lengthList = new LinkedList<Expression>();
			int dimsize = aa.getNumIndices();
			for(int i=0; i<dimsize; i++) {
				startList.add(aa.getIndex(i).clone());
				lengthList.add(new IntegerLiteral(1));
			}
			subArr.setRange(startList, lengthList);
		}
		return subArr;
	}

	/**
	 * Convert ArrayAccess into SubArray.
	 * This method differs from arrayAccessToSubArray() in that both ArrayAccess
	 * and SubArray share the same internal expressions; if expressions in the 
	 * ArrayAccess is changed, SubArray is also changed accordingly.
	 * FIXME: if index expression is an identifier, change in ArrayAccess will not be
	 * reflected in the SubArray.
	 * 
	 * @param aa
	 * @return subarray
	 */
	public static SubArray arrayAccessToSubArray2(ArrayAccess aa) {
		SubArray subArr = null;
		if( aa != null ) {
			subArr = new SubArray(aa.getArrayName());
			List<Expression> startList = new LinkedList<Expression>();
			List<Expression> lengthList = new LinkedList<Expression>();
			int dimsize = aa.getNumIndices();
			for(int i=0; i<dimsize; i++) {
				startList.add(aa.getIndex(i));
				lengthList.add(new IntegerLiteral(1));
			}
			subArr.setRange(startList, lengthList);
		}
		return subArr;
	}

	public static boolean isCudaCall(FunctionCall fCall) {
		if ( fCall == null )
			return false;

		Set<String> cudaCalls = new HashSet<String>(Arrays.asList(
				"CUDA_SAFE_CALL","cudaFree","cudaMalloc","cudaMemcpy",
				"cudaMallocPitch","tex1Dfetch","cudaBindTexture", "cudaMemcpy2D"
		));

		if ( cudaCalls.contains((fCall.getName()).toString()) ) {
			return true;
		}
		return false;
	}
	
	/**
	 * Return a statement before the ref_stmt in the parent CompoundStatement
	 * 
	 * @param parent parent CompoundStatement containing the ref_stmt as a child
	 * @param ref_stmt
	 * @return statement before the ref_stmt in the parent CompoundStatement. If ref_stmt 
	 * is not a child of parent or if there is no previous statement, return null.
	 */
	public static Statement getStatementBefore(CompoundStatement parent, Statement ref_stmt) {
		List<Traversable> children = parent.getChildren();
		int index = Tools.indexByReference(children, ref_stmt);
		if( index <= 0 ) {
			return null;
		}
		return (Statement)children.get(index-1);
	}
	
	/**
	 * Return a statement after the ref_stmt in the parent CompoundStatement
	 * 
	 * @param parent parent CompoundStatement containing the ref_stmt as a child
	 * @param ref_stmt
	 * @return statement after the ref_stmt in the parent CompoundStatement. If ref_stmt 
	 * is not a child of parent or if there is no previous statement, return null.
	 */
	public static Statement getStatementAfter(CompoundStatement parent, Statement ref_stmt) {
		List<Traversable> children = parent.getChildren();
		int index = Tools.indexByReference(children, ref_stmt);
		if( (index == -1) || (index == children.size()-1) ) {
			return null;
		}
		return (Statement)children.get(index+1);
	}
	
	public static boolean extractDimensionInfo(SubArray sArray, List<Expression> startList, 
			List<Expression> lengthList, boolean checkIRSymDimensions, Annotatable tr) {
		Annotatable at = tr;
		boolean noError = true;
		if( sArray == null )  return false;
		Expression aName = sArray.getArrayName();
		int dimension = sArray.getArrayDimension();
		if( (aName instanceof AccessExpression) && checkIRSymDimensions )  {
			//If aName is access expression, and IRSymbolOnly is true, dimension information in the SubArray
			//is not usable, since it is about member expression of the access expression.
			noError = false;
		} else {
			if( dimension == 0 ) {
				noError = true;
				//startList.add(new IntegerLiteral(0));
				//lengthList.add(new IntegerLiteral(1));
			} else if( dimension > 0 ) {
				for( int i=0; i<dimension; i++ ) {
					List<Expression> range = sArray.getRange(i);
					if( range.get(0) == null ) {//Start index can be missing; default is 0
						startList.add(new IntegerLiteral(0));
					} else {
						startList.add(Symbolic.simplify(range.get(0).clone()));
					}
					if( range.get(1) == null ) { //Lenght should not be empty.
						noError = false;
						lengthList.add(null);
					} else {
						lengthList.add(Symbolic.simplify(range.get(1).clone()));
					}
				}
			}
		}
		if( (dimension == -1) || !noError ) { //dimension is not known.
			Symbol sym = SymbolTools.getSymbolOf(aName);
			if( checkIRSymDimensions ) { //Check dimensions of IR symbol
				if( sym instanceof PseudoSymbol ) {
					sym = ((PseudoSymbol)sym).getIRSymbol();
				}
			} else { //Check dimensions of member symbol if PseudoSymbol.
				while( sym instanceof AccessSymbol ) {
					sym = ((AccessSymbol)sym).getMemberSymbol();
				}
			}
			startList.clear();
			lengthList.clear();
			boolean isArray = SymbolTools.isArray(sym);
			boolean isPointer = SymbolTools.isPointer(sym);
			if( sym instanceof NestedDeclarator ) {
				isPointer = true;
			}
			noError = true;
			if( isArray  ) { //array or pointe-to-array variable
				List aspecs = sym.getArraySpecifiers();
				ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
				int dimsize = aspec.getNumDimensions();
				for( int i=0; i<dimsize; i++ ) {
					startList.add(i, new IntegerLiteral(0));
					Expression length = aspec.getDimension(i);
					if( length == null ) {
						lengthList.add(i, null);
						noError = false;
					} else {
						lengthList.add(i, Symbolic.simplify(length.clone()));
					}
				}
				if( isPointer ) {
					startList.add(0, new IntegerLiteral(0));
					lengthList.add(0, null);
					noError = false;
				}
			} else if( !isArray && !isPointer ) { //scalar, non-pointer variable
				noError = true;
				//startList.add(new IntegerLiteral(0));
				//lengthList.add(new IntegerLiteral(1));
			} else { //pointer variable; no way to find dimension info.
				startList.add(0, new IntegerLiteral(0));
				lengthList.add(0, null);
				noError = false;
			}
			if( noError ) {
				//Update SubArray with dimension information.
				List<Expression> tstartList = new LinkedList<Expression>();
				for( Expression ttExp : startList ) {
					tstartList.add(ttExp.clone());
				}
				List<Expression> tlengthList = new LinkedList<Expression>();
				for( Expression ttExp : lengthList ) {
					tlengthList.add(ttExp.clone());
				}
				sArray.setRange(tstartList, tlengthList);
			} else {
				//[FIXME] below pass assumes that the same variable always accesses the same array range. 
				List<Expression> tstartList = new LinkedList<Expression>();
				List<Expression> tlengthList = new LinkedList<Expression>();
				//System.err.println("Check whether enclosing data region contains dimension info of " + sym.getSymbolName());
				//Check whether any enclosing data region contains dimension information.
				while ((at != null) ) {
					ACCAnnotation ttAnnot = at.getAnnotation(ACCAnnotation.class, "data");
					if( ttAnnot != null ) {
						SubArray ttSubArray = findSubArrayInDataClauses(ttAnnot, sym, checkIRSymDimensions);
						if( (ttSubArray != null) && (ttSubArray.getArrayDimension() > 0) ) {
							tstartList.clear();
							tlengthList.clear();
							noError =  true;
							List<Expression> ttList = ttSubArray.getStartIndices();
							for( Expression ttExp : ttList ) {
								if( ttExp == null ) {
									noError = false;
									break;
								} else {
									tstartList.add(ttExp.clone());
								}
							}
							if( noError ) {
								ttList = ttSubArray.getLengths();
								for( Expression ttExp : ttList ) {
									if( ttExp == null ) {
										noError = false;
										break;
									} else {
										tlengthList.add(ttExp.clone());
									}
								}
							}
							if( noError ) {
								break;
							}
						}
					}
					if( at instanceof Procedure ) {
						break;
					} else {
						at = (Annotatable)at.getParent();
					}
				}
				if( !noError ) {
					tstartList.clear();
					tlengthList.clear();
					boolean isGlobal = SymbolTools.isGlobal(sym);
					if( !isGlobal ) {
						List symbolInfo = new ArrayList(2);
						if( AnalysisTools.SymbolStatus.OrgSymbolFound(
								AnalysisTools.findOrgSymbol(sym, at, true, null, symbolInfo, null)) ) {
							sym = (Symbol)symbolInfo.get(0);
							isGlobal = SymbolTools.isGlobal(sym);
						}
					}
					//Check data regions inerprocedurally if the variable is a global symbol.
					if( (at instanceof Procedure) && isGlobal ) {
						//System.err.println("Check whether any data region contains dimension info of the global symbol, " + sym.getSymbolName());
						Traversable prog = at;
						while ( prog !=null ) {
							if( prog instanceof Program ) {
								break;
							} else {
								prog = prog.getParent();
							}
						}
						Set<String> dataDirectives = new HashSet<String>(Arrays.asList("data", "declare"));
						List<ACCAnnotation> dataAnnots = AnalysisTools.collectPragmas(prog, ACCAnnotation.class, dataDirectives, false);
						if( dataAnnots != null ) {
							for( ACCAnnotation dAnnot : dataAnnots ) {
								//System.err.println("current data region\n" + dAnnot);
								SubArray ttSubArray = findSubArrayInDataClauses(dAnnot, sym, checkIRSymDimensions);
								if( (ttSubArray != null) && (ttSubArray.getArrayDimension() > 0) ) {
									//System.err.println("Found data region containing " + sym.getSymbolName());
									tstartList.clear();
									tlengthList.clear();
									noError =  true;
									List<Expression> ttList = ttSubArray.getStartIndices();
									for( Expression ttExp : ttList ) {
										if( ttExp == null ) {
											tstartList.clear();
											tlengthList.clear();
											noError = false;
											break;
										} else {
											tstartList.add(ttExp.clone());
										}
									}
									if( noError ) {
										ttList = ttSubArray.getLengths();
										for( Expression ttExp : ttList ) {
											if( ttExp == null ) {
												tstartList.clear();
												tlengthList.clear();
												noError = false;
												break;
											} else {
												tlengthList.add(ttExp.clone());
											}
										}
									}
									if( noError ) {
										ttList = tstartList;
										for( Expression ttExp : ttList ) {
											Set<Symbol> symSet = SymbolTools.getAccessedSymbols(ttExp);
											for( Symbol tSym : symSet ) {
												if( !SymbolTools.isGlobal(tSym) ) {
													noError = false;
													break;
												}
											}
											if ( !noError ) {
												break;
											}
										}
										if( noError ) {
											ttList = tlengthList;
											for( Expression ttExp : ttList ) {
												Set<Symbol> symSet = SymbolTools.getAccessedSymbols(ttExp);
												for( Symbol tSym : symSet ) {
													if( !SymbolTools.isGlobal(tSym) ) {
														noError = false;
														break;
													}
												}
												if( !noError ) {
													break;
												}
											}
										}
										if( noError ) {
											break;
										}
									}
								}
							}
						}
					}
				}
				if( noError ) {
					startList.clear();
					lengthList.clear();
					for( Expression ttExp : tstartList ) {
						startList.add(ttExp.clone());
					}
					for( Expression ttExp : tlengthList ) {
						lengthList.add(ttExp.clone());
					}
					sArray.setRange(tstartList, tlengthList);
				}
			}
		}
		return noError;
	}
	
	public static boolean extractDimensionInfo(Symbol sym,
			List<Expression> lengthList, boolean checkIRSymDimensions) {
		boolean noError = true;
		if( checkIRSymDimensions ) { //Check dimensions of IR symbol
			if( sym instanceof PseudoSymbol ) {
				sym = ((PseudoSymbol)sym).getIRSymbol();
			}
		} else { //Check dimensions of member symbol if PseudoSymbol.
			while( sym instanceof AccessSymbol ) {
				sym = ((AccessSymbol)sym).getMemberSymbol();
			}
		}
		boolean isArray = SymbolTools.isArray(sym);
		boolean isPointer = SymbolTools.isPointer(sym);
		if( sym instanceof NestedDeclarator ) {
			isPointer = true;
		}
		noError = true;
		if( isArray  ) { //array or pointe-to-array variable
			List aspecs = sym.getArraySpecifiers();
			ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
			int dimsize = aspec.getNumDimensions();
			for( int i=0; i<dimsize; i++ ) {
				Expression length = aspec.getDimension(i);
				if( length == null ) {
					lengthList.add(i, null);
					noError = false;
				} else {
					lengthList.add(i, Symbolic.simplify(length.clone()));
				}
			}
			if( isPointer ) {
				lengthList.add(0, null);
				noError = false;
			}
		} else if( !isArray && !isPointer ) { //scalar, non-pointer variable
			noError = true;
		} else { //pointer variable; no way to find dimension info.
			lengthList.add(0, null);
			noError = false;
		}
		return noError;
	}
	
	/**
	 * Convert general set into a sorted set; this is mainly for consistent code generation, since
	 * HashSet element order can be changed randomly.
	 * 
	 * @param inSet
	 * @return sorted set
	 */
	public static Collection getSortedCollection(Set inSet) {
		    TreeMap<String, Object> sortedMap = new TreeMap<String, Object>();
			for( Object obj : inSet ) {
				String objString;
				if( obj instanceof Procedure ) {
					objString = ((Procedure)obj).getSymbolName();
				} else {
					objString = obj.toString();
				}
				sortedMap.put(objString, obj);
			}
			return sortedMap.values();
	}
	
	/**
	 * Returns true if the symbol set contains a symbol whose name is the specified string.
	 * @param sset		Symbol set being searched
	 * @param symName	symbol name being searched for
	 */
	public static boolean containsSymbol(Set<Symbol> sset, String symName)
	{
		if ( sset == null )
			return false;
	
		for( Symbol sym : sset ) {
			if( sym.getSymbolName().equals(symName) ) {
				return true;
			}
		}
		
		return false;
	}
	
	/**
	 * Find a symbol table containing the input IR symbol, {@code sym}, searching 
	 * from the input traversable, {@code tt}.
	 * 
	 * @param sym input symbol
	 * @param tt traversable from which search starts
	 * @return symbol table containing the symbol, {@code sym}; return null if not found.
	 */
	public static SymbolTable getIRSymbolScope(Symbol sym, Traversable tt) {
		Symbol IRSym = sym;
		if( sym instanceof PseudoSymbol ) {
			IRSym = ((PseudoSymbol)sym).getIRSymbol();
		}
		SymbolTable  targetSymbolTable = null;
		while( tt != null ) {
			if( tt instanceof SymbolTable ) {
				if( ((SymbolTable)tt).containsSymbol(IRSym) ) {
					break;
				}
			}
			tt = tt.getParent();
		}
		if( (tt != null) && (tt instanceof SymbolTable) && ((SymbolTable)tt).containsSymbol(IRSym) ) {
			targetSymbolTable = (SymbolTable)tt;
		}
		return targetSymbolTable;
	}

	/**
	 * Find a set of reduction symbols and store them in the internal annotation if missing.
	 * 
	 * @param at
	 * @param cAnnot
	 * @return
	 */
	public static Set<Symbol> getReductionSymbols(Annotatable at, boolean IRSymbolOnly) {
		ACCAnnotation cAnnot = at.getAnnotation(ACCAnnotation.class, "reduction");
		Annotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
		if( iAnnot == null ) {
			iAnnot = new ACCAnnotation("internal", "_directive");
			iAnnot.setSkipPrint(true);
			at.annotate(iAnnot);
		}
		Set<Symbol> accReductionSymbols = iAnnot.get("accreduction");
		if( accReductionSymbols == null ) {
			accReductionSymbols = new HashSet<Symbol>();
			try { 
				Map valMap = (Map)at.getAnnotation(ACCAnnotation.class, "reduction");
				for( ReductionOperator op : (Set<ReductionOperator>)valMap.keySet() ) {
					Set<SubArray> valSet = (Set<SubArray>)valMap.get(op); 
					Set<Symbol> symDSet = null;
					symDSet = subarraysToSymbols(valSet, IRSymbolOnly);
					if( valSet.size() != symDSet.size() ) {
						Tools.exit("[ERROR in TransformationTools.getReductionSymbols()]: cannot find symbols for " +
								"subarrays of key," + " reduction" + ", in ACCAnnotation, " + cAnnot + "\n");
					} else {
						accReductionSymbols.addAll(symDSet);
					}
				}
			} catch( Exception e ) {
				Tools.exit("[ERROR in TransformationTools.getReductionSymbols()]: <ReductionOperator, Set<SubArray>> type " +
						"is expected for the value of key," + " reduction" + " in ACCAnnotation, " + cAnnot + "\n"
						+ "Exception message: " + e + "\n");
			}
			iAnnot.put("accreduction", accReductionSymbols);
		}
		Set<Symbol> accRedSymCopySet = new HashSet<Symbol>();
		if( accReductionSymbols != null ) {
			accRedSymCopySet.addAll(accReductionSymbols);
		}
		return accRedSymCopySet;
	}
	
	/**
	 * Check if the input symbol, sym, is passed by reference in the function call, fCall.
	 * 
	 * @param sym input symbol to check
	 * @param fCall function call of interest
	 * @return -2 if procedure definition of the function call can not be found
	 *         or -1 if the input symbol is not passed by reference
	 *         or -3 if an error occurs 
	 *         or index of the first argument where the input symbol is passed by reference
	 */
	public static int isCalledByRef(Symbol sym, FunctionCall fCall) {
		int status = -1;
		Procedure calledProc = fCall.getProcedure();
		if( calledProc == null ) {
			status = -2; //Can't find the procedure for the function call.
		} else {
			boolean isPointerTypeArg = false;
			if( SymbolTools.isArray(sym) || SymbolTools.isPointer(sym) ) {
				isPointerTypeArg = true;
			}
			List argList = fCall.getArguments();
			List paramList = calledProc.getParameters();
			int list_size = argList.size();
			for( int i=0; i<list_size; i++ ) {
				Object arg = argList.get(i);
				Symbol paramSym = (Symbol)((VariableDeclaration)paramList.get(i)).getDeclarator(0);
				boolean isPointerTypeParam = false;
				if( SymbolTools.isArray(paramSym) || SymbolTools.isPointer(paramSym) ) {
					isPointerTypeParam = true;
				}
				if( arg instanceof Traversable ) {
					Set<Symbol> usedSyms = DataFlowTools.getUseSymbol((Traversable)arg);
					if( isPointerTypeParam && usedSyms.contains(sym) ) {
						if( isPointerTypeArg ) {
							status = i;
							break;
						} else if ( arg instanceof UnaryExpression ) {
							UnaryExpression uexp = (UnaryExpression)arg;
							if( uexp.getOperator().equals(UnaryOperator.ADDRESS_OF) ) {
								status = i;
								break;
							}
						}
					}
				} else {
					status = -3;
					break;
				}
			}
		}
		return status;
	}
	
	/**
	 * Interprocedurally check whether input symbol, sym, is defined in the Traversable, t.
	 * 
	 * @param sym input symbol
	 * @param t traversable to check
	 * @return true if the input symbol is defined in the traversable.
	 */
	public static boolean ipaIsDefined(Symbol sym, Traversable t, Set<Procedure> accessedProcs ) {
		if( accessedProcs == null ) {
			accessedProcs = new HashSet<Procedure>();
		}
		boolean isDefined = false;
		Set<Symbol> defSet = DataFlowTools.getDefSymbol(t);
		if( defSet.contains(sym)) {
			isDefined = true;
		} else {
			List<FunctionCall> fCallList = IRTools.getFunctionCalls(t);
			for( FunctionCall fCall : fCallList ) {
				if( StandardLibrary.contains(fCall) ) {
					if( StandardLibrary.isSideEffectFree(fCall) ) {
						continue;
					} else {
						Set<Symbol> usedSyms = DataFlowTools.getUseSymbol(fCall);
						if( usedSyms.contains(sym) ) {
							isDefined = true;
							break;
						}
					}
				} else {
					Procedure proc = fCall.getProcedure();
					if( (proc != null) && !accessedProcs.contains(proc) ) {
						accessedProcs.add(proc);
						int index = isCalledByRef(sym, fCall);
						if( index >= 0 ) {
							Symbol paramSym = (Symbol)((VariableDeclaration)proc.getParameter(index)).getDeclarator(0);
							isDefined = ipaIsDefined(paramSym, proc.getBody(), accessedProcs);
						} else if ( SymbolTools.isGlobal(sym) ) {
							isDefined = ipaIsDefined(sym, proc.getBody(), accessedProcs);
						}
						if( isDefined ) {
							break;
						}
					}
				} 
				
			}
		}
		return isDefined;
	}
	
	/**
	 * For each kernel region, which will be transformed into a GPU kernel,
	 * 1) add information of the enclosing procedure name and kernel ID.
	 *   The annotation has the following form:
	 * 	 #pragma acc ainfo procname(procedure-name) kernelid(kernel-id)
	 * 2) apply user directives if existing.
	 * 
	 * @param program input program
	 */
	public static void annotateUserDirectives(Program program, 
			HashMap<String, HashMap<String, Object>> userDirectives) {
		boolean userDirectiveExists = false;
		/* iterate to search for all Procedures */
		List<Procedure> procedureList = IRTools.getProcedureList(program);
		for( Procedure cProc : procedureList ) {
			String procName = cProc.getSymbolName().toString();
			List<ACCAnnotation> compAnnotsOrg = AnalysisTools.collectPragmas(cProc, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
			if( compAnnotsOrg != null ) {
				List<ACCAnnotation> parallelRegionAnnots = new LinkedList<ACCAnnotation>();
				List<ACCAnnotation> kernelsRegionAnnots = new LinkedList<ACCAnnotation>();
				for( ACCAnnotation cAnnot : compAnnotsOrg ) {
					if( cAnnot.containsKey("kernels") ) {
						kernelsRegionAnnots.add(cAnnot);
					} else {
						parallelRegionAnnots.add(cAnnot);
					}
				}
				List<ACCAnnotation> compAnnots = new LinkedList<ACCAnnotation>();
				compAnnots.addAll(parallelRegionAnnots);
				compAnnots.addAll(kernelsRegionAnnots);
				int cnt = 0;
				for( ACCAnnotation cAnnot : compAnnots ) {
					Annotatable at = cAnnot.getAnnotatable();
					//String refName = procName + "_kernel" + cnt++;	
					ARCAnnotation aInfo = at.getAnnotation(ARCAnnotation.class, "ainfo");
					if( aInfo == null ) {
						aInfo = new ARCAnnotation("ainfo", "_directive");
						at.annotate(aInfo);
					}
					aInfo.put("procname", procName);
					aInfo.put("kernelid", new IntegerLiteral(cnt));
					String kernelName = procName + "_kernel" + cnt++;	
					if( !userDirectives.isEmpty() ) {
						userDirectiveExists = true;
						Set<String> kernelSet = userDirectives.keySet();
						if( kernelSet.contains(kernelName) ) {
							HashMap<String, Object> directives = userDirectives.remove(kernelName);
							for( String clause : directives.keySet() ) {
								Object uObj = directives.get(clause);
								PragmaAnnotation uAnnot;
								if( ARCAnnotation.cudaClauses.contains(clause) ) {
									uAnnot = at.getAnnotation(ARCAnnotation.class, "cuda");
									if( uAnnot == null ) {
										uAnnot = new ARCAnnotation("cuda", "_directive");
										at.annotate(uAnnot);
										uAnnot.put(clause, uObj);
									} else {
										Object fObj = uAnnot.get(clause);
										if( fObj instanceof Set ) {
											((Set)fObj).addAll((Set)uObj);
										} else if( fObj instanceof List ) {
											((List)fObj).addAll((List)uObj);
										} else {
											uAnnot.put(clause, uObj);
										}
									}
								} else {
									uAnnot =  at.getAnnotation(ACCAnnotation.class, clause);
									if( uAnnot == null ) {
										uAnnot = cAnnot;
										uAnnot.put(clause, uObj);
									} else {
										Object fObj = uAnnot.get(clause);
										if( fObj instanceof Set ) {
											((Set)fObj).addAll((Set)uObj);
										} else if( fObj instanceof List ) {
											((List)fObj).addAll((List)uObj);
										} else {
											uAnnot.put(clause, uObj);
										}
									}
								}
								//////////////////////////////////////////////////////////////////
								// Due to cloning issues, user directives are applied after     //
								// interprocedural CPU-GPU memory transfor optimizations.       // 
								// Because user directives have more priority to those inserted //
								// by compiler analyses, existing clauses may need to be updated//
								// according to the user directives.                            //
								//////////////////////////////////////////////////////////////////
							}
						}
					}
				}
				if( userDirectiveExists ) {
					// Replace NameIDs in the ACCAnnotation clauses with Identifiers.
					ACCAnalysis.updateSymbolsInACCAnnotations(program, null);
					if( !userDirectives.isEmpty() ) {
						Set<String> kernelSet = userDirectives.keySet();
						PrintTools.println("\n[WARNING in annotateUserDirectives()] user directives for the following" +
								" set of kernels can not be applicable: " + PrintTools.collectionToString(kernelSet, ",") + "\n", 0);
					}
				}
			}
		}
	}
	
	/**
	 * Insert barriers before and after each compute region, so that other analysis can
	 * easily distinguish GPU compute regions from CPU regions. 
	 */
	public static void markIntervalForComputeRegions(Program program) {
		/* iterate to search for all Procedures */
		List<Procedure> procedureList = IRTools.getProcedureList(program);
		CompoundStatement target_parent;
		for (Procedure proc : procedureList)
		{
			/* Search for all compute regions in a given Procedure */
			List<ACCAnnotation> compAnnots = AnalysisTools.collectPragmas(proc, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
			if( compAnnots != null ) {
				for ( ACCAnnotation annot : compAnnots )
				{
					Statement target_stmt = (Statement)annot.getAnnotatable();
					target_parent = (CompoundStatement)target_stmt.getParent();
					ACCAnnotation barrierAnnot = new ACCAnnotation("barrier", "S2P");
					Statement bAStmt = new AnnotationStatement(barrierAnnot);
					target_parent.addStatementBefore(target_stmt, bAStmt);
					barrierAnnot = new ACCAnnotation("barrier", "P2S");
					bAStmt = new AnnotationStatement(barrierAnnot);
					target_parent.addStatementAfter(target_stmt, bAStmt);
				}
			}
		}
	}
	
	static public void deleteBarriers( Traversable t ) {
		List<ACCAnnotation> barrList = new LinkedList<ACCAnnotation>();
        DFIterator<AnnotationStatement> iter =
                new DFIterator<AnnotationStatement>(t, AnnotationStatement.class);
        while (iter.hasNext()) {
            AnnotationStatement at = iter.next();
			ACCAnnotation br_annot = at.getAnnotation(ACCAnnotation.class, "barrier");
			if ( br_annot != null ) {
				barrList.add(br_annot);
			}
		}
		for( ACCAnnotation o_annot : barrList ) {
			Statement astmt = (Statement)o_annot.getAnnotatable();
			if( astmt != null ) {
				Traversable parent = astmt.getParent();
				if( parent != null )
					parent.removeChild(astmt);
				else
					PrintTools.println("[Error in deleteBarriers()] parent is null!", 0);
			}
		}
	}
	
	/**
	 * Return a set of all OpenACC shared variables.
	 * (This method assumes that accshared internal clauses exist for each compute region.)
	 * 
	 * @param program
	 * @return a set of all OpenACC shared symbols (this will return null if any compute region does not
	 * have accshared internal clause.)
	 */
	public static Set<Symbol> getAllACCSharedVariables(Program program) {
		Set<Symbol> ret = new HashSet<Symbol>();
		List<FunctionCall> fCallList = IRTools.getFunctionCalls(program);
		/* iterate to search for all Procedures */
		List<Procedure> procedureList = IRTools.getProcedureList(program);
		CompoundStatement target_parent;
		for (Procedure proc : procedureList)
		{
			/* Search for all compute regions in a given Procedure */
			List<ACCAnnotation> compAnnots = AnalysisTools.collectPragmas(proc, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
			if( compAnnots != null ) {
				for ( ACCAnnotation annot : compAnnots )
				{
					Annotatable at = annot.getAnnotatable();
					ACCAnnotation sAt = at.getAnnotation(ACCAnnotation.class, "accshared");
					if( sAt == null ) {
						return null;
					} else {
						Set<Symbol> shared_set = sAt.get("accshared");
						for( Symbol tSym : shared_set ) {
							List symbolInfo = new ArrayList(2);
							if( AnalysisTools.SymbolStatus.OrgSymbolFound(
									AnalysisTools.findOrgSymbol(tSym, at, true, null, symbolInfo, fCallList)) ) {
								Symbol tOSym = (Symbol)symbolInfo.get(0);
								ret.add(tOSym);
							}
						}
					}
				}
			}
		}
		return ret;
	}
	
	public static boolean isInHeaderFile(Declaration decl, TranslationUnit trUnt) {
		boolean isInHeader = false;
		if( trUnt != null ) {
			Traversable tt = decl.getParent();
			while ( (tt != null) && !(tt instanceof TranslationUnit) ) {
				tt = tt.getParent();
			}
			if( (tt == null) || (tt != trUnt) ) {
				Tools.exit("[ERROR in AnalysisTools.isInHeaderFile()] input declaration does not belong to the input " +
						"file:\nInput file: " + trUnt.getInputFilename() + "\nInput declaration: \n" + decl + "\n");
			}
			Declaration firstdecl = trUnt.getFirstDeclaration();
			if( decl != firstdecl ) {
				boolean foundFirstDecl = false;
				List<Traversable> children = trUnt.getChildren();
				for( Traversable child : children ) {
					if( !foundFirstDecl ) {
						if( child == decl ) {
							isInHeader = true;
							break;
						} else if( child == firstdecl ) {
							foundFirstDecl = true;
							break;
						}
					}
				}
			}
			
		}
		return isInHeader;
	}
	
	public static Procedure findProcedureForFormalSymol(Traversable tr, Symbol inSym) {
		Procedure proc = null;
		if( SymbolTools.isFormal(inSym) ) {
			while( (tr !=null) && !(tr instanceof  Program) ) {
				tr = tr.getParent();
			}
			if( tr instanceof Program ) {
				List<Procedure> procList = IRTools.getProcedureList((Program)tr);
				if( procList != null ) {
					String procName = null;
					Traversable t = inSym.getDeclaration();
					if( t != null ) {
						t = t.getParent();
						if( t instanceof ProcedureDeclarator ) {
							procName = ((ProcedureDeclarator)t).getSymbolName();
							for( Procedure tProc : procList ) {
								String name = tProc.getName().toString();
								if ( name.equals(procName) ) {
									proc = tProc;
									break;
								}
							}
						}
					}
				}
			}
		}
		return proc;
	}
	
	public static Procedure findMainEntryFunction(Program prog, String mainEntryFunc) {
		List<Procedure> procList = IRTools.getProcedureList(prog);
		Procedure main = null;
		if( procList != null ) {
			for( Procedure tProc : procList ) {
				String name = tProc.getName().toString();

				/* f2c code uses MAIN__ */
				if ( ((mainEntryFunc != null) && name.equals(mainEntryFunc)) || 
						((mainEntryFunc == null) && (name.equals("main") || name.equals("MAIN__"))) ) {
					main = tProc;
					break;
				}

			}
		}
		return main;
	}
	
	/**
	 * Find the first procedure in the input file ({@code trUnt}).
	 * 
	 * @param trUnt input Translation Unit (input file)
	 * @return the first procedure in the input file {@code trUnt}
	 */
	public static Procedure findFirstProcedure(TranslationUnit trUnt) {
		Procedure firstProc = null;
		if( trUnt != null ) {
			Declaration firstdecl = trUnt.getFirstDeclaration();
			if( firstdecl instanceof Procedure ) {
				firstProc = (Procedure)firstdecl;
			} else {
				boolean foundFirstDecl = false;
				List<Traversable> children = trUnt.getChildren();
				for( Traversable child : children ) {
					if( foundFirstDecl ) {
						if( child instanceof Procedure ) {
							firstProc = (Procedure)child;
							break;
						}
					} else if( child == firstdecl ) {
						foundFirstDecl = true;
					}
				}
			}
		}
		return firstProc;
	}
	
	/**
	 *  Return the list of procedure declarations belonging to the translation
	 *  unit that the input tr belongs to. If the input tr is Program, this will
	 *  return all procedure declarations in the whole program.
	 * @param tr
	 * @return
	 */
	public static List<ProcedureDeclarator> getProcedureDeclarators(Traversable tr) {
		List<ProcedureDeclarator> retList = new ArrayList<ProcedureDeclarator>();
		while( (tr != null) && !(tr instanceof TranslationUnit) && 
				!(tr instanceof Program) ) {
			tr = tr.getParent();
		}
		if( tr != null ) {
			List<TranslationUnit> trUnts = new ArrayList<TranslationUnit>();
			if( tr instanceof TranslationUnit ) {
				trUnts.add((TranslationUnit)tr);
			} else if( tr instanceof Program ) {
				for( Traversable tt : ((Program)tr).getChildren() ) {
					trUnts.add((TranslationUnit)tt);
				}
			}
			for( TranslationUnit trU : trUnts ) {
				DFIterator<ProcedureDeclarator> iter = new DFIterator<ProcedureDeclarator>(trU, ProcedureDeclarator.class);
				iter.pruneOn(ProcedureDeclarator.class);
				iter.pruneOn(Procedure.class);
				iter.pruneOn(Statement.class);
				for (;;)
				{
					ProcedureDeclarator procDeclr = null;

					try {
						procDeclr = (ProcedureDeclarator)iter.next();
						retList.add(procDeclr);
					} catch (NoSuchElementException e) {
						break;
					}
				}
			}
		}
		return retList;
	}
	
	public static VariableDeclaration getProcedureDeclaration(Traversable inTr, IDExpression fCallName) {
		VariableDeclaration procDecl = null;
		Traversable tt = inTr;
		while( (tt != null) && !(tt instanceof TranslationUnit) && !(tt instanceof Program)) {
			tt = tt.getParent();
		}
		List<TranslationUnit> trUntList = new ArrayList<TranslationUnit>();
		if( tt instanceof TranslationUnit ) {
			trUntList.add((TranslationUnit)tt);
		} else if( tt instanceof Program ) {
			for(Traversable ttt : tt.getChildren() ) {
				trUntList.add((TranslationUnit)ttt);
			}
		}
		for( TranslationUnit cTu : trUntList )
		{
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
				if( procDeclr.getID().equals(fCallName) ) {
					Traversable parent = procDeclr.getParent();
					if( parent instanceof VariableDeclaration ) {
						//Found function declaration.
						procDecl = (VariableDeclaration)parent;
						break;
					}
				}
			}
		}
		return procDecl;
	}
	
	public static int getBitLengthOfType(List<Specifier> typespecs) {
		int bitLength = 0;
		if( typespecs.contains(Specifier.CHAR) ) {
			bitLength = 8;
		} else if( typespecs.contains(Specifier.SHORT) ) {
			bitLength = 16;
		} else if( typespecs.contains(Specifier.INT) ) {
			if( typespecs.contains(Specifier.LONG) ) {
				//Assume sizeof(long int) = 8. 
				bitLength = 64;
			} else {
				//Assume sizeof(int) = 4. 
				bitLength = 32;
			}
		} else if( typespecs.contains(Specifier.FLOAT) ) {
			bitLength = 32;
		} else if( typespecs.contains(Specifier.DOUBLE) ) {
			bitLength = 64;
		} else if( typespecs.contains(Specifier.LONG) ) {
			//Assume sizeof(long) = 8.
			bitLength = 64;
		} else if( typespecs.contains(Specifier.UNSIGNED) ) {
			//Assume sizeof(unsigned) = 4. 
			//unsigned == unsigned int
			bitLength = 32;
		}
		return bitLength;
	}
	
	/**
	 * Find a specifier type whose bit length is the same as the that of input symbol {@code inSym}
	 * FIXME: bitlength of a specifier type is dependent on the target system, and thus this should 
	 * be changeable using a compiler option.
	 * 
	 * @param inSym
	 * @return
	 */
	public static Specifier getBitVecType(Symbol inSym) {
		Specifier uType = null;
		List<Specifier> typespecs = inSym.getTypeSpecifiers();
		int bitLength = getBitLengthOfType(typespecs);
		if( bitLength > 0 ) {
			uType = new UserSpecifier(new NameID("type" + bitLength + "b"));
		}
		return uType;
	}
	
    /**  
     * Search input expression and return an ArrayAccess expression if existing;
     * if there are multiple ArrayAccess expressions, the first one will be returned.
     * If there is no ArrayAccess, return null.
     * 
     * @param iExp input expression to be searched.
     * @return ArrayAccess expression 
     */
    public static ArrayAccess getArrayAccess(Expression iExp) {
        ArrayAccess aAccess = null;
        DepthFirstIterator iter = new DepthFirstIterator(iExp);
        while(iter.hasNext()) {
            Object o = iter.next();
            if(o instanceof ArrayAccess)
            {    
                aAccess = (ArrayAccess)o;
                break;
            }    
        }    
        return aAccess;
    }    
    
    /**  
     * Search input expression and return a list of ArrayAccess expressions if existing;
     * If there is no ArrayAccess, return empty list.
     * 
     * @param iExp input expression to be searched.
     * @return list of ArrayAccess expressions 
     */
    public static List<ArrayAccess> getArrayAccesses(Expression iExp) {
        List<ArrayAccess> aList = new LinkedList<ArrayAccess>();
        DepthFirstIterator iter = new DepthFirstIterator(iExp);
        while(iter.hasNext()) {
            Object o = iter.next();
            if(o instanceof ArrayAccess)
            {    
                aList.add((ArrayAccess)o);
            }    
        }    
        return aList;
    }    
    
    /**
     * Return a string containing the enclosing procedure and translation of the input annotation.
     * 
     * @param annot
     * @return
     */
    public static String getEnclosingAnnotationContext(Annotation annot) {
    	if( annot == null ) {
    		return "";
    	}
    	String ret = null;
    	Traversable at = annot.getAnnotatable();
    	Procedure proc = null;
    	TranslationUnit tUnt = null;
    	while ( at != null ) {
    		if( at instanceof Procedure ) {
    			proc = (Procedure)at;
    		}
    		if( at instanceof TranslationUnit ) {
    			tUnt = (TranslationUnit)at;
    			break;
    		}
    		at = at.getParent();
    	}
    	if( proc != null ) {
    		ret = "\nEnclosing Procedure: " + proc.getSymbolName();
    	} else {
    		ret = "";
    	}
    	if( tUnt != null ) {
    		ret = ret + "\nEnclosing Translation Unit: " + tUnt.getOutputFilename() + "\n";
    	} else {
    		ret = ret + "\n";
    	}
    	return ret;
    }
    
    /**
     * Return a string containing the enclosing procedure and translation of the input annotation.
     * 
     * @param annot
     * @return
     */
    public static String getEnclosingContext(Traversable tr) {
    	if( tr == null ) {
    		return "";
    	}
    	String ret = null;
    	Procedure proc = null;
    	TranslationUnit tUnt = null;
    	while ( tr != null ) {
    		if( tr instanceof Procedure ) {
    			proc = (Procedure)tr;
    		}
    		if( tr instanceof TranslationUnit ) {
    			tUnt = (TranslationUnit)tr;
    			break;
    		}
    		tr = tr.getParent();
    	}
    	if( proc != null ) {
    		ret = "\nEnclosing Procedure: " + proc.getSymbolName();
    	} else {
    		ret = "";
    	}
    	if( tUnt != null ) {
    		ret = ret + "\nEnclosing Translation Unit: " + tUnt.getOutputFilename() + "\n";
    	} else {
    		ret = ret + "\n";
    	}
    	return ret;
    }
    
    /**
     * Return a list of statements where at least one of symbols in the symSet
     * is modified within the input traversable tr. 
     * 
     * 
     * @param symSet a list of symbols to check
     * @param tr 
     * @return
     */
    public static List<Statement> getDefStmts(Set<Symbol> symSet, Traversable tr) {
    	List<Statement> retList = new LinkedList<Statement>();
    	if( tr instanceof ExpressionStatement ) {	
    		Set<Symbol> defSyms = DataFlowTools.getDefSymbol(tr);
    		defSyms.retainAll(symSet);
    		if( !defSyms.isEmpty() ) {
    			retList.add((Statement)tr);
    		}
    	} else if( tr instanceof DeclarationStatement ) {
    		Set<Symbol> defSyms = new HashSet<Symbol>();
    		Declaration decl = ((DeclarationStatement)tr).getDeclaration();
    		if( decl instanceof VariableDeclaration ) {
    			for( Traversable tChild : decl.getChildren() ) {
    				if( tChild instanceof Declarator ) {
    					if( ((Declarator)tChild).getInitializer() != null ) {
    						if( tChild instanceof Symbol ) {
    							defSyms.add((Symbol)tChild);
    						}
    					}
    				}
    			}
    		}
    		defSyms.retainAll(symSet);
    		if( !defSyms.isEmpty() ) {
    			retList.add((Statement)tr);
    		}
    	} else if( tr instanceof CompoundStatement ) {	
    		CompoundStatement cStmt = (CompoundStatement)tr;
    		for( Traversable cChild : cStmt.getChildren() ) {
    			retList.addAll(getDefStmts(symSet, cChild));
    		}
    	} else if( tr instanceof Procedure ) {
    		retList.addAll(getDefStmts(symSet, ((Procedure)tr).getBody()));
    	} else if( tr instanceof IfStatement ) {
    		IfStatement ifstmt = (IfStatement)tr;
    		retList.addAll((getDefStmts(symSet, ifstmt.getThenStatement())));
    		retList.addAll((getDefStmts(symSet, ifstmt.getElseStatement())));
    	} else if( tr instanceof Loop ) {
    		Loop tloop = (Loop)tr;
    		if( tloop instanceof ForLoop ) {
    			retList.addAll(getDefStmts(symSet, ((ForLoop)tloop).getInitialStatement()));
    		}
    		retList.addAll(getDefStmts(symSet, tloop.getBody()));
    	} else if( tr instanceof SwitchStatement ) {
    		retList.addAll(getDefStmts(symSet, ((SwitchStatement)tr).getBody()));
    	} else if( (tr instanceof TranslationUnit) || (tr instanceof Program) ) {
    		DFIterator<Procedure> iter =
    			new DFIterator<Procedure>(tr, Procedure.class);
    		iter.pruneOn(Procedure.class);
    		iter.pruneOn(Statement.class);
    		iter.pruneOn(Declaration.class);
    		while (iter.hasNext()) {
    			retList.addAll(getDefStmts(symSet, iter.next().getBody()));
    		}
    	} else {
    		//Ignore statements that do not modify symbols.
    	}

    	return retList;
    }


}
