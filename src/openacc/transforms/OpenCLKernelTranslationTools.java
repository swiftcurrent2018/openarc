package openacc.transforms;

import cetus.hir.*;
import cetus.exec.Driver;
import cetus.transforms.TransformPass;
import openacc.hir.OpenCLSpecifier;
import openacc.hir.OpenCLStdLibrary;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
 * Created with IntelliJ IDEA.
 * User: Putt Sakdhnagool <psakdhna@purdue.edu>
 * Date: 10/17/13
 * Time: 10:21 AM
 * To change this template use File | Settings | File Templates.
 */
public class OpenCLKernelTranslationTools extends TransformPass
{
	private String kernelFileNameBase = "openarc_kernel";
	private boolean addressSpecifierUpdated = true;
    /**
     * @param program
     */
    public OpenCLKernelTranslationTools(Program program)
    {
        super(program);
		String value = Driver.getOptionValue("SetOutputKernelFileNameBase");
		if( value != null ) {
			kernelFileNameBase = value;
		}
    }

    @Override
    public String getPassName()
    {
        return new String("[OpenCLKernelTranslationTools]");
    }

    @Override
    public void start()
    {
        TranslationUnit kernelsTranslationUnit = null;

        for(Traversable t : program.getChildren())
        {
            if(t instanceof TranslationUnit &&
                    ((TranslationUnit) t).getOutputFilename().compareTo(kernelFileNameBase + ".cl") == 0)
            {
                kernelsTranslationUnit = (TranslationUnit)t;
                break;
            }
        }

        if(kernelsTranslationUnit == null)
        {
            PrintTools.println("[OpenCLKernelTranslationTools] kernel file is missing.", 0);
            return;
        }

        ///////////////////////////////////////////////////////////////
        // Step1: Convert some math function to OpenCL math function //
        ///////////////////////////////////////////////////////////////
        List<FunctionCall> callList = IRTools.getExpressionsOfType(kernelsTranslationUnit, FunctionCall.class);

        for(FunctionCall call : callList)
        {
        	String functionName = call.getName().toString();
        	if( StandardLibrary.contains(call) ) {
        		if( !OpenCLStdLibrary.contains(call) && (functionName.charAt(functionName.length()-1) == 'f') ) {
        			String nFunctionName = functionName.substring(0, functionName.length()-1);
        			if( OpenCLStdLibrary.contains(nFunctionName) ) {
        				call.setFunction(new NameID(nFunctionName));
        			}
        		}
        	}
        }
        
        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Step2: Add OpenCL address space qualifiers to local pointer variables in an OpenCL kernel   //
        // if they refer to either __global, __local, or __constant address space.                     //
        // A pointer to address space A can only be assigned to a pointer to the same address space A. // 
        // Casting a pointer to address space A to a pointer to address space B is illegal.            //
        /////////////////////////////////////////////////////////////////////////////////////////////////
        Set<Procedure> visitedProcedureSet = new HashSet<Procedure>();
        List<Procedure> kernelProcedures = new LinkedList<Procedure>();
        for(Declaration decl : kernelsTranslationUnit.getDeclarations())
        {
        	if(!(decl instanceof Procedure))
        		continue;

        	Procedure kernelProc = (Procedure)decl;
        	List typeList = kernelProc.getReturnType();
        	if( typeList.contains(OpenCLSpecifier.OPENCL_KERNEL)) {
        		kernelProcedures.add(kernelProc);
        	}
        }
        int i=0;
        while( addressSpecifierUpdated ) {
        	//PrintTools.println("\n==> Address space qualitier update phase " + i++, 0);
        	addressSpecifierUpdated = false;
        	for(Procedure kernelProc : kernelProcedures)
        	{
        		addAddressSpaceQualifier(kernelProc, visitedProcedureSet);
        	}
        	visitedProcedureSet.clear();
        }
    }
    
    protected Set<Symbol> getAccessedSymbolsExcludingArguments(Traversable inExp) {
    	Set<Symbol> retSet = new HashSet<Symbol>();
    	if( inExp != null ) {
    		DFIterator<Expression> iter =
            new DFIterator<Expression>(inExp, Expression.class);
    		iter.pruneOn(FunctionCall.class);
    		while (iter.hasNext()) {
    			Symbol tSym = null;
    			Expression tExp = iter.next();
    			if( tExp instanceof FunctionCall ) {
    				tSym = SymbolTools.getSymbolOf(((FunctionCall)tExp).getName());
    			} else if( tExp instanceof IDExpression ) {
    				tSym = SymbolTools.getSymbolOf(tExp);
    			}
    			if (tSym != null) {
    				retSet.add(tSym);
    			}
    		}
    	}
    	return retSet;
    }
    
    protected void addAddressSpaceQualifier(Procedure devProc, Set<Procedure> visitedProcedureSet) {
    	if( visitedProcedureSet.contains(devProc) ) {
    		return;
    	} else {
    		visitedProcedureSet.add(devProc);
    	}
        //PrintTools.println("[OpenCLKernelTranslationTools.addAddressSpaceQualifier()] visit a function, " + devProc.getSymbolName(), 0);
    	CompoundStatement kBody = devProc.getBody();
    	Set<Symbol> localPointers = new HashSet<Symbol>();
    	Set<Symbol> localSymbols = SymbolTools.getLocalSymbols(kBody);
    	for( Symbol lSym : localSymbols ) {
    		if( SymbolTools.isPointer(lSym) ) {
    			localPointers.add(lSym);
    		}
    	}
    	if( !localPointers.isEmpty() ) {
    		DFIterator<Statement> iter =
    				new DFIterator<Statement>(kBody, Statement.class);
    		iter.pruneOn(DeclarationStatement.class);
    		iter.pruneOn(AnnotationStatement.class);
    		iter.pruneOn(ExpressionStatement.class);
    		iter.pruneOn(Expression.class);
    		while (iter.hasNext()) {
    			if( localPointers.isEmpty() ) {
    				break;
    			}
    			Statement tStmt = iter.next();
    			if( tStmt instanceof ExpressionStatement ) {
    				ExpressionStatement eStmt = (ExpressionStatement)tStmt;
    				//Find pointer-assignment expression.
    				if( eStmt.getExpression() instanceof AssignmentExpression ) {
    					AssignmentExpression aExp = (AssignmentExpression)eStmt.getExpression();
    					Expression lExp = aExp.getLHS();
    					Expression rExp = aExp.getRHS();
    					if( lExp instanceof Identifier ) {
    						Symbol tSym = ((Identifier)lExp).getSymbol();
    						if( localPointers.contains(tSym) ) {
    							VariableDeclaration tDecl = (VariableDeclaration)tSym.getDeclaration();
    							List<Specifier> tSpec = tDecl.getSpecifiers();
    							//Set<Symbol> accessedSyms = SymbolTools.getAccessedSymbols(rExp);
    							Set<Symbol> accessedSyms = getAccessedSymbolsExcludingArguments(rExp);
    							for(Symbol ttSym : accessedSyms ) {
    								if( SymbolTools.isPointer(ttSym) || SymbolTools.isArray(ttSym) ) {
    									//List ttSpec = ttSym.getTypeSpecifiers();
    									List<Specifier> ttSpec = null;
    									Declaration ttDecl = ttSym.getDeclaration();
    									if( ttDecl instanceof VariableDeclaration  ) {
    										ttSpec = ((VariableDeclaration)ttDecl).getSpecifiers();
    									} else if( ttDecl instanceof Procedure ) {
    										ttSpec = ((Procedure)ttDecl).getReturnType();
    									}
    									if( ttSpec == null ) {
    										continue;
    									}
    									if( ttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    										if( !tSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    											tSpec.add(0, OpenCLSpecifier.OPENCL_GLOBAL);
    											addressSpecifierUpdated = true;
    											//PrintTools.println("[addAddressSpaceQualifier() case 1] update a local symbol, " + tDecl, 0);
    											if(tSym.getSymbolName().contains("_ret_val_")) {
    												List returnTypes = devProc.getSpecifiers();
    												if( !returnTypes.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    													returnTypes.add(0, OpenCLSpecifier.OPENCL_GLOBAL);
    												}
    											}
    										}
    										localPointers.remove(tSym);
    										if( tSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) || 
    												tSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    											Tools.exit("[ERROR in OpenCLKernelTranslationTools.addAddressSpaceQualiter()] more than one address space qualifier "
    													+ "is added to a variable, " + tSym + " in the procedure, " + devProc.getSymbolName() + "\n");
/*    											DFIterator<Identifier> titer =
    													new DFIterator<Identifier>(rExp, Identifier.class);
    											while( titer.hasNext() ) {
    												VariableDeclaration tttDecl = (VariableDeclaration)titer.next().getSymbol().getDeclaration();
    												List tttSpec = tttDecl.getSpecifiers();
    												if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    													tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    													tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    													tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													break;
    												}
    											}*/
    										}
    										if( rExp instanceof Typecast ) {
    											Typecast rTCExp = (Typecast)rExp;
    											List rTCSpecList = rTCExp.getSpecifiers();
    											if( !rTCSpecList.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    												rTCSpecList.add(0, OpenCLSpecifier.OPENCL_GLOBAL);
    											}
    										}
    										break;
    									} else if( ttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    										if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    											tSpec.add(0, OpenCLSpecifier.OPENCL_LOCAL);
    											addressSpecifierUpdated = true;
    											if(tSym.getSymbolName().contains("_ret_val_")) {
    												List returnTypes = devProc.getSpecifiers();
    												if( !returnTypes.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    													returnTypes.add(0, OpenCLSpecifier.OPENCL_LOCAL);
    												}
    											}
    										}
    										localPointers.remove(tSym);
    										if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) || 
    												tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    											Tools.exit("[ERROR in OpenCLKernelTranslationTools.addAddressSpaceQualiter()] more than one address space qualifier "
    													+ "is added to a variable, " + tSym + " in the procedure, " + devProc.getSymbolName() + "\n");
/*    											DFIterator<Identifier> titer =
    													new DFIterator<Identifier>(rExp, Identifier.class);
    											while( titer.hasNext() ) {
    												VariableDeclaration tttDecl = (VariableDeclaration)titer.next().getSymbol().getDeclaration();
    												List tttSpec = tttDecl.getSpecifiers();
    												if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    													tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    													tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    													tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													break;
    												}
    											}*/
    										}
    										if( rExp instanceof Typecast ) {
    											Typecast rTCExp = (Typecast)rExp;
    											List rTCSpecList = rTCExp.getSpecifiers();
    											if( !rTCSpecList.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    												rTCSpecList.add(0, OpenCLSpecifier.OPENCL_LOCAL);
    											}
    										}
    										break;
    									} else if( ttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    										if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    											tSpec.add(0, OpenCLSpecifier.OPENCL_CONSTANT);
    											tSpec.remove(Specifier.CONST);
    											addressSpecifierUpdated = true;
    											if(tSym.getSymbolName().contains("_ret_val_")) {
    												List returnTypes = devProc.getSpecifiers();
    												if( !returnTypes.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    													returnTypes.add(0, OpenCLSpecifier.OPENCL_CONSTANT);
    													returnTypes.remove(Specifier.CONST);
    												}
    											}
    										}
    										localPointers.remove(tSym);
    										if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) || 
    												tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    											Tools.exit("[ERROR in OpenCLKernelTranslationTools.addAddressSpaceQualiter()] more than one address space qualifier "
    													+ "is added to a variable, " + tSym + " in the procedure, " + devProc.getSymbolName() + "\n");
 /*   											DFIterator<Identifier> titer =
    													new DFIterator<Identifier>(rExp, Identifier.class);
    											while( titer.hasNext() ) {
    												VariableDeclaration tttDecl = (VariableDeclaration)titer.next().getSymbol().getDeclaration();
    												List tttSpec = tttDecl.getSpecifiers();
    												if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    													tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    													tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    													tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													break;
    												}
    											}*/
    										}
    										if( rExp instanceof Typecast ) {
    											Typecast rTCExp = (Typecast)rExp;
    											List rTCSpecList = rTCExp.getSpecifiers();
    											if( !rTCSpecList.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    												rTCSpecList.add(0, OpenCLSpecifier.OPENCL_CONSTANT);
    											}
    										}
    										break;
    									}
    								}
    							}
    						}
    					}
    				}
    			} else if( tStmt instanceof DeclarationStatement ) {
    				VariableDeclaration tDecl = (VariableDeclaration)((DeclarationStatement)tStmt).getDeclaration();
    				Symbol tSym = (Symbol)tDecl.getDeclarator(0);
    				List<Specifier> tSpec = tDecl.getSpecifiers();
    				Initializer tInit = tDecl.getDeclarator(0).getInitializer();
    				if( tInit != null ) {
    					if( localPointers.contains(tSym) ) {
    						Typecast tInitTCExp = null;
    						if( tInit.getChildren().get(0) instanceof Typecast ) {
    							tInitTCExp = (Typecast)tInit.getChildren().get(0);
    						}
    						//Set<Symbol> accessedSyms = SymbolTools.getAccessedSymbols(tInit);
    						Set<Symbol> accessedSyms = getAccessedSymbolsExcludingArguments(tInit);
    						for(Symbol ttSym : accessedSyms ) {
    							if( SymbolTools.isPointer(ttSym) || SymbolTools.isArray(ttSym) ) {
    								//List ttSpec = ttSym.getTypeSpecifiers();
    								List<Specifier> ttSpec = null;
    								Declaration ttDecl = ttSym.getDeclaration();
    								if( ttDecl instanceof VariableDeclaration  ) {
    									ttSpec = ((VariableDeclaration)ttDecl).getSpecifiers();
    								} else if( ttDecl instanceof Procedure ) {
    									ttSpec = ((Procedure)ttDecl).getReturnType();
    								}
    								if( ttSpec == null ) {
    									continue;
    								}
    								if( ttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    									if( !tSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    										tSpec.add(0, OpenCLSpecifier.OPENCL_GLOBAL);
    										addressSpecifierUpdated = true;
    										//PrintTools.println("[addAddressSpaceQualifier() case 2] update a local symbol, " + tDecl, 0);
    										if(tSym.getSymbolName().contains("_ret_val_")) {
    											List returnTypes = devProc.getSpecifiers();
    											if( !returnTypes.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    												returnTypes.add(0, OpenCLSpecifier.OPENCL_GLOBAL);
    											}
    										}
    									}
    									localPointers.remove(tSym);
    									if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) || 
    											tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    										Tools.exit("[ERROR in OpenCLKernelTranslationTools.addAddressSpaceQualiter()] more than one address space qualifier "
    												+ "is added to a variable, " + tSym + " in the procedure, " + devProc.getSymbolName() + "\n");
/*    										DFIterator<Identifier> titer =
    												new DFIterator<Identifier>(tInit, Identifier.class);
    										while( titer.hasNext() ) {
    											//List tttSpec = titer.next().getSymbol().getTypeSpecifiers();
    											VariableDeclaration tttDecl = (VariableDeclaration)titer.next().getSymbol().getDeclaration();
    											List tttSpec = tttDecl.getSpecifiers();
    											if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    												tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    												tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    												tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												break;
    											}
    										}*/
    									}
    									if( tInitTCExp != null ) {
    										List rTCSpecList = tInitTCExp.getSpecifiers();
    										if( !rTCSpecList.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    											rTCSpecList.add(0, OpenCLSpecifier.OPENCL_GLOBAL);
    										}
    									}
    									break;
    								} else if( ttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    									if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    										//tDecl.getSpecifiers().add(0, OpenCLSpecifier.OPENCL_LOCAL);
    										tSpec.add(0, OpenCLSpecifier.OPENCL_LOCAL);
    										addressSpecifierUpdated = true;
    										if(tSym.getSymbolName().contains("_ret_val_")) {
    											List returnTypes = devProc.getSpecifiers();
    											if( !returnTypes.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    												returnTypes.add(0, OpenCLSpecifier.OPENCL_LOCAL);
    											}
    										}
    									}
    									localPointers.remove(tSym);
    									if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) || 
    											tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    										Tools.exit("[ERROR in OpenCLKernelTranslationTools.addAddressSpaceQualiter()] more than one address space qualifier "
    												+ "is added to a variable, " + tSym + " in the procedure, " + devProc.getSymbolName() + "\n");
 /*   										DFIterator<Identifier> titer =
    												new DFIterator<Identifier>(tInit, Identifier.class);
    										while( titer.hasNext() ) {
    											//List tttSpec = titer.next().getSymbol().getTypeSpecifiers();
    											VariableDeclaration tttDecl = (VariableDeclaration)titer.next().getSymbol().getDeclaration();
    											List tttSpec = tttDecl.getSpecifiers();
    											if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    												tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    												tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    												tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												break;
    											}
    										}*/
    									}
    									if( tInitTCExp != null ) {
    										List rTCSpecList = tInitTCExp.getSpecifiers();
    										if( !rTCSpecList.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    											rTCSpecList.add(0, OpenCLSpecifier.OPENCL_LOCAL);
    										}
    									}
    									break;
    								} else if( ttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    									if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    										tSpec.add(0, OpenCLSpecifier.OPENCL_CONSTANT);
    										tSpec.remove(Specifier.CONST);
    										addressSpecifierUpdated = true;
    										if(tSym.getSymbolName().contains("_ret_val_")) {
    											List returnTypes = devProc.getSpecifiers();
    											if( !returnTypes.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    												returnTypes.add(0, OpenCLSpecifier.OPENCL_CONSTANT);
    												returnTypes.remove(Specifier.CONST);
    											}
    										}
    									}
    									localPointers.remove(tSym);
    									if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) || 
    											tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    										Tools.exit("[ERROR in OpenCLKernelTranslationTools.addAddressSpaceQualiter()] more than one address space qualifier "
    												+ "is added to a variable, " + tSym + " in the procedure, " + devProc.getSymbolName() + "\n");
/*    										DFIterator<Identifier> titer =
    												new DFIterator<Identifier>(tInit, Identifier.class);
    										while( titer.hasNext() ) {
    											//List tttSpec = titer.next().getSymbol().getTypeSpecifiers();
    											VariableDeclaration tttDecl = (VariableDeclaration)titer.next().getSymbol().getDeclaration();
    											List tttSpec = tttDecl.getSpecifiers();
    											if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    												tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    												tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												tSpec.remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    												tSpec.remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSpec.remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												break;
    											}
    										}*/
    									}
    									if( tInitTCExp != null ) {
    										List rTCSpecList = tInitTCExp.getSpecifiers();
    										if( !rTCSpecList.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    											rTCSpecList.add(0, OpenCLSpecifier.OPENCL_CONSTANT);
    										}
    									}
    									break;
    								}
    							}
    						}
    					}
    				}
    			}
    		}
    	}
    	//Recursively search device functions called in this function.
    	//[DEBUG] below pass assumes that context-sensitive device-procedure cloning is done before.
    	List<FunctionCall> fCallList = IRTools.getFunctionCalls(kBody);
    	if( fCallList != null ) {
    		for( FunctionCall tCall : fCallList ) {
    			Procedure tProc = tCall.getProcedure();
    			if( (tProc != null) && !StandardLibrary.contains(tCall) && !visitedProcedureSet.contains(tProc) ) {
    				List<Symbol> argSymList = new ArrayList<Symbol>(tCall.getArguments().size());
					for( Expression argExp : tCall.getArguments() ) {
						//Step1: find argument symbol which is a parameter symbol of the calling procedure.
						Symbol argSym = SymbolTools.getSymbolOf(argExp);
						if( argSym == null ) {
							if( argExp instanceof BinaryExpression ) {
								//find argSym which is a parameter symbol of the calling procedure.
								//Set<Symbol> sSet = SymbolTools.getAccessedSymbols(argExp);
								Set<Symbol> sSet = getAccessedSymbolsExcludingArguments(argExp);
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
						argSymList.add(argSym);
					}
					List<VariableDeclaration> oldParamList = 
							(List<VariableDeclaration>)tProc.getParameters();
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
							List<Specifier> typeSpecs = param.getSpecifiers();
							Symbol argSym = argSymList.get(i);
							i++;
							Declarator paramDeclr = param.getDeclarator(0);
							if( paramDeclr instanceof VariableDeclarator ) {
								if( !SymbolTools.isArray((VariableDeclarator)paramDeclr) &&
										!SymbolTools.isPointer((VariableDeclarator)paramDeclr) ) {
									//Scalar function argument belongs to __private space.
									continue;
								}
							}
							if( argSym != null ) {
								List<Specifier> argSpec = null;
								Declaration argDecl = null;
								boolean foundAddressSpecifier = false;
								if( argSym instanceof AccessSymbol ) {
									Symbol memSym = ((AccessSymbol)argSym).getMemberSymbol();
									while( memSym instanceof AccessSymbol ) {
										memSym = ((AccessSymbol)memSym).getMemberSymbol();
									}
									if( memSym instanceof PseudoSymbol ) {
										memSym = ((PseudoSymbol)memSym).getIRSymbol();
									}
									if( SymbolTools.isArray(memSym) || SymbolTools.isPointer(memSym) ) {
										argDecl = memSym.getDeclaration();
										if( argDecl instanceof VariableDeclaration ) {
											argSpec = ((VariableDeclaration)argDecl).getSpecifiers();
										} 
										if( argSpec != null ) {
											if( argSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
												foundAddressSpecifier = true;
												if( !typeSpecs.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
													typeSpecs.add(0, OpenCLSpecifier.OPENCL_GLOBAL);
													addressSpecifierUpdated = true;
													//PrintTools.println("[addAddressSpaceQualifier() case 3] update a local symbol, " + param + " in a procedure, " + tProc.getSymbolName(), 0);
												}
											} else if( argSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
												foundAddressSpecifier = true;
												if( !typeSpecs.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
													typeSpecs.add(0, OpenCLSpecifier.OPENCL_CONSTANT);
													typeSpecs.remove(Specifier.CONST);
													addressSpecifierUpdated = true;
												}
											} else if( argSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
												foundAddressSpecifier = true;
												if( !typeSpecs.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
													typeSpecs.add(0, OpenCLSpecifier.OPENCL_LOCAL);
													addressSpecifierUpdated = true;
												}
											}
										}
									}
								}
								if( !foundAddressSpecifier ) {
									Symbol IRSymbol = null;
									if( argSym instanceof PseudoSymbol ) {
										IRSymbol = ((PseudoSymbol)argSym).getIRSymbol();
									} else {
										IRSymbol = argSym;
									}
									argDecl = IRSymbol.getDeclaration();
									if( argDecl instanceof VariableDeclaration ) {
										argSpec = ((VariableDeclaration)argDecl).getSpecifiers();
									} 
									if( argSpec == null ) {
										continue;
									}
									if( argSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
										if( !typeSpecs.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
											typeSpecs.add(0, OpenCLSpecifier.OPENCL_GLOBAL);
											addressSpecifierUpdated = true;
											//PrintTools.println("[addAddressSpaceQualifier() case 3] update a local symbol, " + param + " in a procedure, " + tProc.getSymbolName(), 0);
										}
									} else if( argSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
										if( !typeSpecs.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
											typeSpecs.add(0, OpenCLSpecifier.OPENCL_CONSTANT);
											typeSpecs.remove(Specifier.CONST);
											addressSpecifierUpdated = true;
										}
									} else if( argSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
										if( !typeSpecs.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
											typeSpecs.add(0, OpenCLSpecifier.OPENCL_LOCAL);
											addressSpecifierUpdated = true;
										}
									}
								}
							}
						}
					}
    				addAddressSpaceQualifier(tProc, visitedProcedureSet);
    			}
    		}
    	}
    }
}
