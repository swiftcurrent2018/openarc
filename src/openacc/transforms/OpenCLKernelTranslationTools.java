package openacc.transforms;

import cetus.hir.*;
import cetus.transforms.TransformPass;
import openacc.hir.OpenCLSpecifier;
import openacc.hir.OpenCLStdLibrary;

import java.util.ArrayList;
import java.util.HashSet;
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
    /**
     * @param program
     */
    public OpenCLKernelTranslationTools(Program program)
    {
        super(program);
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
                    ((TranslationUnit) t).getOutputFilename().compareTo("openarc_kernel.cl") == 0)
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
        for(Declaration decl : kernelsTranslationUnit.getDeclarations())
        {
        	if(!(decl instanceof Procedure))
        		continue;

        	Procedure kernelProc = (Procedure)decl;
        	List typeList = kernelProc.getReturnType();
        	if( typeList.contains(OpenCLSpecifier.OPENCL_KERNEL)) {
        		addAddressSpaceQualifier(kernelProc, visitedProcedureSet);
        	}
        }
    }
    
    protected void addAddressSpaceQualifier(Procedure devProc, Set<Procedure> visitedProcedureSet) {
    	if( visitedProcedureSet.contains(devProc) ) {
    		return;
    	} else {
    		visitedProcedureSet.add(devProc);
    	}
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
    							Set<Symbol> accessedSyms = SymbolTools.getAccessedSymbols(rExp);
    							for(Symbol ttSym : accessedSyms ) {
    								if( SymbolTools.isPointer(ttSym) || SymbolTools.isArray(ttSym) ) {
    									List ttSpec = ttSym.getTypeSpecifiers();
    									if( ttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    										if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    											VariableDeclaration tDecl = (VariableDeclaration)tSym.getDeclaration();
    											tDecl.getSpecifiers().add(0, OpenCLSpecifier.OPENCL_GLOBAL);
    										}
    										localPointers.remove(tSym);
    										if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) || 
    												tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    											DFIterator<Identifier> titer =
    													new DFIterator<Identifier>(rExp, Identifier.class);
    											while( titer.hasNext() ) {
    												List tttSpec = titer.next().getSymbol().getTypeSpecifiers();
    												if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													break;
    												}
    											}
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
    											VariableDeclaration tDecl = (VariableDeclaration)tSym.getDeclaration();
    											tDecl.getSpecifiers().add(0, OpenCLSpecifier.OPENCL_LOCAL);
    										}
    										localPointers.remove(tSym);
    										if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) || 
    												tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    											DFIterator<Identifier> titer =
    													new DFIterator<Identifier>(rExp, Identifier.class);
    											while( titer.hasNext() ) {
    												List tttSpec = titer.next().getSymbol().getTypeSpecifiers();
    												if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													break;
    												}
    											}
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
    											VariableDeclaration tDecl = (VariableDeclaration)tSym.getDeclaration();
    											tDecl.getSpecifiers().add(0, OpenCLSpecifier.OPENCL_CONSTANT);
    											tDecl.getSpecifiers().remove(Specifier.CONST);
    										}
    										localPointers.remove(tSym);
    										if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) || 
    												tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    											DFIterator<Identifier> titer =
    													new DFIterator<Identifier>(rExp, Identifier.class);
    											while( titer.hasNext() ) {
    												List tttSpec = titer.next().getSymbol().getTypeSpecifiers();
    												if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    													break;
    												} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    													tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    													break;
    												}
    											}
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
    				Initializer tInit = tDecl.getDeclarator(0).getInitializer();
    				if( tInit != null ) {
    					if( localPointers.contains(tSym) ) {
    						Typecast tInitTCExp = null;
    						if( tInit.getChildren().get(0) instanceof Typecast ) {
    							tInitTCExp = (Typecast)tInit.getChildren().get(0);
    						}
    						Set<Symbol> accessedSyms = SymbolTools.getAccessedSymbols(tInit);
    						for(Symbol ttSym : accessedSyms ) {
    							if( SymbolTools.isPointer(ttSym) || SymbolTools.isArray(ttSym) ) {
    								List ttSpec = ttSym.getTypeSpecifiers();
    								if( ttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    									if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    										tDecl.getSpecifiers().add(0, OpenCLSpecifier.OPENCL_GLOBAL);
    									}
    									localPointers.remove(tSym);
    									if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) || 
    											tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    										DFIterator<Identifier> titer =
    												new DFIterator<Identifier>(tInit, Identifier.class);
    										while( titer.hasNext() ) {
    											List tttSpec = titer.next().getSymbol().getTypeSpecifiers();
    											if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												break;
    											}
    										}
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
    										tDecl.getSpecifiers().add(0, OpenCLSpecifier.OPENCL_LOCAL);
    									}
    									localPointers.remove(tSym);
    									if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) || 
    											tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    										DFIterator<Identifier> titer =
    												new DFIterator<Identifier>(tInit, Identifier.class);
    										while( titer.hasNext() ) {
    											List tttSpec = titer.next().getSymbol().getTypeSpecifiers();
    											if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												break;
    											}
    										}
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
    										tDecl.getSpecifiers().add(0, OpenCLSpecifier.OPENCL_CONSTANT);
    										tDecl.getSpecifiers().remove(Specifier.CONST);
    									}
    									localPointers.remove(tSym);
    									if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) || 
    											tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    										DFIterator<Identifier> titer =
    												new DFIterator<Identifier>(tInit, Identifier.class);
    										while( titer.hasNext() ) {
    											List tttSpec = titer.next().getSymbol().getTypeSpecifiers();
    											if( tttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_LOCAL);
    												break;
    											} else if( tttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_CONSTANT);
    												tSym.getTypeSpecifiers().remove(OpenCLSpecifier.OPENCL_GLOBAL);
    												break;
    											}
    										}
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
						//Step1: find argument symbol which is a parameber symbol of the calling procedure.
						Symbol argSym = SymbolTools.getSymbolOf(argExp);
						if( argSym == null ) {
							if( argExp instanceof BinaryExpression ) {
								//find argSym which is a parameter symbol of the calling procedure.
								Set<Symbol> sSet = SymbolTools.getAccessedSymbols(argExp);
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
							if( argSym != null ) {
								if( argSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
									if( !typeSpecs.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
										typeSpecs.add(0, OpenCLSpecifier.OPENCL_GLOBAL);
									}
								} else if( argSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
									if( !typeSpecs.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
										typeSpecs.add(0, OpenCLSpecifier.OPENCL_CONSTANT);
										typeSpecs.remove(Specifier.CONST);
									}
								} else if( argSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
									if( !typeSpecs.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
										typeSpecs.add(0, OpenCLSpecifier.OPENCL_LOCAL);
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
