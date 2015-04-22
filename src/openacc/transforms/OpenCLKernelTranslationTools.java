package openacc.transforms;

import cetus.hir.*;
import cetus.transforms.TransformPass;
import openacc.hir.OpenCLSpecifier;
import openacc.hir.OpenCLStdLibrary;

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
        for(Declaration decl : kernelsTranslationUnit.getDeclarations())
        {
        	if(!(decl instanceof Procedure))
        		continue;

        	Procedure kernelProc = (Procedure)decl;
        	List typeList = kernelProc.getReturnType();
        	if( typeList.contains(OpenCLSpecifier.OPENCL_KERNEL)) {
        		CompoundStatement kBody = kernelProc.getBody();
        		Set<Symbol> localPointers = new HashSet<Symbol>();
        		Set<Symbol> localSymbols = SymbolTools.getLocalSymbols(kBody);
        		for( Symbol lSym : localSymbols ) {
        			if( SymbolTools.isPointer(lSym) ) {
        				localPointers.add(lSym);
        			}
        		}
        		if( localPointers.isEmpty() ) {
        			continue;
        		}

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
        					if( lExp instanceof Identifier ) {
        						Symbol tSym = ((Identifier)lExp).getSymbol();
        						if( localPointers.contains(tSym) ) {
        							Set<Symbol> accessedSyms = SymbolTools.getAccessedSymbols(aExp.getRHS());
        							for(Symbol ttSym : accessedSyms ) {
        								if( SymbolTools.isPointer(ttSym) || SymbolTools.isArray(ttSym) ) {
        									List ttSpec = ttSym.getTypeSpecifiers();
        									if( ttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
        										if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
        											VariableDeclaration tDecl = (VariableDeclaration)tSym.getDeclaration();
        											tDecl.getSpecifiers().add(OpenCLSpecifier.OPENCL_GLOBAL);
        										}
        										localPointers.remove(tSym);
        										if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) || 
        												tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
        											DFIterator<Identifier> titer =
        												new DFIterator<Identifier>(aExp.getRHS(), Identifier.class);
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
        										break;
        									} else if( ttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
        										if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
        											VariableDeclaration tDecl = (VariableDeclaration)tSym.getDeclaration();
        											tDecl.getSpecifiers().add(OpenCLSpecifier.OPENCL_LOCAL);
        										}
        										localPointers.remove(tSym);
        										if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) || 
        												tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
        											DFIterator<Identifier> titer =
        												new DFIterator<Identifier>(aExp.getRHS(), Identifier.class);
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
        										break;
        									} else if( ttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
        										if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
        											VariableDeclaration tDecl = (VariableDeclaration)tSym.getDeclaration();
        											tDecl.getSpecifiers().add(OpenCLSpecifier.OPENCL_CONSTANT);
        										}
        										localPointers.remove(tSym);
        										if( tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) || 
        												tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
        											DFIterator<Identifier> titer =
        												new DFIterator<Identifier>(aExp.getRHS(), Identifier.class);
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
        						Set<Symbol> accessedSyms = SymbolTools.getAccessedSymbols(tInit);
        						for(Symbol ttSym : accessedSyms ) {
        							if( SymbolTools.isPointer(ttSym) || SymbolTools.isArray(ttSym) ) {
        								List ttSpec = ttSym.getTypeSpecifiers();
        								if( ttSpec.contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
        									if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_GLOBAL) ) {
        										tDecl.getSpecifiers().add(OpenCLSpecifier.OPENCL_GLOBAL);
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
        									break;
        								} else if( ttSpec.contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
        									if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL) ) {
        										tDecl.getSpecifiers().add(OpenCLSpecifier.OPENCL_LOCAL);
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
        									break;
        								} else if( ttSpec.contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
        									if( !tSym.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_CONSTANT) ) {
        										tDecl.getSpecifiers().add(OpenCLSpecifier.OPENCL_CONSTANT);
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
        									break;
        								}
        							}
        						}
        					}
        				}
        			}
        		}
        	}
        }
    }
}
