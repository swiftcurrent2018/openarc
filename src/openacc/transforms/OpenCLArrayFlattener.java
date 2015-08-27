package openacc.transforms;

import cetus.hir.*;
import openacc.hir.OpenCLSpecifier;
import cetus.transforms.TransformPass;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Created with IntelliJ IDEA.
 * User: Putt Sakdhnagool <psakdhna@purdue.edu>
 * Date: 10/17/13
 * Time: 10:21 AM
 * To change this template use File | Settings | File Templates.
 */
public class OpenCLArrayFlattener extends TransformPass
{
	private boolean addRestrictQualifier = false;
    /**
     * @param program
     */
    public OpenCLArrayFlattener(Program program, boolean addRestrictQual)
    {
        super(program);
        addRestrictQualifier = addRestrictQual;
    }

    @Override
    public String getPassName()
    {
        return new String("[array flattening transformation]");
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
            PrintTools.println("[OpenCLArrayFlattener] kernel file is missing.", 0);
            return;
        }

        for(Declaration decl : kernelsTranslationUnit.getDeclarations())
        {
            if(!(decl instanceof Procedure))
                continue;

            Procedure kernelProc = (Procedure)decl;
            CompoundStatement kernelBody = kernelProc.getBody();

            //Flatten all array accesses
            List<ArrayAccess> arrayAccesses = IRTools.getExpressionsOfType(kernelBody, ArrayAccess.class);
            for(ArrayAccess access : arrayAccesses)
            {
                if(access.getNumIndices() > 1)
                {
                    Symbol accessSymbol = SymbolTools.getSymbolOf(access.getArrayName());

                    List<Expression> accessIndices = access.getIndices();
					ArraySpecifier accessSpecifier = (ArraySpecifier)accessSymbol.getArraySpecifiers().get(0);

					boolean isPointer = false;

                    Expression indexExpr = accessIndices.get(0);

					// Skip for shared memory
					if(accessSymbol.getTypeSpecifiers().contains(OpenCLSpecifier.OPENCL_LOCAL))
						continue;

					/*
					PrintTools.println("access: " + access.toString(),0);
					PrintTools.println("symbol: " + accessSymbol.toString(),0);
					PrintTools.println("spec:   " + accessSpecifier.toString(),0);
					PrintTools.println("typespec: " + accessSymbol.getTypeSpecifiers().toString(), 0);
					PrintTools.println("name:   " + accessSymbol.getSymbolName(), 0);
					PrintTools.println("decl:   " + accessSymbol.getDeclaration().toString(), 0);
					*/

					// In some benchmark, the array declaration use the following pattern
					// float (*var)[CONST]
					// which is 2D array but cetus will recognize as 1D array.
					if(accessSymbol.getDeclaration().getChildren().get(0) instanceof NestedDeclarator)
                    {
                        NestedDeclarator declarator = (NestedDeclarator)accessSymbol.getDeclaration().getChildren().get(0);
						List<Specifier> nestedSpecifiers = declarator.getDeclarator().getSpecifiers();
						if(nestedSpecifiers.size() == 1)
						{
							//[DEBUG at Aug. 3, 2015]
							//if(nestedSpecifiers.get(0) == PointerSpecifier.UNQUALIFIED)
							if(nestedSpecifiers.get(0) instanceof PointerSpecifier)
							{
								isPointer = true;
							}
							else
							{
								Tools.exit("[OpenCLArrayFlattener] Unexpected Nested Declaration Specifier: " + nestedSpecifiers.get(0));
							}
						}
						else
						{
							Tools.exit("[OpenCLArrayFlattener] Nested Declaration: " + accessSymbol.toString() + " is not supported.");
						}
                    }


                    for(int i = 1; i < access.getNumIndices(); i++)
                    {
						if(isPointer)
                        {
							indexExpr = new BinaryExpression(indexExpr.clone(), BinaryOperator.MULTIPLY, accessSpecifier.getDimension(i-1).clone());
                        }
						else
						{
							indexExpr = new BinaryExpression(indexExpr.clone(), BinaryOperator.MULTIPLY, accessSpecifier.getDimension(i).clone());
                        }
						indexExpr = new BinaryExpression(indexExpr.clone(), BinaryOperator.ADD, accessIndices.get(i).clone());
                    }

					ArrayAccess newAccess = new ArrayAccess(access.getArrayName().clone(), indexExpr);
                    access.swapWith(newAccess);
                }
            }

			//Change function parameters from static array to pointer
            //e.g., float a[N]; => float *a;
            //      float b[M][N] => float *b;
            //      float (*c)[N] => float *c;
            ProcedureDeclarator kernelProcDeclarator = (ProcedureDeclarator)kernelProc.getDeclarator();
            Map<Declaration, Declaration> replaceDeclMap = new HashMap<Declaration, Declaration>();
            for(int i = 0; i < kernelProcDeclarator.getChildren().size(); i++)
            {
                if(kernelProcDeclarator.getChildren().get(i) instanceof Declaration)
                {
                    VariableDeclaration param = (VariableDeclaration)kernelProcDeclarator.getChildren().get(i);
                    Declarator paramDeclr = param.getDeclarator(0);
                    List specs = null;
                    if( paramDeclr.getArraySpecifiers().size() > 0 ) {
                    	paramDeclr.getArraySpecifiers().clear();
                    	if( paramDeclr instanceof VariableDeclarator ) {
                    		VariableDeclarator varSym = (VariableDeclarator)paramDeclr;
                    		specs = varSym.getTypeSpecifiers();
                    		// Separate declarator/declaration specifiers.
                    		List declaration_specs = new ArrayList(specs.size());
                    		List declarator_specs = new ArrayList(specs.size());
                    		for (int k = 0; k < specs.size(); k++) {
                    			Object spec = specs.get(k);
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
                    		if( declarator_specs.isEmpty() ) {
                    			if( addRestrictQualifier ) {
                    				declarator_specs.add(PointerSpecifier.RESTRICT);
                    			} else {
                    				declarator_specs.add(PointerSpecifier.UNQUALIFIED);
                    			}
                    		}
                    		VariableDeclarator Declr = new VariableDeclarator(declarator_specs, new NameID(varSym.getSymbolName()));

                    		Declaration decls = new VariableDeclaration(declaration_specs, Declr);
                    		replaceDeclMap.put(param, decls);
                    	} else if( paramDeclr instanceof NestedDeclarator) {
                    		NestedDeclarator nestedSym = (NestedDeclarator)paramDeclr;
                    		Declarator childDeclarator = nestedSym.getDeclarator();
                    		List childDeclr_specs = null;
                    		if( childDeclarator instanceof VariableDeclarator ) {
                    			VariableDeclarator varSym = (VariableDeclarator)childDeclarator;
                    			childDeclr_specs = varSym.getSpecifiers();
                    			varSym.getArraySpecifiers().clear();
                    		} else {
                    			Tools.exit("[ERROR in OpenCLArrayFlattener()] nested declarator whose child declarator is also " +
                    					"nested declarator is not supported yet; exit!\n" +
                    					"Symbol: " + paramDeclr + "\n");
                    		}
                    		specs = nestedSym.getTypeSpecifiers();
                    		if( childDeclr_specs != null ) {
                    			specs.addAll(childDeclr_specs);
                    		}
                    		// Separate declarator/declaration specifiers.
                    		List declaration_specs = new ArrayList(specs.size());
                    		List declarator_specs = new ArrayList(specs.size());
                    		for (int k = 0; k < specs.size(); k++) {
                    			Object spec = specs.get(k);
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
                    		if( declarator_specs.isEmpty() ) {
                    			if( addRestrictQualifier ) {
                    				declarator_specs.add(PointerSpecifier.RESTRICT);
                    			} else {
                    				declarator_specs.add(PointerSpecifier.UNQUALIFIED);
                    			}
                    		}
                    		VariableDeclarator Declr = new VariableDeclarator(declarator_specs, new NameID(nestedSym.getSymbolName()));

                    		Declaration decls = new VariableDeclaration(declaration_specs, Declr);
                    		replaceDeclMap.put(param, decls);
                    	}
                    }
                }
            }
            for( Declaration refD : replaceDeclMap.keySet() ) {
            	kernelProc.replaceDeclaration(refD, replaceDeclMap.get(refD));
            }
        }
    }
}
