package openacc.transforms;

import cetus.hir.*;
import openacc.hir.OpenCLSpecifier;
import cetus.transforms.TransformPass;

import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: Putt Sakdhnagool <psakdhna@purdue.edu>
 * Date: 10/17/13
 * Time: 10:21 AM
 * To change this template use File | Settings | File Templates.
 */
public class OpenCLArrayFlattener extends TransformPass
{
    /**
     * @param program
     */
    public OpenCLArrayFlattener(Program program)
    {
        super(program);
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
							if(nestedSpecifiers.get(0) == PointerSpecifier.UNQUALIFIED)
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
            ProcedureDeclarator kernelProcDeclarator = (ProcedureDeclarator)kernelProc.getDeclarator();
            for(int i = 0; i < kernelProcDeclarator.getChildren().size(); i++)
            {
                if(kernelProcDeclarator.getChildren().get(i) instanceof Declaration)
                {
                    VariableDeclaration param = (VariableDeclaration)kernelProcDeclarator.getChildren().get(i);

					if(param.getDeclarator(0).getArraySpecifiers().size() > 0)
                    {
						//Remove * from (*var)[CONST] declaration
						if(param.getDeclarator(0) instanceof NestedDeclarator)
						{
							((NestedDeclarator)param.getDeclarator(0)).getDeclarator().getSpecifiers().clear();
						}

						//Add * to make a pointer type
    		            List<Specifier> specifiers = param.getSpecifiers();
						specifiers.add(PointerSpecifier.UNQUALIFIED);

						param.getDeclarator(0).getArraySpecifiers().clear();
                    }
                }
            }
        }
    }
}
