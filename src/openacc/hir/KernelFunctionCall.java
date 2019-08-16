package openacc.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

import cetus.exec.Driver;
import cetus.hir.*;
import openacc.analysis.AnalysisTools;

/**
 * <b>KernelFunctionCall</b> represents a function or method call.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future TechnologiesGroup, Oak Ridge National Laboratory
 *
 */
public class KernelFunctionCall extends FunctionCall	
{
  private static Method class_print_method;
  private LinkedList<Traversable> configuration;
  private LinkedList<Integer> argTraits;
  private LinkedList<Expression> argSizes;
  //configuration list contains configuration values used for this kernel function call.
  //if targetArch < 4: {Grid dimensions, threadblock dimensions, shared memory size, async ID, wait ID list}
  //if targetArch == 4: {task handle, Global dimensions, threadblock dimensions, mcl flags, source code pointer, async ID, wait ID list}
  private Procedure linkedProcedure;
  private static int trait_readonly = 0;
  private static int trait_writeonly = 1;
  private static int trait_readwrite = 2;
  private static int trait_temporary = 3;

  static
  {
    Class[] params = new Class[2];

    try {
      params[0] = KernelFunctionCall.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  /**
   * Creates a function call.
   *
   * @param function An expression that evaluates to a function.
   */
  public KernelFunctionCall(Expression function)
  {
    super(function);
    configuration = new LinkedList<Traversable>();
    argTraits = new LinkedList<Integer>();
    argSizes = new LinkedList<Expression>();
    object_print_method = class_print_method;
    
  }

  /**
   * Creates a function call.
   *
   * @param function An expression that evaluates to a function.
   * @param args A list of arguments to the function.
   */
  public KernelFunctionCall(Expression function, List args)
  {
    super(function, args);
    configuration = new LinkedList<Traversable>();
    argTraits = new LinkedList<Integer>();
    argSizes = new LinkedList<Expression>();
    object_print_method = class_print_method;
  }

/**
   * Creates a function call.
   *
   * @param function An expression that evaluates to a function.
   * @param args A list of arguments to the function.
   * @param confargs A list of configuration arguments to the function.
   */
  public KernelFunctionCall(Expression function, List args, List confargs)
  {
    super(function, args);
    configuration = new LinkedList<Traversable>();
    argTraits = new LinkedList<Integer>();
    argSizes = new LinkedList<Expression>();
    object_print_method = class_print_method;
    setConfArguments(confargs);
  }

  /**
   * Prints a function call to a stream.
   *
   * @param call The call to print.
   * @param p printwriter
   */
  public static void defaultPrint(KernelFunctionCall call, PrintWriter p)
  {
	  int targetArch = 0;
	  int forcedSyncCall = 0;
	  String value = Driver.getOptionValue("targetArch");
	  if( value != null ) {
		  targetArch = Integer.valueOf(value).intValue();
	  } else {
		  value = System.getenv("OPENARC_ARCH");
		  if( value != null)
		  {
			  targetArch = Integer.valueOf(value).intValue();
		  }
	  }
	  value = Driver.getOptionValue("forceSyncKernelCall");
	  if( value != null ) {
		  forcedSyncCall = Integer.valueOf(value).intValue();
	  }

	  List conflist = call.getConfArguments();

	  if (call.needs_parens)
		  p.print("(");


	  Map<String, String> env = System.getenv();
	  if( targetArch == -1 ) {
		  call.getName().print(p);
		  p.print("<<<");
		  p.print(conflist.get(0));
		  p.print(",");
		  p.print(conflist.get(1));
		  p.print(",");
		  p.print(conflist.get(2));
		  if( conflist.get(3) != null ) {
			  p.print(conflist.get(3));
		  }
		  p.print(">>>");
		  p.print("(");
	  } else {
		  if( targetArch == 4 ) {
			  p.print("mcl_task_set_kernel(");
			  p.print(conflist.get(0)); //add MCL handle
			  p.print(",");
			  p.print(conflist.get(4)); //add src_code pointer
			  p.print(",");

		  } else {
			  p.print("HI_register_kernel_numargs(");
		  }
		  p.print("\"" + call.getName() + "\",");
		  p.println(call.getNumArguments() + ");");
	  }
	for(int i = 0; i < call.getNumArguments(); i++)
	{
		if( targetArch >= 0 ) {
			if( targetArch == 4 ) {
				p.print("mcl_task_set_arg(");
				p.print(conflist.get(0)); //add MCL handle
				p.print(",");

			} else {
				p.print("HI_register_kernel_arg(");
				p.print("\"" + call.getName() + "\",");
			}
			p.print(i + ",");
		}

		VariableDeclaration paramDecl = (VariableDeclaration)call.linkedProcedure.getParameter(i);
		Declarator paramDeclarator = (Declarator)paramDecl.getChildren().get(0);
		List<Specifier> specifierList = paramDecl.getSpecifiers();
		boolean isPointer = false;
/*		if( paramDeclarator instanceof NestedDeclarator ) {
			isPointer = true;
		} else if( paramDeclarator instanceof VariableDeclarator ) {
			VariableDeclarator vDeclr = (VariableDeclarator)paramDeclarator;
			if( SymbolTools.isArray(vDeclr) || SymbolTools.isPointer(vDeclr) ) {
				isPointer = true;
			}
		} else {
			Tools.exit("[ERROR in KernelFunctionCall] Unexpected parameter type");
		}*/
		int symType = AnalysisTools.checkSymbolType((Symbol)paramDeclarator, call, false);
		if( symType == -1 ) {
			Tools.exit("[ERROR in KernelFunctionCall] Unexpected parameter type");
		} else if( symType> 1 ) {
			isPointer = true;
		}

		if( targetArch == 4 ) {
			Expression argExp = null;
			Expression argTrait = null;
			List<Specifier> voidPointer = new ArrayList<Specifier>(2);
			voidPointer.add(Specifier.VOID);
			voidPointer.add(PointerSpecifier.UNQUALIFIED);
			if( isPointer ) {
				argExp = new Typecast(voidPointer, call.getArgument(i).clone());
				p.print(argExp);
				p.print(",");
				p.print(call.getArgSize(i));
				p.print(",");
				if( call.getArgTrait(i).intValue() == trait_readonly ) {
					argTrait = new BinaryExpression(new NameID("MCL_ARG_INPUT"), BinaryOperator.BITWISE_INCLUSIVE_OR, new NameID("MCL_ARG_BUFFER"));
				} else if( call.getArgTrait(i).intValue() == trait_writeonly ) {
					argTrait = new BinaryExpression(new NameID("MCL_ARG_OUTPUT"), BinaryOperator.BITWISE_INCLUSIVE_OR, new NameID("MCL_ARG_BUFFER"));
				} else if( call.getArgTrait(i).intValue() == trait_readwrite ) {
					argTrait = new BinaryExpression(new NameID("MCL_ARG_INPUT"), BinaryOperator.BITWISE_INCLUSIVE_OR, new NameID("MCL_ARG_OUTPUT"));
					argTrait = new BinaryExpression(argTrait, BinaryOperator.BITWISE_INCLUSIVE_OR, new NameID("MCL_ARG_BUFFER"));
				} else {
					argTrait = new NameID("MCL_ARG_BUFFER");
				}
				p.print(argTrait);
			} else {
				argExp = new Typecast(voidPointer, new UnaryExpression(UnaryOperator.ADDRESS_OF, call.getArgument(i).clone()));
				p.print(argExp);
				p.print(",");
				p.print(new SizeofExpression(specifierList));
				p.print(",");
				if( call.getArgTrait(i).intValue() == trait_readonly ) {
					argTrait = new BinaryExpression(new NameID("MCL_ARG_INPUT"), BinaryOperator.BITWISE_INCLUSIVE_OR, new NameID("MCL_ARG_SCALAR"));
				} else if( call.getArgTrait(i).intValue() == trait_writeonly ) {
					argTrait = new BinaryExpression(new NameID("MCL_ARG_OUTPUT"), BinaryOperator.BITWISE_INCLUSIVE_OR, new NameID("MCL_ARG_SCALAR"));
				} else if( call.getArgTrait(i).intValue() == trait_readwrite ) {
					argTrait = new BinaryExpression(new NameID("MCL_ARG_INPUT"), BinaryOperator.BITWISE_INCLUSIVE_OR, new NameID("MCL_ARG_OUTPUT"));
					argTrait = new BinaryExpression(argTrait, BinaryOperator.BITWISE_INCLUSIVE_OR, new NameID("MCL_ARG_SCALAR"));
				} else {
					argTrait = new NameID("MCL_ARG_SCALAR");
				}
				p.print(argTrait);
			}
		} else if (targetArch == -1) {
			if( i > 0 ) {
				p.print(",");
			}
			p.print(call.getArgument(i).clone());
		} else {
			//Special case for OpenCL
			//If use OpenCL, the use of texture memory is not allowed
			if(targetArch != 0)
			{
				if( isPointer ) {
					p.print("sizeof(void*),");
				} else {
					p.print(new SizeofExpression(specifierList));
					p.print(",");
				}
			}
			else
			{
				p.print("sizeof(void*),");
			}
			if(call.getArgument(i) instanceof Typecast)
			{
				p.print(new UnaryExpression(UnaryOperator.ADDRESS_OF, ((Typecast)call.getArgument(i)).getExpression().clone()));
			}
			else
			{
				boolean isConstSymbol = false;
				Expression tArg = call.getArgument(i);
				if( tArg instanceof Identifier ) {
					Symbol tSym = ((Identifier)tArg).getSymbol();
					if( tSym.getTypeSpecifiers().contains(Specifier.CONST) ) {
						isConstSymbol = true;
					}
				}
				if( isConstSymbol ) {
					List<Specifier> types = new ArrayList<Specifier>(2);
					types.add(Specifier.VOID);
					types.add(PointerSpecifier.UNQUALIFIED);
					p.print(new Typecast(types, new UnaryExpression(UnaryOperator.ADDRESS_OF, call.getArgument(i).clone())));
				} else {
					p.print(new UnaryExpression(UnaryOperator.ADDRESS_OF, call.getArgument(i).clone()));
				}
			}
			if( isPointer ) {
				p.print(",1");
			} else {
				p.print(",0");
			}
		}
		if (targetArch >= 0) {
			p.println(");");
		}
	}
	if (targetArch == -1) {
		p.print(")");
	}

	if( targetArch == 4 ) {
		p.print("mcl_exec(");
		p.print(conflist.get(0)); //add MCL handle
		p.print(",");
		p.print(conflist.get(1)); //add global dimensions
		p.print(",");
		p.print(conflist.get(2)); //add thread-block dimensions
		p.print(",");
		p.print(conflist.get(3)); //add MCL flags
		p.println(");");
		if(conflist.get(5) == null)
		{
			p.print("mcl_wait(");
			p.print(conflist.get(0)); //add MCL handle
			p.print(")");
		} else {
			p.print("mcl_acc_set_handle(");
			p.print(conflist.get(5)); //add asyncID
			p.print(",");
			p.print(conflist.get(0)); //add MCL handle
			p.print(")");
		}
	} else if( targetArch >= 0) {
		p.print("HI_kernel_call(");
		p.print("\"" + call.getName() + "\",");
		p.print(conflist.get(0));
		p.print(",");
		p.print(conflist.get(1));
		//argument for shared memory size
		//p.print(",");
		//p.print(conflist.get(2));
		if(conflist.get(3) != null)
		{
			p.print(",");
			p.print(conflist.get(3));
		} else {
			p.print(",");
			p.print("DEFAULT_QUEUE");
		}
		if( conflist.size() > 4 ) {
			IntegerLiteral num_waits = (IntegerLiteral)conflist.get(4);
			long num_waits_value = num_waits.getValue();
			if( num_waits_value > 0 ) {
				p.print(",");
				p.print(num_waits.toString());
				p.print(",");
				p.print("openarc_waits");
			}
		}
		p.println(");");
		if(conflist.get(3) == null)
		{
			if( forcedSyncCall == 0) {
				p.print("HI_synchronize(0)");
			} else {
				p.print("HI_synchronize(1)");
			}
		}
	}
    if (call.needs_parens)
      p.print(")");
  }

	public String toString()
	{
		StringBuilder str = new StringBuilder(80);

		if ( needs_parens )
			str.append("(");

		str.append(getName());
		str.append("<<<");
		List tmp = configuration;
		str.append(PrintTools.listToString(tmp, ", "));
		str.append(">>>");
		str.append("(");
		tmp = (new ChainedList()).addAllLinks(children);
		tmp.remove(0);
		str.append(PrintTools.listToString(tmp, ", "));
		str.append(")");

		if ( needs_parens )
			str.append(")");

		return str.toString();
	}

  public Expression getConfArgument(int n)
  {
    return (Expression)configuration.get(n);
  }

  public List getConfArguments()
  {
    return configuration;
  }

  public void setConfArgument(int n, Expression expr)
  {
    configuration.set(n, expr);
  }

  public void setConfArguments(List args)
  {
    configuration.clear();
    //configuration.addAll(args);
	for(Object o : args)
	{
      Expression expr = null;
      try {
        expr = (Expression)o;
      } catch (ClassCastException e) {
        throw new IllegalArgumentException();
      }
      configuration.add(expr);
	}
  }

  public Integer getArgTrait(int n)
  {
    return (Integer)argTraits.get(n);
  }

  public void addArgTrait(Integer atrait)
  {
    argTraits.add(atrait);
  }

  public Expression getArgSize(int n)
  {
    return (Expression)argSizes.get(n);
  }

  public void addArgSize(Expression aSize)
  {
    argSizes.add(aSize);
  }

/**
   * Overrides the class print method, so that all subsequently
   * created objects will use the supplied method.
   *
   * @param m The new print method.
   */
  static public void setClassPrintMethod(Method m)
  {
    class_print_method = m;
  }

   public void setLinkedProcedure(Procedure proc)
   {
       linkedProcedure = proc;
   }
   
   @Override
   public void accept(TraversableVisitor v) {
     ((OpenACCTraversableVisitor)v).visit(this);
   }
}
