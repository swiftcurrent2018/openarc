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
  private Procedure linkedProcedure;

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

	  if (call.needs_parens)
		  p.print("(");

    //call.getName().print(p);
    //p.print("<<<");
    //List tmp = call.getConfArguments();
    //PrintTools.printListWithComma(tmp, p);
    //p.print(">>>");
    //p.print("(");
    Map<String, String> env = System.getenv();
    p.print("HI_register_kernel_numargs(");
	p.print("\"" + call.getName() + "\",");
	p.println(call.getNumArguments() + ");");
	for(int i = 0; i < call.getNumArguments(); i++)
	{
    	p.print("HI_register_kernel_arg(");
		p.print("\"" + call.getName() + "\",");
		p.print(i + ",");
        //Special case for OpenCL

        //If use OpenCL, the use of texture memory is not allowed
        if(targetArch != 0)
        {
            VariableDeclaration paramDecl = (VariableDeclaration)call.linkedProcedure.getParameter(i);
            Declarator paramDeclarator = (Declarator)paramDecl.getChildren().get(0);
            List<Specifier> specifierList = paramDecl.getSpecifiers();
            //[DEBUG] Modified by Seyong Lee; below checking is incorrect.
/*            if(paramDeclarator.getArraySpecifiers().size() > 0 || paramDeclarator.getSpecifiers().size() > 0 || specifierList.size() > 1 )
            {
                //This param is an array or pointer
                p.print("sizeof(void*),");
            }
            else if(specifierList.size() == 1)
            {
                p.print(new SizeofExpression(specifierList));
                p.print(",");
            }
            else
            {
                 Tools.exit("[Function Call] Unspecify parameter");
            }*/
            boolean isPointer = false;
            if( paramDeclarator instanceof NestedDeclarator ) {
            	isPointer = true;
            } else if( paramDeclarator instanceof VariableDeclarator ) {
            	VariableDeclarator vDeclr = (VariableDeclarator)paramDeclarator;
            	if( SymbolTools.isArray(vDeclr) || SymbolTools.isPointer(vDeclr) ) {
            		isPointer = true;
            	}
            } else {
                 Tools.exit("[ERROR in KernelFunctionCall] Unexpected parameter type");
            }
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
		p.println(");");
	}
    //p.print(")");

    p.print("HI_kernel_call(");
	p.print("\"" + call.getName() + "\",");
	p.print(call.getConfArgument(0));
	p.print(",");
	p.print(call.getConfArgument(1));
	if(call.getConfArgument(3) != null)
    {
        p.print(",");
        p.print(call.getConfArgument(3));
    } else {
        p.print(",");
        p.print("DEFAULT_QUEUE");
	}
	p.println(");");
	if(call.getConfArgument(3) == null)
	{
		p.print("HI_synchronize()");
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
