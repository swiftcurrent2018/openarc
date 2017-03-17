/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
package cetus.hir;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

/**
 * @author f6l
 *
 */
public class AttributeSpecifier extends Specifier {

	public static class Attribute implements Printable {
		private static HashSet<String> attributes_noarg = 
				new HashSet<String>(Arrays.asList(
						//GNU attributes
						"__gnu_inline__", "__always_inline__", "noreturn",
						//Altera OpenCL attributes
						"blocking", "packed"
						));
		private static HashSet<String> attributes_withargs = 
				new HashSet<String>(Arrays.asList(
						//GNU attributes
						"aligned", 
						//Altera OpenCL attributes
						"reqd_work_group_size", "max_work_group_size",
						"num_simd_work_items", "num_compute_compute_units", "local_mem_size", "buffer_location",
						"depth", "io"));
		
		private String attributeName;
		private List<Expression> argList;

		private void checkAttribute(String str, List<Expression> tList) {
			attributeName = str;
			if( attributes_noarg.contains(str) ) {
				if( (tList != null) && (!tList.isEmpty()) ) {
					PrintTools.println("[WARNING] Attribute, " + str + " can not have arguments: " + tList, 0);
				}
			} else if( attributes_withargs.contains(str) ) {
				if( (tList == null) || (tList.isEmpty()) ) {
					PrintTools.println("[WARNING] Attribute, " + str + " requires arguments but missing.", 0);
				}
			} else {
				//PrintTools.println("[WARNING] Not-supported Attribute: " + str, 2);
			}
		}

		public Attribute (String att) {
			checkAttribute(att, null);
			argList = new ArrayList<Expression>(1);
		}

		public Attribute (String att, List<Expression> tList) {
			checkAttribute(att, tList);
			argList = new ArrayList<Expression>(tList);
		}

		public Attribute (String att, Expression tArg) {
			argList = new ArrayList<Expression>(1);
			argList.add(tArg);
			checkAttribute(att, argList);
		}

		public Attribute clone() {
			List<Expression> nArgList = new ArrayList<Expression>(this.argList.size());
			for(Expression tArg : this.argList) {
				nArgList.add(tArg.clone());
			}
			Attribute nAttr = new Attribute(this.attributeName, nArgList);
			return nAttr;
		}

		public Expression getArg(int i) {
			return (Expression)argList.get(i);
		}
		
		public List<Expression> getArgs() {
			return argList;
		}

		public int getArgSize() {
			return argList.size();
		}

		public void addArg(Expression arg) {
			if (arg.getParent() != null) {
				throw new NotAnOrphanException(this.getClass().getName());
			}
			argList.add(arg);
		}

		public void setArg(int i, Expression arg) {
			if (arg.getParent() != null) {
				throw new NotAnOrphanException(this.getClass().getName());
			}
			argList.set(i, arg);
		}

		/**
		 * Set the list of index expressions.
		 *
		 * @param args A list of Expression.
		 * object.
		 */
		public void setArgs(List<Expression> args) {
			argList.clear();
			if( args != null ) {
				for (Expression t : args) {
					if (t.getParent() != null) {
						throw new NotAnOrphanException(this.getClass().getName());
					}
					argList.add(t);
				}
			}
		}

		public String getAttributeName() {
			return attributeName;
		}

		/* (non-Javadoc)
		 * @see cetus.hir.Printable#print(java.io.PrintWriter)
		 */
		@Override
		public void print(PrintWriter o) {
			o.print(attributeName);
			if( !argList.isEmpty() ) {
				o.print("(");
				PrintTools.printListWithComma(argList, o);
				o.print(")");
			}
		}

		/** Returns a string representation of the specifier. */
		@Override
		public String toString() {
			StringWriter sw = new StringWriter(16);
			print(new PrintWriter(sw));
			return sw.toString();
		}

		public boolean equals(Object o) {
			boolean result = true;
			if( (o == null) || !(o instanceof Attribute) ) {
				result = false;
			} else {
				Attribute nAtt = (Attribute)o;
				if(nAtt.getArgSize() != this.getArgSize()) {
					result = false;
				} else {
					if( !nAtt.argList.equals(this.argList) ) {
						return false;
					}
				}
			}
			return result;
		}
	} //end of Attribute class declaration.
	
	private List<Attribute> attributeList;

	/**
	 * 
	 */
	public AttributeSpecifier() {
		attributeList = new ArrayList<Attribute>(1);
	}

	public AttributeSpecifier(Attribute tAtt) {
		attributeList = new ArrayList<Attribute>(1);
		attributeList.add(tAtt);
	}

	public AttributeSpecifier(List<Attribute> tAttList) {
		attributeList = new ArrayList<Attribute>(tAttList.size());
		attributeList.addAll(tAttList);
	}

	public AttributeSpecifier clone() {
		List<Attribute> nArgList = new ArrayList<Attribute>(this.attributeList.size());
		for(Attribute tArg : this.attributeList) {
			nArgList.add(tArg.clone());
		}
		AttributeSpecifier nAS = new AttributeSpecifier(nArgList);
		return nAS;
	}

	public boolean contains(String name) {
		for (Attribute attr : attributeList) {
			if (attr.getAttributeName().equals(name))
				return true;
		}
		return false;
	}

	public Attribute getAttribute(int i) {
		return (Attribute)attributeList.get(i);
	}
	
	public List<Attribute> getAttributes() {
		return attributeList;
	}

	public int getNumAttributes() {
		return attributeList.size();
	}

	public void addAttribute(Attribute arg) {
		attributeList.add(arg);
	}

	public void setAttribute(int i, Attribute arg) {
		attributeList.set(i, arg);
	}

	/**
	 * Set the list of attributes
	 *
	 * @param args A list of attributes.
	 * object.
	 */
	public void setAttributes(List<Attribute> args) {
		attributeList.clear();
		if( args != null ) {
			for (Attribute t : args) {
				attributeList.add(t);
			}
		}
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		if( !attributeList.isEmpty() ) {
			o.print("__attribute__((");
			PrintTools.printListWithComma(attributeList, o);
			o.print("))");
		}
	}

    public boolean equals(Object o) {
    	boolean result = true;
    	if( (o == null) || !(o instanceof AttributeSpecifier) ) {
    		result = false;
    	} else {
    		AttributeSpecifier nAS = (AttributeSpecifier)o;
    		if(nAS.getNumAttributes() != this.getNumAttributes()) {
    			result = false;
    		} else {
    			if( !nAS.attributeList.equals(this.attributeList) ) {
    				return false;
    			}
    		}
    	}
    	return result;
    }

}
