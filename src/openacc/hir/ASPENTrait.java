/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import cetus.hir.PrintTools;
import cetus.hir.Expression;
import cetus.hir.Traversable;
import cetus.hir.TraversableVisitor;

/**
 * @author lee222<lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 */
public class ASPENTrait extends ASPENExpression {

    private static HashMap<String, ASPENTrait> trait_map =
            new HashMap<String, ASPENTrait>(64);
    private static HashSet<String> parallelism_traits = 
    		new HashSet<String>(Arrays.asList("MasterWorker", "HierarchicalMasterWorker",
    				"2DGrid", "3DGrid", "2DWavefront", "2DWavefront"));
    private static HashSet<String> data_traits = 
    		new HashSet<String>(Arrays.asList("Array", "Matrix", "3DVolume"
    				));
    //[FIXME] integer trait is treated as a flops trait; this should be fixed when Aspen supports Integer 
    //operations as a separate resource.
    private static HashSet<String> flops_traits = 
    		new HashSet<String>(Arrays.asList("sp", "dp", "simd", "fmad", "complex", "integer"));
    private static HashSet<String> memory_traits = 
    		new HashSet<String>(Arrays.asList("stride", "random"));
    private static HashSet<String> message_traits = 
    		new HashSet<String>(Arrays.asList("barrier", "broadcast", "scatter", "gather", "allScatter",
    				"allGather", "reduction", "allReduce", "allToAll", "scan", "exScan"));
    private static HashSet<String> intracomm_traits = 
    		new HashSet<String>(Arrays.asList("copy", "copyin", "copyout", "pcopy", "pcopyin", "pcopyout"));
    
    public static enum TraitType { ParallelismTrait, DataTrait, FlopsTrait, MemoryTrait, MessageTrait,
    	IntraCommTrait}
    

    private String trait;
	private TraitType tType;
	/**
	 * 
	 */
	public ASPENTrait(String str) {
		checkTrait(str);
	}
	
	public ASPENTrait(String str, List<Expression> args) {
		checkTrait(str);
		if( args != null ) {
			setArgs(args);
		}
	}
	
	private void checkTrait(String str) {
		if( parallelism_traits.contains(str) ) {
			tType = ASPENTrait.TraitType.ParallelismTrait;
			trait = str;
		} else if( data_traits.contains(str) ) {
			tType = ASPENTrait.TraitType.DataTrait;
			trait = str;
		} else if( flops_traits.contains(str) ) {
			tType = ASPENTrait.TraitType.FlopsTrait;
			trait = str;
		} else if( memory_traits.contains(str) ) {
			tType = ASPENTrait.TraitType.MemoryTrait;
			trait = str;
		} else if( message_traits.contains(str) ) {
			tType = ASPENTrait.TraitType.MessageTrait;
			trait = str;
		} else if( intracomm_traits.contains(str) ) {
			tType = ASPENTrait.TraitType.IntraCommTrait;
			trait = str;
		} else {
			System.err.println("Unknown ASPEN Trait: " + str);
			System.err.println("Exit the ASPEN Model Generator.");
			System.exit(0);
		}
	}
	
	public boolean equals(Object o) {
		if( super.equals(o) ) {
			ASPENTrait obj = (ASPENTrait)o;
			if( !trait.equals(obj.trait) ) {
				return false;
			} 
			if( !tType.equals(obj.tType) ) {
				return false;
			}
			return true;
		} else {
			return false;
		}
	}
	
	public ASPENTrait clone() {
		ASPENTrait nTrait = (ASPENTrait)super.clone();
		nTrait.trait = this.trait;
		nTrait.tType = this.tType;
		return nTrait;
	}
	
	public Expression getArg(int i) {
		return (Expression)children.get(i);
	}
	
	public int getArgSize() {
		return children.size();
	}
	
	public void setArg(int i, Expression arg) {
		setChild(i, arg);
	}
	
	public void addArg(Expression arg) {
		addChild(arg);
	}
	
	public TraitType getTraitType() {
		return tType;
	}
	
	public String getTrait() {
		return trait;
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print(trait);
		if( !children.isEmpty() ) {
			o.print("(");
			PrintTools.printListWithComma(children, o);
			o.print(")");
		}
	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.print(trait);
		if( !children.isEmpty() ) {
			o.print("(");
			PrintTools.printListWithComma(children, o);
			o.print(")");
		}
	}
	
    /**
    * Set the list of index expressions.
    *
    * @param args A list of Expression.
    * @throws NotAnOrphanException if an element of <b>args</b> has a parent
    * object.
    */
    public void setArgs(List args) {
        children.clear();
		if( args != null ) {
        	for (Object o : args) {
            	addChild((Traversable)o);
        	}
		}
    }

    @Override
    public void accept(TraversableVisitor v) {
      ((OpenACCTraversableVisitor)v).visit(this);
    }
}
