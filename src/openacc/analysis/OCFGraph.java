package openacc.analysis;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Set;

import cetus.analysis.CFGraph;
import cetus.analysis.DFAGraph;
import cetus.analysis.DFANode;
import cetus.analysis.LoopTools;
import cetus.hir.DoLoop;
import cetus.hir.WhileLoop;
import cetus.hir.SymbolTable;
import cetus.hir.Symbolic;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ForLoop;
import cetus.hir.IntegerLiteral;
import cetus.hir.NullStatement;
import cetus.hir.Tools;
import cetus.hir.DataFlowTools;
import cetus.hir.Traversable;
import cetus.hir.AnnotationStatement;
import openacc.hir.ACCAnnotation;

/**
 * OCFGraph creates statement-level control flow graphs with optimization on
 * for-loops; if compiler can know that a for-loop is not a zero-trip loop,  
 * the edge from condition to exit node is removed, and an edge from the body
 * of the loop to exit node is added for more accurate
 * array section analysis.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 */
public class OCFGraph extends CFGraph {

	private static boolean assumeNonZeroTripLoops = false;
	/**
	 * Constructs a OCFGraph object with the given traversable object.
	 * The entry node contains a string "ENTRY" for the key "stmt".
	 *
	 * @param t the traversable object.
	 */
	public OCFGraph(Traversable t)
	{
		this(t, null);
	}

	/**
	 * Constructs a OCFGraph object with the given traversable object and the
	 * IR type whose sub graph is pruned. The resulting control
	 * flow graph does contain the sub graphs for the specified IR type but those
	 * sub graphs are not connected to/from the whole graph. Depending on the
	 * applications of OCFGraph, those isolated sub graphs can be removed from or
	 * reconnected to the whole graph.
	 *
	 * @param t the traversable object.
	 * @param supernode IR type that is pruned.
	 */
	public OCFGraph(Traversable t, Class supernode)
	{
		super(t, supernode);
	}
	
	public OCFGraph(Traversable t, Class supernode, boolean includeDeclarators)
	{
		super(t, supernode, includeDeclarators, null);
	}
	
	static public void setNonZeroTripLoops(boolean assmNonZeroTripLoops) {
		assumeNonZeroTripLoops = assmNonZeroTripLoops;
	}
	// Override buildForLoop() in CFGraph class.
	// If compiler can know that a for-loop is not a zero-trip loop,  
	// the edge from condition to exit node is removed, and an edge
	// from the body of the loop to the exit node is added. 
	protected DFAGraph buildForLoop(ForLoop stmt)
	{
		DFAGraph ret = new DFAGraph();

		CompoundStatement bs = (CompoundStatement)stmt.getBody();

		// Build nodes.
		DFANode init = new DFANode("stmt", stmt);
		DFANode condition = new DFANode("ir", stmt.getCondition());
		DFANode step = new DFANode("ir", stmt.getStep());
		DFANode exit = new DFANode("stmt-exit", stmt);

		// Delay links.
		break_link.push(new ArrayList());
		continue_link.push(new ArrayList());

		// Build subgraph.
		DFAGraph body = buildGraph(bs);
		DFANode lastnode = body.getLast();

		// Put data.
		init.putData("ir", stmt.getInitialStatement());
		init.putData("for-condition", condition);
		init.putData("for-step", step);
		init.putData("for-exit", exit);

		// Keep special string for null condition (should be a unique entity).
		if ( stmt.getCondition() == null )
		{
			condition.putData("ir", new NullStatement());
			//condition.putData("tag", "CONDITION"+System.identityHashCode(stmt));
		}
		condition.putData("true", body.getFirst());
		condition.putData("false", exit);
		condition.putData("back-edge-from", step);

		// Add loop variants
		condition.putData("loop-variants", DataFlowTools.getDefSymbol(stmt));
		if ( !bs.getSymbols().isEmpty() )
		{
			List symbol_exits = new ArrayList();
			symbol_exits.add(bs);
			exit.putData("symbol-exit", symbol_exits);
		}

		// Keep special string for null step (should be a unique entity).
		if ( stmt.getStep() == null )
		{
			step.putData("ir", new NullStatement());
			//step.putData("tag", "STEP"+System.identityHashCode(stmt));
		}
		exit.putData("tag", "FOREXIT");

		// Add edges; init = ret[0] and exit = ret[last].
		ret.addEdge(init, condition);
		ret.addEdge(condition, body.getFirst());
		ret.absorb(body);
		if ( !isJump(body.getLast()) )
			ret.addEdge(body.getLast(), step);
		ret.addEdge(step, condition);

		if( !assumeNonZeroTripLoops ) {
			boolean may_zerotrip_loop = true;
			if ( LoopTools.isCanonical(stmt) ) {
				Expression lb = LoopTools.getLowerBoundExpression(stmt);
				Expression ub = LoopTools.getUpperBoundExpression(stmt);
				Expression iterspace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
				if( iterspace instanceof IntegerLiteral ) {
					long isize = ((IntegerLiteral)iterspace).getValue();
					if( isize > 0 ) {
						may_zerotrip_loop = false;
					}
				}
			} 
			if( may_zerotrip_loop ) {
				ret.addEdge(condition, exit);
			} 
			/*
		else {
			//ret.addEdge(lastnode, exit);
			ret.addEdge(step, exit);
		}
			 */
		}
		ret.addEdge(step, exit);

		// Finalize delayed jumps.
		for ( Object from : break_link.pop() )
			ret.addEdge((DFANode)from, exit);
		for ( Object from : continue_link.pop() )
			ret.addEdge((DFANode)from, step);

		return ret;
	}
	
	  // Build a graph for a do while loop.
	  protected DFAGraph buildDoLoop(DoLoop stmt)
	  {
	    DFAGraph ret = new DFAGraph();

	    CompoundStatement bs = (CompoundStatement)stmt.getBody();

	    // Build nodes.
	    DFANode entry = new DFANode("stmt", stmt);
	    DFANode condition = new DFANode("ir", stmt.getCondition());
	    DFANode exit = new DFANode("stmt-exit", stmt);

	    // Delay links.
	    break_link.push(new ArrayList<DFANode>());
	    continue_link.push(new ArrayList<DFANode>());

	    // Build subgraph.
	    DFAGraph body = buildGraph(bs);

	    // Put data.
	    entry.putData("tag", "DOLOOP");
	    entry.putData("do-condition", condition);
	    entry.putData("do-exit", exit);
	    condition.putData("true", body.getFirst());
	    condition.putData("false", exit);
	    condition.putData("loop-variants", DataFlowTools.getDefSymbol(stmt));
	    //Added by SY Lee.
	    body.getFirst().putData("back-edge-from", condition);

	    if ( !bs.getDeclarations().isEmpty() )
	    {
	      List<SymbolTable> symbol_exits = new ArrayList<SymbolTable>();
	      symbol_exits.add(bs);
	      exit.putData("symbol-exit", symbol_exits);
	    }

	    // Add edges; entry = ret[0] and exit = ret[last]
	    ret.addEdge(entry, body.getFirst());
	    ret.absorb(body);
	    if ( !isJump(body.getLast()) )
	      ret.addEdge(body.getLast(), condition);
	    ret.addEdge(condition, body.getFirst());
	    ret.addEdge(condition, exit);

	    // Finalize delayed jumps.
	    for ( DFANode from : break_link.pop() )
	      ret.addEdge(from, exit);
	    for ( DFANode from : continue_link.pop() )
	      ret.addEdge(from, condition);

	    return ret;
	  }
	  
	    // Build a graph for a while statement.
	    protected DFAGraph buildWhile(WhileLoop stmt) {
	        DFAGraph ret = new DFAGraph();
	        CompoundStatement bs = (CompoundStatement)stmt.getBody();
	        // Build nodes.
	        DFANode entry = new DFANode("stmt", stmt);
	        DFANode condition = new DFANode("ir", stmt.getCondition());
	        DFANode exit = new DFANode("stmt-exit", stmt);
	        // Delay links.
	        break_link.push(new ArrayList<DFANode>(4));
	        continue_link.push(new ArrayList<DFANode>(4));
	        // Build subgraph.
	        DFAGraph body = buildGraph(bs);
	        // Put data.
	        entry.putData("tag", "WHILE");
	        entry.putData("while-condition", condition);
	        entry.putData("while-exit", exit);
	        condition.putData("true", body.getFirst());
	        condition.putData("false", exit);
	        condition.putData("loop-variants", DataFlowTools.getDefSymbol(stmt));
	        condition.putData("back-edge-from", body.getLast());
	        exit.putData("tag", "WHILEEXIT");
	        if (!bs.getDeclarations().isEmpty()) {
	            List<SymbolTable> symbol_exits = new ArrayList<SymbolTable>(4);
	            symbol_exits.add(bs);
	            exit.putData("symbol-exit", symbol_exits);
	        }
	        // Add edges; entry = ret[0] and exit = ret[last].
	        ret.addEdge(entry, condition);
	        ret.absorb(body);
	        ret.addEdge(condition, body.getFirst());
	        if( assumeNonZeroTripLoops ) {
	        	ret.addEdge(body.getLast(), exit);
	        } else {
	        	ret.addEdge(condition, exit);
	        }
	        if (!isJump(body.getLast())) {
	        	ret.addEdge(body.getLast(), condition);
	        }
	        // Finalize delayed jumps.
	        List<DFANode> break_links = break_link.pop();
	        for (int i = 0; i < break_links.size(); i++) {
	            ret.addEdge(break_links.get(i), exit);
	        }
	        List<DFANode> continue_links = continue_link.pop();
	        for (int i = 0; i < continue_links.size(); i++) {
	            ret.addEdge(continue_links.get(i), condition);
	        }
	        return ret;
	    }

	
	protected DFAGraph buildAnnotationStatement(AnnotationStatement stmt)
	{
		DFAGraph ret = new DFAGraph();

		ACCAnnotation omp_annot = stmt.getAnnotation(ACCAnnotation.class, "barrier");
		if ( omp_annot != null && ( ((String)(omp_annot.get("barrier"))).equals("S2P") ||
				((String)(omp_annot.get("barrier"))).equals("P2S") ) )
		{
			DFANode node = new DFANode();
			node.putData("tag", "barrier");
			node.putData("type", omp_annot.get("barrier"));
			node.putData("stmt", stmt);
			node.putData("ir", stmt);
			ret.addNode(node);
		}

		return ret;
	}

	private boolean isEmptyWithSingleSuccNode(DFANode node)
	{
		if ( node.getSuccs().size() == 1 && 
				node.getData(Arrays.asList("stmt","ir","symbol-exit","stmt-exit","super-entry"))==null)
			return true;
		else    
			return false;
	}

	/**
	 * removable_node overrides the super class's removable_node in order not to
	 * remove nodes with barrier tags.
	 */
	protected boolean removable_node(DFANode node)
	{
		if ( isEmptyWithSingleSuccNode(node) )
		{
			String tag = (String)node.getData("tag");
			if ( tag != null && tag.compareTo("barrier")==0 )
			{
				DFANode next_node = null; 
				for ( DFANode nn : (Set<DFANode>)node.getSuccs() ) { next_node = nn; } 
				// if the successor node is also a barrier, then remove the current one.
				// check this condition until the node is not empty or the node has more than 
				// one successor
				while ( isEmptyWithSingleSuccNode(next_node) )
				{       
					String next_tag = (String)next_node.getData("tag");
					if ( next_tag != null && next_tag.compareTo("barrier")==0 )
						return true;
					for ( DFANode nn : (Set<DFANode>)next_node.getSuccs() ) { next_node = nn; } 
				}       
				return false;
			}
			else    
				return true;
		}

		return false;
	}

}
