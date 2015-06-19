/**
 * 
 */
package openacc.hir;

import cetus.exec.Driver;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.NotAChildException;
import cetus.hir.NotAnOrphanException;
import cetus.hir.Symbol;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import cetus.hir.TraversableVisitor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENModel implements Traversable, ASPENPrintable {

    /** The parent traversable object */
    protected Traversable parent;

    /** The list of children of the statement */
    protected List<Traversable> children;
    
    private IDExpression modelID;
    private String output_filename;
    private Map<Symbol, ASPENParamDeclaration> paramMap;
    private Map<IDExpression, ASPENParamDeclaration> internalParamMap;
    private Map<Symbol, ASPENDataDeclaration> dataMap;
    private Map<IDExpression, ASPENKernel> kernelMap;
    private Set<ASPENModel> importedModels;
    private IDExpression entryFunction; //represents the entry function ID of the input program, which may not be "main".

	/**
	 * 
	 */
	
	public ASPENModel(IDExpression tID) {
		parent = null;
		children = new ArrayList<Traversable>(1);
		paramMap = new HashMap<Symbol, ASPENParamDeclaration>();
		internalParamMap = new HashMap<IDExpression, ASPENParamDeclaration>();
		dataMap = new HashMap<Symbol, ASPENDataDeclaration>();
		kernelMap = new HashMap<IDExpression, ASPENKernel>();
		modelID = tID;
		entryFunction = null;
		output_filename = modelID.toString() + ".aspen";
		//Add a built-in ASPEN parameter.
		ASPENParam tParam = new ASPENParam(ASPENParam.defaultParamID, ASPENParam.defaultParamValue);
		ASPENParamDeclaration defaultParamDecl = new ASPENParamDeclaration(tParam);
		addASPENDeclaration(defaultParamDecl);
	}
	
	public ASPENModel(IDExpression tID, int size) {
		parent = null;
        if (size < 0) {
            children = new ArrayList<Traversable>();
            paramMap = new HashMap<Symbol, ASPENParamDeclaration>();
            internalParamMap = new HashMap<IDExpression, ASPENParamDeclaration>();
            dataMap = new HashMap<Symbol, ASPENDataDeclaration>();
            kernelMap = new HashMap<IDExpression, ASPENKernel>();
        } else {
            children = new ArrayList<Traversable>(size);
            paramMap = new HashMap<Symbol, ASPENParamDeclaration>(size);
            internalParamMap = new HashMap<IDExpression, ASPENParamDeclaration>(size);
            dataMap = new HashMap<Symbol, ASPENDataDeclaration>(size);
            kernelMap = new HashMap<IDExpression, ASPENKernel>(size);
        }
		modelID = tID;
		output_filename = modelID.toString() + ".aspen";
		//Add a built-in ASPEN parameter.
		ASPENParam tParam = new ASPENParam(ASPENParam.defaultParamID, ASPENParam.defaultParamValue);
		ASPENParamDeclaration defaultParamDecl = new ASPENParamDeclaration(tParam);
		addASPENDeclaration(defaultParamDecl);
	}
	
	public IDExpression getModelID() {
		return modelID;
	}
	
	public void setModelID(IDExpression tID) {
		modelID = tID;
		output_filename = modelID.toString() + ".aspen";
	}
	
	public IDExpression getEntryFuncID() {
		return entryFunction;
	}
	
	public void setEntryFuncID(IDExpression tID) {
		entryFunction = tID.clone();
	}
	
	protected void addDeclaration(ASPENDeclaration decl) {
		IDExpression ID = decl.getDeclaredID().clone();
		if( decl instanceof ASPENParamDeclaration ) {
			Symbol IDSym = null;
			if( ID instanceof Identifier ) {
				IDSym = ((Identifier)ID).getSymbol();
			}
			if( IDSym == null ) {
				if( internalParamMap.containsKey(ID) ) {
					System.err.println("[WARNING in ASPENModel.addDeclaration()] duplicate parameter is found: " + ID);
				}
				internalParamMap.put(ID, (ASPENParamDeclaration)decl);
			} else {
				if( paramMap.containsKey(IDSym) ) {
					System.err.println("[WARNING in ASPENModel.addDeclaration()] duplicate parameter is found: " + ID);
				}
				paramMap.put(IDSym, (ASPENParamDeclaration)decl);

			}
		} else if( decl instanceof ASPENDataDeclaration ) {
			Symbol IDSym = null;
			if( ID instanceof Identifier ) {
				IDSym = ((Identifier)ID).getSymbol();
			}
			if( IDSym == null ) {
				System.err.println("[WARNING in ASPENModel.addDeclaration()] Can not find symbol for the following data: " + ID);
			} else {
				if( dataMap.containsKey(IDSym) ) {
					System.err.println("[WARNING in ASPENModel.addDeclaration()] duplicate data is found: " + ID);
				}
				dataMap.put(IDSym, (ASPENDataDeclaration)decl);
			}
		} else if( decl instanceof ASPENKernel ) {
			if( kernelMap.containsKey(ID) ) {
				System.err.println("[WARNING in ASPENModel.addDeclaration()] duplicate kernel is found: " + ID);
			}
			kernelMap.put(ID, (ASPENKernel)decl);
		} else {
			System.err.println("[ERROR in ASPENModel.addDeclaration()] not-supported type of input declaration: " + ID);
			System.exit(0);
		}
	}
	
	protected void removeDeclaration(ASPENDeclaration decl) {
		IDExpression ID = decl.getDeclaredID().clone();
		Symbol IDSym = null;
		if( ID instanceof Identifier ) {
			IDSym = ((Identifier)ID).getSymbol();
		}
		if( decl instanceof ASPENParamDeclaration ) {
			if( IDSym == null ) {
				if( !internalParamMap.containsKey(ID) ) {
					System.err.println("[WARNING in ASPENModel.removeDeclaration()] the parameter does not exist: " + ID);
				}
				internalParamMap.remove(ID);
			} else {
				if( !paramMap.containsKey(IDSym) ) {
					System.err.println("[WARNING in ASPENModel.removeDeclaration()] the parameter does not exist: " + ID);
				}
				paramMap.remove(IDSym);
			}
		} else if( decl instanceof ASPENDataDeclaration ) {
			if( IDSym == null ) {
				System.err.println("[WARNING in ASPENModel.removeDeclaration()] Can not find symbol for the following data: " + ID);
			} else {
				if( !dataMap.containsKey(IDSym) ) {
					System.err.println("[WARNING in ASPENModel.removeDeclaration()] the data does not exist: " + ID);
				}
				dataMap.remove(IDSym);
			}
		} else if( decl instanceof ASPENKernel ) {
			if( !kernelMap.containsKey(ID) ) {
				System.err.println("[WARNING in ASPENModel.removeDeclaration()] the kernel does not exist: " + ID);
			}
			kernelMap.remove(ID);
		} else {
			System.err.println("[ERROR in ASPENModel.removeDeclaration()] not-supported type of input declaration: " + ID);
			System.exit(0);
		}
	}
	
	public void addASPENDeclaration(ASPENDeclaration decl) {
		if( decl instanceof ASPENKernel ) {
			addChild(decl);
		} else {
			ASPENKernel firstKernel = getFirstASPENKernelDeclaration();
			if( firstKernel == null ) {
				addChild(decl);
			} else {
				addASPENDeclarationBefore(firstKernel, decl);
			}
		}
    }

	public void addASPENDeclarationFirst(ASPENDeclaration decl) {
		if( decl instanceof ASPENKernel ) {
			//Let the input ASPENKernel be the first ASPEN kernel.
			ASPENKernel firstKernel = getFirstASPENKernelDeclaration();
			if( firstKernel == null ) {
				addChild(decl);
			} else {
				addASPENDeclarationBefore(firstKernel, decl);
			}
		} else {
			//Let the input ASPEN Param/Data be the first ASPEN declaration.
			addChild(0, decl);
		}
    }
    
    public void addASPENDeclarationBefore(ASPENDeclaration ref_decl, ASPENDeclaration new_decl) {
        int index = Tools.identityIndexOf(children, ref_decl);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        addChild(index, new_decl);
    }
    
    public void addASPENDeclarationAfter(ASPENDeclaration ref_decl, ASPENDeclaration new_decl) {
        int index = Tools.identityIndexOf(children, ref_decl);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        addChild(index+1, new_decl);
    }
    
    public ASPENKernel getFirstASPENKernelDeclaration() {
    	ASPENKernel firstKernel = null;
    	for( Traversable child : children ) {
    		if( child instanceof ASPENKernel ) {
    			firstKernel = (ASPENKernel)child;
    			break;
    		}
    	}
    	return firstKernel;
    }
    
    public void removeASPENDeclaration(ASPENDeclaration decl) {
    	removeChild(decl);
    }
    
    public Set<IDExpression> getInternalParamIDs() {
    	return internalParamMap.keySet();
    }
    
    public Set<Symbol> getParamSymbols() {
    	return paramMap.keySet();
    }
    
    public Set<Symbol> getDataSymbols() {
    	return dataMap.keySet();
    }
    
    public ASPENDeclaration getASPENDeclaration(Symbol InSym) {
    	ASPENDeclaration aDecl = paramMap.get(InSym);
    	if( aDecl == null ) {
    		aDecl = dataMap.get(InSym);
    	}
    	return aDecl;
    }
    
    public ASPENParamDeclaration getParamDeclaration(IDExpression ID) {
    	Symbol IDSym = null;
    	if( ID instanceof Identifier ) {
    		IDSym = ((Identifier)ID).getSymbol();
    	}
    	if( IDSym == null ) {
    		return internalParamMap.get(ID);
    	} else {
    		return paramMap.get(IDSym);
    	}
    }
    
    public ASPENDataDeclaration getDataDeclaration(IDExpression ID) {
    	Symbol IDSym = null;
    	if( ID instanceof Identifier ) {
    		IDSym = ((Identifier)ID).getSymbol();
    	}
    	if( IDSym == null ) {
    		return null;
    	} else {
    		return dataMap.get(IDSym);
    	}
    }
    
    public ASPENKernel getKernel(IDExpression ID) {
    	IDExpression tID = ID;
    	if( ID.equals(entryFunction) ) {
    		tID = new NameID("main");
    	}
    	return kernelMap.get(tID);
    }
    
    public ASPENParamDeclaration removeParam(IDExpression pID) {
    	ASPENParamDeclaration paramDecl = null;
    	Symbol IDSym = null;
    	if( pID instanceof Identifier ) {
    		IDSym = ((Identifier)pID).getSymbol();
    	}
    	if( IDSym == null ) {
    		paramDecl = internalParamMap.get(pID);
    	} else {
    		paramDecl = paramMap.get(IDSym);
    	}
    	if( paramDecl != null ) {
    		removeASPENDeclaration(paramDecl);
    	}
    	return paramDecl;
    }
    
    public ASPENParamDeclaration removeParamSymbol(Symbol IDSym) {
    	ASPENParamDeclaration paramDecl = null;
    	if( IDSym != null ) {
    		paramDecl = paramMap.get(IDSym);
    	}
    	if( paramDecl != null ) {
    		removeASPENDeclaration(paramDecl);
    	}
    	return paramDecl;
    }
    
    public ASPENDataDeclaration removeData(IDExpression pID) {
    	ASPENDataDeclaration dataDecl = null;
    	Symbol IDSym = null;
    	if( pID instanceof Identifier ) {
    		IDSym = ((Identifier)pID).getSymbol();
    	}
    	if( IDSym != null ) {
    		dataDecl = dataMap.get(IDSym);
    	}
    	if( dataDecl != null ) {
    		removeASPENDeclaration(dataDecl);
    	}
    	return dataDecl;
    }
    
    public ASPENDataDeclaration removeDataSymbol(Symbol IDSym) {
    	ASPENDataDeclaration dataDecl = null;
    	if( IDSym != null ) {
    		dataDecl = dataMap.get(IDSym);
    	}
    	if( dataDecl != null ) {
    		removeASPENDeclaration(dataDecl);
    	}
    	return dataDecl;
    }
    
    public boolean containsParam(IDExpression ID) {
    	Symbol IDSym = null;
    	if( ID instanceof Identifier ) {
    		IDSym = ((Identifier)ID).getSymbol();
    	}
    	if( IDSym == null ) {
    		return internalParamMap.containsKey(ID);
    	} else {
    		return paramMap.containsKey(IDSym);
    	}
    }
    
    public boolean containsData(IDExpression ID) {
    	Symbol IDSym = null;
    	if( ID instanceof Identifier ) {
    		IDSym = ((Identifier)ID).getSymbol();
    	}
    	if( IDSym == null ) {
    		return false;
    	} else {
    		return dataMap.containsKey(IDSym);
    	}
    }
    
    public boolean containsKernel(IDExpression ID) {
    	IDExpression tID = ID;
    	if( ID.equals(entryFunction) ) {
    		tID = new NameID("main");
    	}
    	return kernelMap.containsKey(tID);
    }
    
    public boolean containsASPENDeclaration(ASPENDeclaration decl) {
    	IDExpression ID = decl.getDeclaredID().clone();
    	Symbol IDSym = null;
    	if( ID instanceof Identifier ) {
    		IDSym = ((Identifier)ID).getSymbol();
    	}
    	if( decl instanceof ASPENParamDeclaration ) {
    		if( IDSym == null ) {
    			return internalParamMap.containsKey(ID);
    		} else {
    			return paramMap.containsKey(IDSym);
    		}
    	} else if( decl instanceof ASPENDataDeclaration ) {
    		if( IDSym == null ) {
    			return false;
    		} else {
    			return dataMap.containsKey(IDSym);
    		}
    	} else if( decl instanceof ASPENKernel ) {
    		return kernelMap.containsKey(ID);
    	} else {
    		return false;
    	}
    }
	
    public List<Traversable> getChildren() {
        return children;
    }

    public Traversable getParent() {
        return parent;
    }

    /**
    * Removes a specific child of this model;
    *
    * @param child The child to remove.
    */
    public void removeChild(Traversable child) {
        int index = Tools.identityIndexOf(children, child);
        if (index == -1) {
            throw new NotAChildException();
        }
        child.setParent(null);
        children.remove(index);
        removeDeclaration((ASPENDeclaration)child);
    }

    public void setChild(int index, Traversable t) {
        if (t == null || index < 0 || index >= children.size() || !(t instanceof ASPENDeclaration)) {
            throw new IllegalArgumentException();
        }
        if (t.getParent() != null) {
            throw new NotAnOrphanException();
        }
        // Detach the old child
        Traversable oldChild = children.get(index);
        if (oldChild != null) {
            oldChild.setParent(null);
        }
        removeDeclaration((ASPENDeclaration)oldChild);
        children.set(index, t);
        t.setParent(this);
        addDeclaration((ASPENDeclaration)t);
    }

    public void setParent(Traversable t) {
        parent = t;
    }

    /**
    * Inserts the specified traversable object at the end of the child list.
    *
    * @param t the traversable object to be inserted.
    * @throws IllegalArgumentException if <b>t</b> is null.
    * @throws NotAnOrphanException if <b>t</b> has a parent.
    */
    protected void addChild(Traversable t) {
        if ((t == null) || !(t instanceof ASPENDeclaration) ) {
            throw new IllegalArgumentException("invalid child inserted.");
        }
        if (t.getParent() != null) {
            throw new NotAnOrphanException(this.getClass().getName());
        }
        children.add(t);
        t.setParent(this);
        addDeclaration((ASPENDeclaration)t);
    }

    /**
    * Inserts the specified traversable object at the specified position.
    *
    * @param t the traversable object to be inserted.
    * @throws IllegalArgumentException if <b>t</b> is null or index is
    * out-of-bound.
    * @throws NotAnOrphanException if <b>t</b> has a parent.
    */
    protected void addChild(int index, Traversable t) {
        if (t == null || index < 0 || index > children.size() || !(t instanceof ASPENDeclaration)) {
            throw new IllegalArgumentException("invalid child inserted.");
        }
        if (t.getParent() != null) {
            throw new NotAnOrphanException(this.getClass().getName());
        }
        children.add(index, t);
        t.setParent(this);
        addDeclaration((ASPENDeclaration)t);
    }
    
	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print("model ");
		o.print(modelID);
		o.println(" {");
        ASPENPrintTools.printlnList(children, o);
		o.print("}");
	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.print("model ");
		o.print(modelID);
		o.println(" {");
        ASPENPrintTools.printlnList(children, o);
		o.println("}");
	}

    /** Returns a string representation of the ASPEN Statement */
    @Override
    public String toString() {
        StringWriter sw = new StringWriter(40);
        printASPENModel(new PrintWriter(sw));
        return sw.toString();
    }
    
    /** Returns a string representation of the ASPEN Statement */
    public String toASPENString() {
        StringWriter sw = new StringWriter(40);
        printASPENModel(new PrintWriter(sw));
        return sw.toString();
    }
    
    /**
    * Write ASPEN model to an output file.
    *
    * @throws FileNotFoundException if a file could not be opened. 
    */
    public void print() throws IOException {
        String outdir = Driver.getOptionValue("outdir");
        // make sure the output directory exists
        File dir = null;
        try {
            dir = new File(outdir);
            if (!dir.exists()) {
                if (!dir.mkdir()) {
                    throw new IOException("mkdir failed");
                }
            }
        } catch(IOException e) {
            System.err.println("cetus: could not create output directory, "+e);
            Tools.exit(1);
        } catch(SecurityException e) {
            System.err.println("cetus: could not create output directory, "+e);
            Tools.exit(1);
        }
        
        //Write the model to the output file
        File to = new File(dir, output_filename);
        try {
            // default buffer size 8192 (characters).
            PrintWriter o = new PrintWriter(
                    new BufferedWriter(new FileWriter(to)));
             printASPENModel(o);
             o.close();
        } catch(IOException e) {
            throw new FileNotFoundException(e.getMessage());
        }
    }

    @Override
    public void accept(TraversableVisitor v) {
      ((OpenACCTraversableVisitor)v).visit(this);
    }
}
