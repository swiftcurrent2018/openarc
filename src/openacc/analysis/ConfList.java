/**
 * 
 */
package openacc.analysis;

import java.util.LinkedList;
import java.util.List;

import cetus.hir.Expression;

/**
 * Represent configuration list used in OpenARC transform directives.
 * (e.g., <1, 3, 2> or <M, N, P>
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ConfList implements Cloneable {
	private List<Expression> configList;
	
	/**
	 * 
	 */
	public ConfList() {
		configList = new LinkedList<Expression>();
	}
	
	public ConfList(List<Expression> inList) {
		configList = new LinkedList<Expression>();
		configList.addAll(inList);
	}
	
    /** Returns a clone of the ConfList*/
    @Override
    public ConfList clone() {
    	ConfList clonedConfList = new ConfList();
    	int tsize = configList.size();
    	if( tsize > 0 ) {
    		for(int i=0; i<tsize; i++) {
    			clonedConfList.addConfig(configList.get(i).clone());
    		}
    	}
    	return clonedConfList;
    }
    
    public boolean equals(Object o) {
    	if (o == null || this.getClass() != o.getClass()) {
    		return false;
    	}
    	List<Expression> oConfigList = ((ConfList)o).getConfigList();
    	if (oConfigList == null) {
    		if (configList != null) {
    			return false;
    		}
    	} else if( configList.size() != oConfigList.size() ) {
    		return false;
    	} else if( !configList.equals(oConfigList) ) {
    			return false;
    	}
    	return true;
    }
    
    public int hashCode() {
        String s = toString();
        int h = 0;
        for (int i = 0; i < s.length(); i++) {
            h = 31 * h + s.charAt(i);
        }
        return h;
    }
    
    /**
    * Returns a string representation of the ConfList.
    * 
    * @return the string representation.
    */
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        int tSize = configList.size();
        sb.append("[");
        if( tSize > 0 ) {
        	sb.append(configList.get(0));
        	for(int i=1; i<tSize; i++) {
        		sb.append(", ");
        		sb.append(configList.get(i));
        	}
        }
        sb.append("]");
        return sb.toString();
    }
    
	public void addConfig(Expression tExp) {
		configList.add(tExp);
	}
	
	public void removeConfig(Expression tExp) {
		configList.remove(tExp);
	}
	
	public void setConfig(int index, Expression tExp) {
    	if( (configList.size() <= index) || (index < 0) ) throw new IndexOutOfBoundsException();
    	else {
    		configList.set(index, tExp);
    	}
	}
	
	public Expression getConfig(int index) {
    	if( (configList.size() <= index) || (index < 0) ) throw new IndexOutOfBoundsException();
    	else {
    		return configList.get(index);
    	}
	}
	
	public void setConfigList (List<Expression> tconflist ) {
		configList.clear();
		if( tconflist != null ) {
			configList.addAll(tconflist);
		}
	}
	
	public List<Expression> getConfigList() {
		return configList;
	}

}
