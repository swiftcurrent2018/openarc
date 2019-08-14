/**
 * 
 */
package openacc.analysis;

import java.util.LinkedList;
import java.util.List;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class SubArrayWithConf implements Cloneable {
	private SubArray sArray;
	private List<ConfList> listOfConfList;

	/**
	 * 
	 */
	public SubArrayWithConf(SubArray tSubArray) {
		sArray = tSubArray;
		listOfConfList = new LinkedList<ConfList>();
	}
	
	public SubArrayWithConf(SubArray tSubArray, ConfList tConfList) {
		sArray = tSubArray;
		listOfConfList = new LinkedList<ConfList>();
		listOfConfList.add(tConfList);
	}
	
	public SubArrayWithConf(SubArray tSubArray, List<ConfList> tListOfConfList) {
		sArray = tSubArray;
		listOfConfList = new LinkedList<ConfList>();
		listOfConfList.addAll(tListOfConfList);
	}
	
	public SubArray getSubArray() {
		return sArray;
	}
	
	public void addConfList(ConfList tConfList) {
		listOfConfList.add(tConfList);
	}
	
	public void removeConfList(ConfList tConfList) {
		listOfConfList.remove(tConfList);
	}
	
	public void setConfList(int index, ConfList tConfList) {
    	if( (listOfConfList.size() <= index) || (index < 0) ) throw new IndexOutOfBoundsException();
    	else {
    		listOfConfList.set(index, tConfList);
    	}
	}
	
	public ConfList getConfList(int index) {
    	if( (listOfConfList.size() <= index) || (index < 0) ) throw new IndexOutOfBoundsException();
    	else {
    		return listOfConfList.get(index);
    	}
	}
	
	public List<ConfList> getListOfConfList() {
		return listOfConfList;
	}
	
	public int confSize() {
		return listOfConfList.size();
	}
	
    /** Returns a clone of this object */
    @Override
    public SubArrayWithConf clone() {
    	SubArrayWithConf clonedObj = new SubArrayWithConf(sArray.clone());
    	int tsize = listOfConfList.size();
    	if( tsize > 0 ) {
    		ConfList tConf = null;
    		for(int i=0; i<tsize; i++) {
    			tConf = listOfConfList.get(i);
    			if( tConf == null ) {
    				clonedObj.addConfList(tConf);
    			} else {
    				clonedObj.addConfList(tConf.clone());
    				
    			}
    		}
    	}
    	return clonedObj;
    }
    
    public boolean equals(Object o) {
    	if (o == null || this.getClass() != o.getClass()) {
    		return false;
    	}
    	SubArrayWithConf thatObj = (SubArrayWithConf)o;
    	SubArray thatSArray = thatObj.getSubArray();
    	if( thatSArray == null ) {
    		if( sArray != null ) {
    			return false;
    		}
    	} else if ( (sArray == null) || !(thatSArray.equals(sArray)) ) {
    		return false;
    	}
    	List<ConfList> thatListOfList = thatObj.getListOfConfList();
    	if( thatListOfList == null ) {
    		if( listOfConfList != null ) {
    			return false;
    		} 
    	} else if( (listOfConfList == null) || !(listOfConfList.equals(thatListOfList)) ) {
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
    * Returns a string representation of this object.
    * 
    * @return the string representation.
    */
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        sb.append(sArray);
        int tSize = listOfConfList.size();
        if( tSize > 0 ) {
        	sb.append("::");
        	sb.append(listOfConfList.get(0));
        	for(int i=1; i<tSize; i++) {
        		sb.append("::");
        		sb.append(listOfConfList.get(i));
        	}
        }
        return sb.toString();
    }

}
