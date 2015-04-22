package cetus.hir;

import java.util.*;

/**
* Iterates over Traversable objects in depth-first order.
* 
* <p>
* The iteration starts from the root object that was specified in the
* constructor. By default, all objects are visited before their children
* (pre-order).
* </p>
* 
* <p>
* [Post-order traversal and tracing added by Joel E. Denny]
* </p>
*/
public class DepthFirstIterator<E extends Traversable> extends IRIterator<E> {

    private Vector<Traversable> stack;

    // Prune set was replaced by prune list to avoid allocation of set iterator,
    // which requires extra memory; this is not a good idea since every IR
    // object needs to allocate such an iterator. By keeping this as an array
    // list there is no need for allocating iterator.
    private List<Class<? extends Traversable>> prune_list;

    // Types of objects that should not be traversed in the default order.
    private List<Class<? extends Traversable>> reverse_list = null;
    // For each object in stack, the corresponding Boolean here indicates
    // whether that object's children have already been visited, which can only
    // happen if the object is being traversed in post-order.
    private Vector<Boolean> stack_post = null;
    // Whether the default traversal order has been changed from pre to post.
    private boolean default_order_is_post = false;

    // For each object in stack, the prefix to print in the trace of the
    // traversal.
    private Vector<String> stack_prefix = null;
    // The prefix to print for the root object.
    private String stack_prefix_init;
    // The prefix that would be printed for the most recent object's children.
    private String prefix_children = null;

    /**
     * Same as {@link #DepthFirstIterator(Traversable, boolean)} but
     * post-order traversals and tracing are disabled.
     */
    public DepthFirstIterator(Traversable init) {
      this(init, false, null);
    }

    /**
    * Creates a new iterator with the specified initial traversable object and
    * the optional pruned types.
    *
    * @param init The first object to visit.
    * @param enablePostOrder whether to set up internal data structures
    *                        required for post-order traversals (see
    *                        {@link #setDefaultOrderToPost} and
    *                        {@link #reverseOrderFor} to activate post-order)
    * @param tracePrefix the text to print to stderr before each object in a
    *                    trace of the traversal, or null if tracing should be
    *                    disabled
    */
    public DepthFirstIterator(Traversable init, boolean enablePostOrder,
                              String tracePrefix) {
        super(init);
        stack = new Vector<Traversable>();
        stack.add(init);
        prune_list = new ArrayList<Class<? extends Traversable>>(4);
        if (enablePostOrder) {
            reverse_list = new ArrayList<Class<? extends Traversable>>();
            stack_post = new Vector<Boolean>();
            stack_post.add(false);
        }
        if (tracePrefix != null) {
            stack_prefix = new Vector<String>();
            stack_prefix_init = tracePrefix;
            stack_prefix.add(stack_prefix_init);
        }
    }

    public boolean hasNext() {
        return !stack.isEmpty();
    }

    @SuppressWarnings("unchecked")
    public E next() {
        boolean starting_post;
        Traversable t;
        String prefix = null;
        do {
            starting_post = false;
            try {
                t = stack.remove(0);
            } catch(ArrayIndexOutOfBoundsException e) {
                // catching ArrayIndexOutofBoundsException, as remove method
                // throws this exception
                throw new NoSuchElementException();
            }
            if (stack_prefix != null) {
                prefix = stack_prefix.remove(0);
                prefix_children = prefix + "^";
            }
            if (stack_post != null) {
                if (stack_post.remove(0)) {
                    // Finishing post-order traversal of t.
                    if (stack_prefix != null)
                        System.err.println("visit post: " + prefix
                                           + (prefix.isEmpty()?"":" ")
                                           + t.getClass());
                    return (E)t;
                }
                if (default_order_is_post != needsReverseOrder(t.getClass())) {
                    // Starting post-order traversal of t.
                    starting_post = true;
                    stack.add(0, t);
                    stack_post.add(0, true);
                    if (stack_prefix != null)
                        stack_prefix.add(0, prefix);
                }
            }
            boolean foundChild = false;
            if (t.getChildren() != null &&
                (prune_list.isEmpty() || !needsPruning(t.getClass()))) {
                List<Traversable> children = t.getChildren();
                // iterator or for-each statement will allocate a new object, which
                // is not memory-efficient. However this conventional style for
                // loop is efficient only for collection types that support random
                // access.
                for (int j = 0, i = 0; j < children.size(); j++) {
                    Traversable child = children.get(j);
                    if (child != null) {
                        stack.add(i, child);
                        if (stack_post != null)
                            stack_post.add(i, false);
                        if (stack_prefix != null)
                            stack_prefix.add(i, prefix+(starting_post?"v":"^"));
                        ++i;
                        foundChild = true;
                    }
                }
            }
            // If no children were pushed, avoid saying "pre" or "post" in the
            // trace.  In the case of pruning and iterating children
            // externally, the effective order will be pre, so saying "post"
            // would be especially confusing.
            if (!foundChild) {
                if (starting_post) {
                    stack.remove(0);
                    stack_post.remove(0);
                    if (stack_prefix != null)
                        stack_prefix.remove(0);
                }
                if (stack_prefix != null) {
                    String kind = "    ";
                    if (t.getChildren() != null
                        && (!prune_list.isEmpty()
                            && needsPruning(t.getClass())))
                        kind = "prnd";
                    System.err.println("visit " + kind + ": " + prefix
                                       + (prefix.isEmpty()?"":" ")
                                       + t.getClass());
                }
                return (E)t;
            }
        } while (starting_post);
        if (stack_prefix != null)
            System.err.println("visit pre : " + prefix
                               + (prefix.isEmpty()?"":" ") + t.getClass());
        return (E)t;
    }

    private boolean needsPruning(Class<? extends Traversable> c) {
        for (int i = 0; i < prune_list.size(); i++) {
            if (prune_list.get(i).isAssignableFrom(c)) {
                return true;
            }
        }
        return false;
    }

    private boolean needsReverseOrder(Class<? extends Traversable> c) {
        for (int i = 0; i < reverse_list.size(); i++) {
            if (reverse_list.get(i).isAssignableFrom(c)) {
                return true;
            }
        }
        return false;
    }

    /**
    * Disables traversal from an object having the specified type. For example,
    * if traversal reaches an object with type <b>c</b>, it does not visit the
    * children of the object.
    *
    * @param c the object type to be pruned on.
    */
    public void pruneOn(Class<? extends E> c) {
        prune_list.add(c);
    }

    /**
    * Selects the non-default traversal order for an object having the
    * specified type (or a subtype). Does not affect whether its children in
    * the IR are traversed in pre or post-order relative to their children.
    *
    * @param c the object type to visited in the non-default order.
    * 
    * @throw UnsupportedOperationException if post-order traversal was not
    *                                      enabled during construction
    */
    public void reverseOrderFor(Class<? extends E> c) {
        if (reverse_list == null)
            throw new UnsupportedOperationException(
                          "post-order traversal selected but not enabled");
        reverse_list.add(c);
    }
    
    /**
     * Sets the default traversal order to post-order.
     * 
     * @throw UnsupportedOperationException if post-order traversal was not
     *                                      enabled during construction
     */
    public void setDefaultOrderToPost() {
        if (stack_post == null)
            throw new UnsupportedOperationException(
                          "post-order traversal selected but not enabled");
        default_order_is_post = true;
    }

    /**
    * Returns a linked list of objects of Class c in the IR
    *
    * @param c the object type to be collected.
    * @return the collected list.
    */
    @SuppressWarnings("unchecked")
    public <T extends Traversable> List<T> getList(Class<T> c) {
        List<T> ret = new ArrayList<T>();
        while (hasNext()) {
            Object o = next();
            if (c.isInstance(o)) {
                ret.add((T) o);
            }
        }
        return ret;
    }

    /**
    * Returns a set of objects of Class c in the IR
    *
    * @param c the object type to be collected.
    * @return the collected set.
    */
    @SuppressWarnings("unchecked")
    public <T extends Traversable> Set<T> getSet(Class<T> c) {
        HashSet<T> set = new HashSet<T>();
        while (hasNext()) {
            Object obj = next();
            if (c.isInstance(obj)) {
                set.add((T)obj);
            }
        }
        return set;
    }

    /**
    * Resets the iterator by setting the current position to the root object.
    * The pruned types, the types with reversed traversal order, and the
    * default traversal order are not cleared.
    */
    public void reset() {
        stack.clear();
        stack.add(root);
        if (stack_post != null) {
            stack_post.clear();
            stack_post.add(false);
        }
        if (stack_prefix != null) {
            stack_prefix.clear();
            stack_prefix.add(stack_prefix_init);
            prefix_children = null;
        }
    }

    /**
    * Unlike the <b>reset</b> method, <b>clear</b> method also clears the
    * pruned types and the types with reversed traversal order, and it sets
    * the default traversal order back to pre.
    */
    public void clear() {
        reset();
        prune_list.clear();
        if (reverse_list != null)
            reverse_list.clear();
        default_order_is_post = false;
    }

    /**
     * Get the trace prefix for the most recent object's children. When pruning
     * on an object, this can be passed to the constructor of a new iterator
     * for its child objects.
     * @return the trace prefix or null if tracing was not activated during
     *         construction
     */
    public String getChildTracePrefix() {
        return prefix_children;
    }
}
