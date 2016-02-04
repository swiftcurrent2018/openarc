package cetus.hir;

import cetus.hir.Traversable;

/**
 * Visitor pattern visitor interface for {@link Traversable} class hierarchy.
 * Use of a visitor class within a library requires caution.  See below for
 * details.
 *
 * <p>
 * For every concrete descendant class of {@link Traversable}, there must be
 * one {@code visit} method here whose only argument is of that class.  Each
 * such class must directly implement (not inherit) an {@code accept} method
 * that simply calls {@code v.visit(this)}.
 * </p>
 * 
 * <p>
 * When extending the {@link Traversable} hierarchy in new packages (such as
 * {@link openacc.hir}), create a new interface (such as
 * {@link openacc.hir.OpenACCTraversableVisitor}) extending
 * {@link TraversableVisitor} with {@code visit} methods for the new package's
 * concrete descendant classes of {@link Traversable}. In each of those
 * classes, the formal parameter type for the {@code accept} method must be
 * {@link TraversableVisitor}.  However, the {@code accept} method must cast
 * the parameter to the new interface type before calling the {@code visit}
 * method.  For example:
 * </p>
 * 
 * <pre>public void accept(TraversableVisitor v) {
 *   ((OpenACCTraversableVisitor)v).visit(this);
 * }</pre>
 * 
 * <p>
 * Nodes from the new package cannot be visited by visitors that are not
 * descendants of the new interface. Otherwise, a
 * {@link java.lang.ClassCastException} will be thrown at runtime. To extend
 * such a visitor class to handle nodes from the new package, define a new
 * visitor class that extends the old visitor class and implements the new
 * interface, and use the new visitor class instead of the old one.
 * </p>
 * 
 * <p>
 * Thus, in general in a library, if some library user might attempt to apply a
 * library component to a tree containing {@link Traversable} nodes for which
 * the library was not originally designed, it is ill advised for the library
 * component to employ a visitor class within it.  However, it's fine for the
 * library to expose any such visitor class directly so the library's user can
 * extend the visitor class with a new visitor class and use that instead, as
 * described above.  Alternatively, a library component can avoid the visitor
 * pattern altogether and instead extend the {@link Traversable} hierarchy with
 * virtual methods.  That approach is simpler but loses the visitor pattern's
 * clean separation of concerns as well as compiler errors for new node types
 * that don't have explicit handlers.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public interface TraversableVisitor {
  // Declaration
    public void visit(@SuppressWarnings("deprecation") AccessLevel node);
    public void visit(AnnotationDeclaration node);
    public void visit(ClassDeclaration node);
    public void visit(Enumeration node);
    public void visit(LinkageSpecification node);
    public void visit(Namespace node);
    public void visit(NamespaceAlias node);
    public void visit(PreAnnotation node);
    public void visit(Procedure node);
    public void visit(TemplateDeclaration node);
    public void visit(UsingDeclaration node);
    public void visit(UsingDirective node);
    public void visit(VariableDeclaration node);
  // Declarator
    public void visit(NestedDeclarator node);
    public void visit(ProcedureDeclarator node);
    public void visit(VariableDeclarator node);
  // Expression
    public void visit(AlignofExpression node);
    public void visit(ArrayAccess node);
    public void visit(BinaryExpression node);
      public void visit(AccessExpression node);
      public void visit(AssignmentExpression node);
    public void visit(CommaExpression node);
    public void visit(CompoundLiteral node);
    public void visit(ConditionalExpression node);
    public void visit(DeleteExpression node);
    public void visit(FunctionCall node);
    // IDExpression
      public void visit(DestructorID node);
      public void visit(Identifier node);
      public void visit(NameID node);
      public void visit(OperatorID node);
      public void visit(QualifiedID node);
      public void visit(TemplateID node);
    public void visit(InfExpression node);
    // Literal
      public void visit(BooleanLiteral node);
      public void visit(CharLiteral node);
      public void visit(EscapeLiteral node);
      public void visit(FloatLiteral node);
      public void visit(IntegerLiteral node);
      public void visit(StringLiteral node);
    public void visit(MinMaxExpression node);
    public void visit(NewExpression node);
    public void visit(NVLGetRootExpression node);
    public void visit(NVLAllocNVExpression node);
    public void visit(OffsetofExpression node);
    public void visit(RangeExpression node);
    public void visit(SizeofExpression node);
    public void visit(SomeExpression node);
    public void visit(StatementExpression node);
    public void visit(ThrowExpression node);
    public void visit(Typecast node);
    public void visit(UnaryExpression node);
    public void visit(VaArgExpression node);
  public void visit(Initializer node);
    public void visit(ConstructorInitializer node);
    public void visit(ListInitializer node);
    public void visit(ValueInitializer node);
  public void visit(Program node);
  // Statement
    public void visit(AnnotationStatement node);
    public void visit(BreakStatement node);
    public void visit(Case node);
    public void visit(CompoundStatement node);
      public void visit(ExceptionHandler node);
    public void visit(ContinueStatement node);
    public void visit(DeclarationStatement node);
    public void visit(Default node);
    public void visit(DoLoop node);
    public void visit(ExpressionStatement node);
    public void visit(ForLoop node);
    public void visit(GotoStatement node);
    public void visit(IfStatement node);
    public void visit(Label node);
    public void visit(NullStatement node);
    public void visit(ReturnStatement node);
    public void visit(SwitchStatement node);
    public void visit(WhileLoop node);
  public void visit(TranslationUnit node);
}