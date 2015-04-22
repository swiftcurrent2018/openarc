package openacc.hir;

import cetus.hir.TraversableVisitor;

/**
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public interface OpenACCTraversableVisitor extends TraversableVisitor {
  // ASPENDeclaration
    public void visit(ASPENDataDeclaration node);
    public void visit(ASPENKernel node);
    public void visit(ASPENParamDeclaration node);
  // ASPENExpression
    public void visit(ASPENData node);
    public void visit(ASPENParam node);
    public void visit(ASPENResource node);
    public void visit(ASPENTrait node);
  public void visit(ASPENModel node);
  // ASPENStatement
    public void visit(ASPENCompoundStatement node);
    // ASPENControlStatement
      public void visit(ASPENControlExecuteStatement node);
      public void visit(ASPENControlIfStatement node);
      public void visit(ASPENControlIterateStatement node);
      public void visit(ASPENControlKernelCallStatement node);
      public void visit(ASPENControlMapStatement node);
      public void visit(ASPENControlParallelStatement node);
      public void visit(ASPENControlProbabilityStatement node);
      public void visit(ASPENControlSeqStatement node);
    // ASPENExpressionStatement
      public void visit(@SuppressWarnings("deprecation") ASPENExposesExpressionStatement node);
      public void visit(ASPENMemoryExpressionStatement node);
      public void visit(ASPENRequiresExpressionStatement node);
  // Declaration
  // Declarator
  // Expression
    // FunctionCall (concrete supertype)
      public void visit(KernelFunctionCall node);
    // IDExpression
    // Literal
  // Statement
}