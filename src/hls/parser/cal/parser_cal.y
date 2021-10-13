%{
#include <stdio.h>

extern int yylineno;
extern char* yytext;
extern int yylex();
void yyerror(const char *s);
%}

%token Key_at Key_at2 Key_else Key_end Key_if Key_while Key_foreach
%token Key_actor Key_endactor Key_endif Key_action Key_call
%token Key_multi Key_time Key_import Key_all Key_mutable Key_old
%token Key_true Key_false Key_null Key_then Key_let Key_endlet
%token Key_const Key_lambda Key_endlambda Key_var Key_proc Key_endproc
%token Key_do Key_begin Key_function Key_procedure Key_map Key_for
%token Key_in Key_endwhile Key_endforeach Key_choose Key_endchoose
%token Key_endaction Key_guard Key_delay Key_any Key_repeat 
%token Key_initialize Key_endinitialize Key_schedule Key_endschedule
%token Key_fsm Key_regexp Key_priority Identifiers Numeric_literals
%token Oper_eqarrow Oper_doublearrow Oper_assignment
%token Oper_vert Oper_arrow 
%token Delimiters_point Delimiters_comma Delimiters_cir_op 
%token Delimiters_cir_cl Delimiters_sq_op Delimiters_sq_cl
%token Delimiters_fig_op Delimiters_fig_cl Delimiters_colon
%token Delimiters_semicolon   
%left Oper_less Oper_equal Oper_more
%left Oper_plus Oper_minus
%left Oper_div Oper_star

%%

Start:
| Actor
;

Imports: Import
| Import Imports
;

TypePars: TypePar
| TypePar Delimiters_comma TypePars
;

ActorPars: ActorPar
| ActorPar Delimiters_semicolon ActorPars
;

PortDecls: PortDecl
| PortDecl Delimiters_comma PortDecls
;

FormalPars: FormalPar
| FormalPar Delimiters_comma FormalPars
;

Expressions: Expression
| Expression Delimiters_comma Expressions
;

IDs: Identifiers
| Identifiers Delimiters_comma IDs
;

Statements: Statement
| Statement  Statements
;

Types: Type
| Type Delimiters_comma Types
;

InputPatterns: InputPattern
| InputPattern Delimiters_comma InputPatterns
;

OutputExpressions: OutputExpression
| OutputExpression Delimiters_comma OutputExpressions
;

Actor: OptImports Key_actor Identifiers OptTypePars
Delimiters_cir_op ActorPars Delimiters_cir_cl IOSig
OptTimeClause Delimiters_colon OptActionActor Key_schedule ActionSchedule OptActionActor ActorEnd
| OptImports Key_actor Identifiers OptTypePars
Delimiters_cir_op ActorPars Delimiters_cir_cl IOSig
OptTimeClause Delimiters_colon OptActionActor ActorEnd

OptImports:
| Imports
;

OptTypePars:
| Delimiters_sq_op TypePars Delimiters_sq_cl
;

OptTimeClause:
| TimeClause
;

ActorEnd: Key_end
| Key_endactor
;

OptActionActor:
| Key_mutable VarDecltmp Delimiters_semicolon OptActionActor
| FunDecl OptActionActor
| ProcDecl OptActionActor
| WrapActionInit OptActionActor
| Identifiers OptionalEqExpression Delimiters_semicolon OptActionActor
| Identifiers Identifiers OptionalEqExpression Delimiters_semicolon OptActionActor
| Identifiers Delimiters_point OptActionTag WrapActionInit OptActionActor
| Identifiers Delimiters_colon WrapActionInit OptActionActor
| Identifiers Delimiters_sq_op TypePars Delimiters_sq_cl 
Identifiers OptionalEqExpression Delimiters_semicolon OptActionActor
| Identifiers Delimiters_cir_op OptTypeAttrs Delimiters_cir_cl 
Identifiers OptionalEqExpression Delimiters_semicolon OptActionActor
| Delimiters_sq_op OptTypes Oper_doublearrow TypeTail 
Identifiers OptionalEqExpression Delimiters_semicolon OptActionActor
| Key_priority PriorityOrder OptActionActor
;

WrapActionInit: Key_initialize InitializationActionTail
| Key_action ActionTail
;

OptActionTag: Identifiers Delimiters_colon 
| Identifiers Delimiters_point OptActionTag
;

TypePar: Identifiers 
| Identifiers Oper_less Type
;

ActorPar: Type Identifiers Oper_equal Expression 
| Identifiers Oper_equal Expression 
| Type Identifiers 
| Identifiers 
;

IOSig: PortDecls Oper_eqarrow PortDecls
| PortDecls Oper_eqarrow
| Oper_eqarrow PortDecls
| Oper_eqarrow
;

PortDecl: Key_multi Type Identifiers
| Type Identifiers
| Key_multi Identifiers
| Identifiers
;

TimeClause: Key_time Type
;

Import: SingleImport Delimiters_semicolon
| GroupImport Delimiters_semicolon
;

SingleImport: Key_import QualID Oper_equal Identifiers
| Key_import QualID
;

GroupImport: Key_import Key_all QualID
;

QualID: Identifiers
| Identifiers Delimiters_point Identifiers OptPointID
| Identifiers Delimiters_point Identifiers
;

OptPointID: Delimiters_point Identifiers
| Delimiters_point Identifiers OptPointID
;

Type: Identifiers
| Identifiers Delimiters_sq_op TypePars Delimiters_sq_cl
| Identifiers Delimiters_cir_op OptTypeAttrs Delimiters_cir_cl
| Delimiters_sq_op OptTypes Oper_doublearrow TypeTail
;

TypeTail: Type Delimiters_sq_cl
| Delimiters_sq_cl
;

OptTypeAttrs:
| TypeAttr CycleTypeAttr

CycleTypeAttr:
| Delimiters_comma TypeAttr CycleTypeAttr
;

OptTypes:
| Types
;

TypeAttr: Identifiers Delimiters_colon Type
| Identifiers Oper_equal Expression
;

Operator: Oper_star
| Oper_plus
| Oper_minus
| Oper_div
| Oper_more
| Oper_less
| Oper_equal
;

Expression: PrimaryExpression
| PrimaryExpression Operator Expression
;

PrimaryExpression: Operator SingleExpression ExpCycleOfIndexFieldFunc
| SingleExpression ExpCycleOfIndexFieldFunc
;

ExpCycleOfIndexFieldFunc: 
| Delimiters_cir_op OptionalExp ExpCycleOfIndexFieldFunc
| Delimiters_sq_op Expression Delimiters_sq_cl ExpCycleOfIndexFieldFunc
| Delimiters_point Identifiers ExpCycleOfIndexFieldFunc
;

OptionalExp: Delimiters_cir_cl
| Expression Delimiters_cir_cl
;

SingleExpression: Key_old Identifiers
| Identifiers
| ExpressionLiteral
| Delimiters_cir_op Expression Delimiters_cir_cl
| Key_if IfExpression
| Key_const Key_lambda LambdaExpression
| Key_lambda LambdaExpression
| Key_proc ProcExpression
| Key_let LetExpression
| Delimiters_sq_op ListComprehension
| Delimiters_fig_op SetComprehension
| Key_map MapComprehension
;

ExpressionLiteral: Numeric_literals
| Key_true
| Key_false
| Key_null
;

IfExpression: Expression Key_then Expression 
Key_else Expression IfExpressionEnd
;

IfExpressionEnd: Key_end
| Key_endif
;

LetExpression: LocalVarDecls Delimiters_colon Expression LetExpressionEnd
;

LetExpressionEnd: Key_end
| Key_endlet
;

LambdaExpression: Delimiters_cir_op
OptFormalPars Delimiters_cir_cl OptDoubleArrowType
OptVarVarDecls Delimiters_colon Expression LambdaExpressionEnd
;

OptFormalPars:
| FormalPars
;

OptDoubleArrowType:
| Oper_doublearrow Type
;

OptVarVarDecls:
| Key_var LocalVarDecls
;

LambdaExpressionEnd: Key_end
| Key_endlambda
;

FormalPar: Type Identifiers 
| Identifiers
;

ProcExpression:  Delimiters_cir_op OptFormalPars
Delimiters_cir_cl OptVarVarDecls ProcExpressionStart
OptionalStmt ProcExpressionEnd
;

ProcExpressionStart: Key_do
| Key_begin
;

OptionalStmt:
| Statement OptionalStmt
;

ProcExpressionEnd: Key_end
| Key_endproc
;

FunDecl: Key_function Identifiers Delimiters_cir_op
OptFormalPars Delimiters_cir_cl OptVarDeclsColon OptionalStmt Key_end
;

OptVarDeclsColon:
| Key_var LocalVarDecls Delimiters_colon
;

LocalVarDecls: LocalVarDecl
| LocalVarDecl Delimiters_semicolon LocalVarDecls
;

LocalVarDecl: OptionalMutable VarDecltmp
| FunDecl
| ProcDecl
;

ProcDecl: Key_procedure Identifiers Delimiters_cir_op
OptFormalPars Delimiters_cir_cl OptVarDeclsColon OptionalStmt Key_end
;

SetComprehension: SetComprehensionExp Delimiters_fig_cl
;

SetComprehensionExp: 
| Expressions Delimiters_colon Generator
| Expressions
;

ListComprehension: ListComprehensionExp Delimiters_sq_cl
;

ListComprehensionExp:
| Expressions OptGenerators Oper_vert Expression
| Expressions OptGenerators
;

OptGenerators:
| Delimiters_colon Generator
;

MapComprehension: Delimiters_fig_op MapComprMapping Delimiters_fig_cl
;

MapComprMapping:
| Mappings OptGenerators
;

Mappings: Mapping
| Mapping Delimiters_comma Mappings
;

Mapping: Expression Oper_arrow Expression
;

Generator: Key_for OptType OptCommaIDs Key_in
Expression CycleGenExpression
;

OptType: Identifiers
| Type Identifiers
;

OptCommaIDs:
| Delimiters_comma IDs
;

CycleGenExpression:
| Delimiters_comma Expression CycleGenExpression
| Delimiters_comma Generator
;

Statement: AssignmentStmt
| Key_call CallStmt
| Key_begin BlockStmt
| Key_if IfStmt
| Key_while WhileStmt
| Key_foreach ForeachStmt
| Key_choose ChooseStmt
;

AssignmentStmt: LHS Oper_assignment
Expression Delimiters_semicolon
;

LHS: Identifiers
| LHS Index
| LHS FieldRef
;

Index: Delimiters_sq_op Expressions Delimiters_sq_cl
| Delimiters_sq_op Delimiters_sq_cl
;

FieldRef: Delimiters_point Identifiers
;

CallStmt: Expression Delimiters_colon Delimiters_cir_op CallStmtOptExp
 Delimiters_semicolon
;

CallStmtOptExp: Delimiters_cir_cl
| Expressions Delimiters_cir_cl
;

BlockStmt: BlockStmtLocalVars OptionalStmt Key_end
;

BlockStmtLocalVars:
| Key_var LocalVarDecls Key_do
;

IfStmt: Expression Key_then OptionalStmt IfStmtElse IfStmtEnd
;

IfStmtElse:
| Key_else OptionalStmt
;

IfStmtEnd: Key_end
| Key_endif
;

WhileStmt: Expression OptVarVarDecls Key_do SomeStmts
WhileStmtEnd
;


SomeStmts:
| Statements
;

WhileStmtEnd: Key_end
| Key_endwhile
;

ForeachStmt: ForeachGenerator OptVarVarDecls
Key_do SomeStmts ForeachStmtEnd
;

ForeachStmtEnd: Key_end
| Key_endforeach
;

ForeachGenerator: OptionalType
SomeIDs Key_in Expression CycleForeachGenExpression
;

OptionalType: Identifiers OptionalTypeTail 
| Delimiters_sq_op TypePars Delimiters_sq_cl Identifiers
| Delimiters_cir_op OptTypeAttrs Delimiters_cir_cl Identifiers
;

OptionalTypeTail: 
| Identifiers
| Delimiters_sq_op TypePars Delimiters_sq_cl Identifiers
| Delimiters_cir_op OptTypeAttrs Delimiters_cir_cl Identifiers
;

SomeIDs:
| Delimiters_comma Identifiers SomeIDs
;

CycleForeachGenExpression:
| Delimiters_comma  Expression CycleForeachGenExpression
| Delimiters_comma Key_foreach ForeachGenerator
;

ChooseStmt: ChooseGenerator OptVarVarDecls
Key_do SomeStmts ChooseStmtElse ChooseStmtEnd
;

ChooseStmtElse:
| Key_else ChooseStmtLocalDo SomeStmts
;

ChooseStmtLocalDo:
| OptVarVarDecls Key_do
;

ChooseStmtEnd: Key_end
| Key_endchoose
;

ChooseGenerator: OptionalType SomeIDs 
Key_in Expression CycleChooseGenExpression
;

CycleChooseGenExpression:
| Delimiters_comma Expression CycleChooseGenExpression
| Delimiters_comma Key_choose ChooseGenerator
;

VarDecltmp: OptionalType OptionalEqExpression
;

OptionalMutable:
| Key_mutable
;

OptionalEqExpression:
| Equal Expression
;

Equal: Oper_equal
| Oper_assignment
;

ActionTail: ActionHead ActionDoStmts ActionEnd
; 

ActionDoStmts:
| Key_do Statements
;

ActionEnd: Key_end
| Key_endaction
;

ActionTag: Identifiers
| Identifiers OptPointID
;

ActionHead: InputPatterns Oper_eqarrow OutputExpressions
ActionHeadGuardExp OptVarVarDecls ActionHeadDelayExp
;

ActionHeadGuardExp:
| Key_guard Expressions
;

ActionHeadDelayExp:
| Key_delay Expression
;

InputPattern: OptIDcolon Delimiters_sq_op IDs
Delimiters_sq_cl OptRepeatClause OptChannelSelector
;

OptIDcolon:
| Identifiers Delimiters_colon
;

OptRepeatClause:
| RepeatClause
;

OptChannelSelector:
| ChannelSelector
;

ChannelSelector: Key_at Expression
| Key_at2 Expression
| OptionalAT2 Key_any
| OptionalAT2 Key_all
;

OptionalAT2:
| Key_at2
;

RepeatClause: Key_repeat Expression
;

OutputExpression: OptIDcolon Delimiters_sq_op
Expressions Delimiters_sq_cl OptRepeatClause
OptChannelSelector
;

InitializationActionTail: 
InitializerHead ActionDoStmts InitializationActionEnd
;

InitializationActionEnd: Key_end
| Key_endinitialize
;

InitializerHead: Oper_eqarrow OutputExpressions
ActionHeadGuardExp OptVarVarDecls ActionHeadDelayExp
;

ActionSchedule:  ScheduleFSM
|  Key_regexp ScheduleRegExp
;

ScheduleFSM:  OptionalFSM Identifiers Delimiters_colon
CycleStateTransition ScheduleFSMend
;

OptionalFSM:
| Key_fsm
;

CycleStateTransition:
| StateTransition Delimiters_semicolon CycleStateTransition
;

ScheduleFSMend: Key_end
| Key_endschedule
;

StateTransition: Identifiers Delimiters_cir_op ActionTags
Delimiters_cir_cl Oper_doublearrow Identifiers StateTransitionAlternative
;

StateTransitionAlternative:
| Oper_vert Delimiters_cir_op ActionTags Delimiters_cir_cl
Oper_doublearrow Identifiers StateTransitionAlternative
;

ActionTags: ActionTag
| ActionTag Delimiters_comma ActionTags
;

ScheduleRegExp: RegExp ScheduleFSMend
;

RegExp: Identifiers ActionTagTale Tail1
| Delimiters_cir_op RegExp Delimiters_cir_cl Tail1
| Delimiters_sq_op RegExp Delimiters_sq_cl Tail1

Tail1:Oper_vert RegExp
| RegExp2
| Oper_star Stars

RegExp2: 
| Identifiers ActionTagTale RegExp2
| Delimiters_cir_op RegExp Delimiters_cir_cl RegExp2
| Delimiters_sq_op RegExp Delimiters_sq_cl RegExp2

Stars:
| Oper_star Stars

ActionTagTale:
| Delimiters_point Identifiers ActionTagTale

PriorityOrder: CyclePriorInequal Key_end
;

CyclePriorInequal:
| PriorityInequality Delimiters_semicolon CyclePriorInequal
;

PriorityInequality: ActionTag Oper_more ActionTag CycleMoreActionTag
;

CycleMoreActionTag:
| Oper_more ActionTag CycleMoreActionTag
;

%%

int main()
{
    yyparse();

    return 0;
}

void yyerror(const char *s)
{
    fprintf(stderr, "error: %s\n", s);
}