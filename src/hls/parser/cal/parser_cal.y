%{
#include <stdio.h>
#include "structs.hpp"
#include <iostream>
extern int yylineno;
extern char* yytext;
extern int yylex();
void yyerror(const char *s);
AST* head = new AST;
Context* context = new Context;
%}

%locations

%union
{
    int intValue;
    float floatValue;
    char * stringValue;
}

%token <stringValue> Identifiers
%token <intValue> Numeric_literals

%token Key_at Key_at2 Key_else Key_end Key_if Key_while Key_foreach
%token Key_actor Key_endactor Key_endif Key_action Key_call
%token Key_multi Key_time Key_import Key_all Key_mutable Key_old
%token Key_true Key_false Key_null Key_then Key_let Key_endlet
%token Key_const Key_lambda Key_endlambda Key_var Key_proc Key_endproc
%token Key_do Key_begin Key_function Key_procedure Key_map Key_for
%token Key_in Key_endwhile Key_endforeach Key_choose Key_endchoose
%token Key_endaction Key_guard Key_delay Key_any Key_repeat 
%token Key_initialize Key_endinitialize Key_schedule Key_endschedule
%token Key_fsm Key_regexp Key_priority
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
{
    context->qualID = new QualID;
    ID* tmp = new ID;
    tmp->id = $1;
    tmp->pos.line = @1.first_line;
    tmp->pos.col = @1.first_column;
    context->qualID->addID(tmp);
}
| IDs Delimiters_comma Identifiers
{
    ID* tmp = new ID;
    tmp->id = $3;
    tmp->pos.line = @3.first_line;
    tmp->pos.col = @3.first_column;
    context->qualID->addID(tmp);
}
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
{
    context->actor->timeClause = NULL;
    ID* name = new ID;
    name->id = $3;
    name->pos.line = @3.first_line;
    name->pos.col = @3.first_column;
    context->actor->id = name;
    for(int i = 0; i < context->actorPars.size(); ++i)
        context->actor->addActorPar(context->actorPars[i]);
    for(int i = 0; i < context->typePars.size(); ++i)
        context->actor->addTypePar(context->typePars[i]);
    head->actor = context->actor;
}
;

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
| Key_mutable VarDecltmp Delimiters_semicolon 
{
    VarType* tmp = dynamic_cast<VarType*>(context->varDecls[0]);
    tmp->_mutable = true;
    context->actor->addVarType(tmp);
}
OptActionActor
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
{
    TypePar * tmp = new TypePar;
    tmp->type = NULL;
    ID* name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    tmp->id = name;
    context->typePars.push_back(tmp);
}
| Identifiers Oper_less Type
{
    TypePar * tmp = new TypePar;
    tmp->type = context->type;
    ID* name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    tmp->id = name;
    context->typePars.push_back(tmp);
}
;

ActorPar: Type Identifiers Oper_equal Expression
{
    ActorPar* tmp = new ActorPar;
    tmp->type = context->type;
    tmp->exp = context->exp;
    ID* name = new ID;
    name->id = $2;
    name->pos.line = @2.first_line;
    name->pos.col = @2.first_column;
    tmp->id = name;
    context->actorPars.push_back(tmp);
}
| Identifiers Oper_equal Expression
{
    ActorPar* tmp = new ActorPar;
    tmp->type = NULL;
    tmp->exp = context->exp;
    ID* name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    tmp->id = name;
    context->actorPars.push_back(tmp);
}
| Type Identifiers
{
    ActorPar* tmp = new ActorPar;
    tmp->type = context->type;
    tmp->exp = NULL;
    ID* name = new ID;
    name->id = $2;
    name->pos.line = @2.first_line;
    name->pos.col = @2.first_column;
    tmp->id = name;
    context->actorPars.push_back(tmp);
}
| Identifiers
{
    ActorPar* tmp = new ActorPar;
    tmp->type = NULL;
    tmp->exp = NULL;
    ID* name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    tmp->id = name;
    context->actorPars.push_back(tmp);
}
;

IOSig: PortDecls Oper_eqarrow {context->portType = 1;} PortDecls
| PortDecls Oper_eqarrow
| Oper_eqarrow {context->portType = 1;} PortDecls
| Oper_eqarrow
;

PortDecl: Key_multi Type Identifiers
{
    PortDecl* tmp = new PortDecl;
    tmp->type = context->type;
    ID* name = new ID;
    name->id = $3;
    name->pos.line = @3.first_line;
    name->pos.col = @3.first_column;
    tmp->id = name;
    tmp->multi = true;
    if(context->portType == 0)
        context->actor->addInPort(tmp);
    else
        context->actor->addOutPort(tmp);
}
| Type Identifiers
{
    PortDecl* tmp = new PortDecl;
    tmp->type = context->type;
    ID* name = new ID;
    name->id = $2;
    name->pos.line = @2.first_line;
    name->pos.col = @2.first_column;
    tmp->id = name;
    tmp->multi = false;
    if(context->portType == 0)
        context->actor->addInPort(tmp);
    else
        context->actor->addOutPort(tmp);
}
| Key_multi Identifiers
{
    PortDecl* tmp = new PortDecl;
    ID* name = new ID;
    name->id = $2;
    name->pos.line = @2.first_line;
    name->pos.col = @2.first_column;
    tmp->id = name;
    tmp->multi = true;
    if(context->portType == 0)
        context->actor->addInPort(tmp);
    else
        context->actor->addOutPort(tmp);
}
| Identifiers
{
    PortDecl* tmp = new PortDecl;
    ID* name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    tmp->id = name;
    tmp->multi = false;
    if(context->portType == 0)
        context->actor->addInPort(tmp);
    else
        context->actor->addOutPort(tmp);
}
;

TimeClause: Key_time Type
{
    context->actor->timeClause = context->type;
}
;

Import: SingleImport Delimiters_semicolon
{
    head->addsImport(context->sImport);
}
| GroupImport Delimiters_semicolon
{
    head->addgImport(context->gImport);
}
;

SingleImport: Key_import QualID Oper_equal Identifiers
{
    context->sImport = new SingleImport;
    context->sImport->qualID = context->qualID;
    ID * name = new ID;
    name->id = $4;
    name->pos.line = @4.first_line;
    name->pos.col = @4.first_column;
    context->sImport->alias = name;
}
| Key_import QualID
{
    context->sImport = new SingleImport;
    context->sImport->qualID = context->qualID;
    context->sImport->alias = NULL;
}
;

GroupImport: Key_import Key_all QualID
{
    context->gImport = new GroupImport;
    context->gImport->qualID = context->qualID;
}
;

QualID: Identifiers 
{
    QualID * tmp = new QualID;
    ID * name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    tmp->addID(name);
    context->qualID = tmp;
}
| QualID Delimiters_point Identifiers
{
    ID * name = new ID;
    name->id = $3;
    name->pos.line = @3.first_line;
    name->pos.col = @3.first_column;
    context->qualID->addID(name);
}
;

OptPointID: Delimiters_point Identifiers
{
    context->qualID = new QualID;
    ID * name = new ID;
    name->id = $2;
    name->pos.line = @2.first_line;
    name->pos.col = @2.first_column;
    context->qualID->addID(name);
}
| Delimiters_point Identifiers OptPointID
{
    
    ID * name = new ID;
    name->id = $2;
    name->pos.line = @2.first_line;
    name->pos.col = @2.first_column;
    context->qualID->addID(name);
}
;

Type: Identifiers
{
    context->type = new Type;
    ID* name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    context->type->id = name;
}
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
{
    context->exp = new Exp;
    for(int i = 0; i < context->expContent.size(); ++i)
        context->exp->addToExp(context->expContent[i]);
    context->expContent.clear();
}
| PrimaryExpression Operator
{
    context->expContent.push_back(context->oper);
}
Expression
;

PrimaryExpression: Operator SingleExpression ExpCycleOfIndexFieldFunc
{
    PrimaryExp* tmp = new PrimaryExp;
    tmp->singleExp = context->singleExp;
    tmp->oper = context->oper;
    context->expContent.push_back(tmp);
    //todo: ExpCycleOfIndexFieldFunc
}
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
{
    IDExp* tmp = new IDExp;
    tmp->old = true;
    ID* name = new ID;
    name->id = $2;
    name->pos.line = @2.first_line;
    name->pos.col = @2.first_column;
    tmp->id = name;
    context->singleExp = tmp;
}
| Identifiers
{
    IDExp* tmp = new IDExp;
    tmp->old = false;
    ID* name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    tmp->id = name;
    context->singleExp = tmp;
}
| ExpressionLiteral
| Delimiters_cir_op Expression Delimiters_cir_cl
{
    BracketsExp * tmp = new BracketsExp;
    tmp->exp = context->exp;
    context->singleExp = tmp;
}
| Key_if IfExpression
{
    context->singleExp = context->ifExp;
}
| Key_const Key_lambda LambdaExpression
| Key_lambda LambdaExpression
| Key_proc ProcExpression
| Key_let LetExpression
{
    context->singleExp = context->letExp;
}
| Delimiters_sq_op ListComprehension
| Delimiters_fig_op SetComprehension
| Key_map MapComprehension
;

ExpressionLiteral: Numeric_literals
{
    ExpLiteral* tmp = new ExpLiteral;
    ID* lit = new ID;
    lit->id = std::to_string($1);
    tmp->expLiteral = lit;
}
| Key_true
{
    ExpLiteral* tmp = new ExpLiteral;
    ID* lit = new ID;
    lit->id = "true";
    tmp->expLiteral = lit;
}
| Key_false
{
    ExpLiteral* tmp = new ExpLiteral;
    ID* lit = new ID;
    lit->id = "false";
    tmp->expLiteral = lit;
}
| Key_null
{
    ExpLiteral* tmp = new ExpLiteral;
    ID* lit = new ID;
    lit->id = "null";
    tmp->expLiteral = lit;
}
;

IfExpression: Expression
{
    context->ifExp = new IfExp;
    context->ifExp->ifExp = context->exp;
}
Key_then Expression
{
    context->ifExp->thenExp = context->exp;
}
Key_else Expression IfExpressionEnd
{
    context->ifExp->elseExp = context->exp;
}
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
{
    context->formalPar = new FormalPar;
    context->formalPar->type = context->type;
    ID* name = new ID;
    name->id = $2;
    name->pos.line = @2.first_line;
    name->pos.col = @2.first_column;
    context->formalPar->id = name;
}
| Identifiers
{
    context->formalPar = new FormalPar;
    ID* name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    context->formalPar->id = name;
}
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
OptFormalPars Delimiters_cir_cl OptVarDeclsColon Expression Key_end
{

}
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
{
    context->id = new ID;
    context->id->id = $1;
    context->id->pos.line = @1.first_line;
    context->id->pos.col = @1.first_column;
}
| Type Identifiers
{
    context->id = new ID;
    context->id->id = $2;
    context->id->pos.line = @2.first_line;
    context->id->pos.col = @2.first_column;
}
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

VarDecltmp: OptType OptionalEqExpression
{
    VarType* var = new VarType;
    var->type = context->type;
    var->exp = context->exp;
    var->id = context->id;
    context->varDecls.push_back(var);
}
;

OptionalMutable:
| Key_mutable
;

OptionalEqExpression:
{
    context->exp = NULL;
}
| Equal Expression
;

Equal: Oper_equal
| Oper_assignment
;

ActionTail: ActionHead ActionDoStmts ActionEnd
{
    context->action->actionHead = context->actionHead;
}
; 

ActionDoStmts:
| Key_do Statements
;

ActionEnd: Key_end
| Key_endaction
;

ActionTag: Identifiers
{
    context->actionTag = new ActionTag;
    ID* name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    context->actionTag->qualID = new QualID;
    context->actionTag->qualID->addID(name);
}
| Identifiers
{
    context->qualID = new QualID;
    ID* name = new ID;
    name->id = $1;
    name->pos.line = @1.first_line;
    name->pos.col = @1.first_column;
    context->qualID->addID(name);
}
OptPointID
{
    context->actionTag = new ActionTag;
    context->actionTag->qualID = context->qualID;
}
;

ActionHead: InputPatterns Oper_eqarrow OutputExpressions
ActionHeadGuardExp OptVarVarDecls ActionHeadDelayExp
{
    context->actionHead = new ActionHead;
    for(int i = 0; i < context->inputPattern.size(); i++)
        context->actionHead->inputPatterns.push_back(context->inputPattern[i]);
    for(int i = 0; i < context->outputExp.size(); i++)
        context->actionHead->outputExps.push_back(context->outputExp[i]);
}
;

ActionHeadGuardExp:
| Key_guard Expressions
;

ActionHeadDelayExp:
| Key_delay Expression
;

InputPattern: OptIDcolon Delimiters_sq_op IDs
Delimiters_sq_cl OptRepeatClause OptChannelSelector
{
    InputPattern * tmp = new InputPattern;
    tmp->id = context->id;
    tmp->IDs = context->qualID;
    tmp->repeatClause = context->repeatClause;
    context->inputPattern.push_back(tmp);
}
;

OptIDcolon: {context->id = NULL;}
| Identifiers Delimiters_colon
{
    context->id = new ID;
    context->id->id = $1;
    context->id->pos.line = @1.first_line;
    context->id->pos.col = @1.first_column;
}
;

OptRepeatClause:
{
    context->repeatClause = NULL;
}
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
{
    context->repeatClause = new RepeatClause;
    context->repeatClause->exp = context->exp;
}
;

OutputExpression: OptIDcolon Delimiters_sq_op
Expressions Delimiters_sq_cl OptRepeatClause
OptChannelSelector
{
    OutputExp* tmp = new OutputExp;
    tmp->id = context->id;
    //should be exps
    tmp->repeatClause = context->repeatClause;
    //should be selector
    context->outputExp.push_back(tmp); 
}
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
;

ActionTagTale:
| Delimiters_point Identifiers ActionTagTale
;

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
    context->actor = new Actor;
    yyparse();
    head->print();
    return 0;
}

void yyerror(const char *s)
{
    fprintf(stderr, "error: %s\n", s);
}
