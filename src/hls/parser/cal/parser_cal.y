%{
#include <stdio.h>
#include "structs.hpp"
#include <iostream>
#include <stdlib.h>
extern int yylineno;
extern char* yytext;
extern int yylex();
void yyerror(const char *s);
AST* head = new AST;
Context* context = new Context;

void expHandler()
{
    Exp* op2 = context->exp.back();
    context->exp.pop_back();
    Exp* op1 = context->exp.back();
    context->exp.pop_back();
    int op = context->opers.back();
    context->opers.pop_back();
    Exp* tmp = new Exp(op, op1, op2);
    context->exp.push_back(tmp);
};
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
    ID* tmp = new ID($1, @1.first_line, @1.first_column);
    context->qualID->addID(tmp);
}
| IDs Delimiters_comma Identifiers
{
    ID* tmp = new ID($3, @3.first_line, @3.first_column);
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
    ID* name = new ID($3, @3.first_line, @3.first_column);
    context->actor->id = name;
    for(int i = 0; i < context->actorPars.size(); ++i)
        context->actor->addActorPar(context->actorPars[i]);
    for(int i = 0; i < context->typePars.size(); ++i)
        context->actor->addTypePar(context->typePars[i]);
    head->actor = context->actor;
    for(int i = 0; i < context->varDecls.size(); ++i)
        context->actor->addVarDecl(context->varDecls[i]);
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

VarDeclmute:
Key_mutable VarDecltmp Delimiters_semicolon
{
    VarType* var = dynamic_cast<VarType*>(context->varDecls.back());
    var->_mutable = true;
    context->varDecls.pop_back();
    context->varDecls.push_back(var);
}

VarDeclunmute:
Identifiers OptionalEqExpression Delimiters_semicolon
{
    Exp* exp = context->exp.back();
    context->exp.pop_back();
    ID* name = new ID($1, @1.first_line, @1.first_column);
    VarType* var = new VarType(false, name, NULL, exp);
    context->varDecls.push_back(var);
}
;

VarDeclunmutetype:
Identifiers Identifiers OptionalEqExpression Delimiters_semicolon
{
    Exp* exp = context->exp.back();
    context->exp.pop_back();
    ID* name = new ID($2, @2.first_line, @2.first_column);
    Type* type = new Type(new ID($1, @1.first_line, @1.first_column));
    VarType* var = new VarType(false, name, type, exp);
    context->varDecls.push_back(var);
}
;

VarDeclunmutetypepars:
Identifiers Delimiters_sq_op TypePars Delimiters_sq_cl 
Identifiers OptionalEqExpression Delimiters_semicolon
;


OptActionActor:
| VarDeclmute OptActionActor
| FunDecl OptActionActor
| ProcDecl OptActionActor
| WrapActionInit OptActionActor
| VarDeclunmute OptActionActor
| VarDeclunmutetype OptActionActor
| Identifiers Delimiters_point OptActionTag WrapActionInit OptActionActor
| Identifiers Delimiters_colon WrapActionInit OptActionActor
| VarDeclunmutetypepars OptActionActor
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
    ID* name = new ID($1, @1.first_line, @1.first_column);
    TypePar* tmp = new TypePar(name);
    context->typePars.push_back(tmp);
}
| Identifiers Oper_less Type
{
    ID* name = new ID($1, @1.first_line, @1.first_column);
    TypePar* tmp = new TypePar(name, context->type);
    context->typePars.push_back(tmp);
}
;

ActorPar: Type Identifiers Oper_equal Expression
{
    Exp* exp = context->exp.back();
    context->exp.pop_back();
    ID* name = new ID($2, @2.first_line, @2.first_column);
    ActorPar* tmp = new ActorPar(name, exp, context->type);
    context->actorPars.push_back(tmp);
}
| Identifiers Oper_equal Expression
{
    Exp* exp = context->exp.back();
    context->exp.pop_back();
    ID* name = new ID($1, @1.first_line, @1.first_column);
    ActorPar* tmp = new ActorPar(name, exp);
    context->actorPars.push_back(tmp);
}
| Type Identifiers
{
    ID* name = new ID($2, @2.first_line, @2.first_column);
    ActorPar* tmp = new ActorPar(name, NULL, context->type);
    context->actorPars.push_back(tmp);
}
| Identifiers
{
    ID* name = new ID($1, @1.first_line, @1.first_column);
    ActorPar* tmp = new ActorPar(name);
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
    ID* name = new ID($3, @3.first_line, @3.first_column);
    PortDecl* tmp = new PortDecl(true, name, context->type);
    if(context->portType == 0)
        context->actor->addInPort(tmp);
    else
        context->actor->addOutPort(tmp);
}
| Type Identifiers
{
    ID* name = new ID($2, @2.first_line, @2.first_column);
    PortDecl* tmp = new PortDecl(false, name, context->type);
    if(context->portType == 0)
        context->actor->addInPort(tmp);
    else
        context->actor->addOutPort(tmp);
}
| Key_multi Identifiers
{
    ID* name = new ID($2, @2.first_line, @2.first_column);
    PortDecl* tmp = new PortDecl(true, name);
    if(context->portType == 0)
        context->actor->addInPort(tmp);
    else
        context->actor->addOutPort(tmp);
}
| Identifiers
{
    ID* name = new ID($1, @1.first_line, @1.first_column);
    PortDecl* tmp = new PortDecl(false, name);
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
    ID* name = new ID($4, @4.first_line, @4.first_column);
    context->sImport = new SingleImport(context->qualID, name);
}
| Key_import QualID
{
    context->sImport = new SingleImport(context->qualID);
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
    ID* name = new ID($1, @1.first_line, @1.first_column);
    tmp->addID(name);
    context->qualID = tmp;
}
| QualID Delimiters_point Identifiers
{
    ID* name = new ID($3, @3.first_line, @3.first_column);
    context->qualID->addID(name);
}
;

OptPointID: Delimiters_point Identifiers
{
    context->qualID = new QualID;
    ID* name = new ID($2, @2.first_line, @2.first_column);
    context->qualID->addID(name);
}
| Delimiters_point Identifiers OptPointID
{
    ID* name = new ID($2, @2.first_line, @2.first_column);
    context->qualID->addID(name);
}
;

Type: Identifiers
{
    ID* name = new ID($1, @1.first_line, @1.first_column);
    context->type = new Type(name);
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

Operatorlvl0: Oper_plus
{
    context->opers.push_back(ADD);
}
| Oper_minus
{
    context->opers.push_back(SUB);
}
;

Operatorlvl1: Oper_star
{
    context->opers.push_back(MUL);
}
| Oper_div
{
    context->opers.push_back(DIV);
}
;

Operatorlvl2: Oper_more
{
    context->opers.push_back(MORE);
}
| Oper_less
{
    context->opers.push_back(LESS);
}
| Oper_equal
{
    context->opers.push_back(EQ);
}

Expression: Expressionlvl0
| Expression Operatorlvl0 Expressionlvl0
{
    expHandler();
}
;

Expressionlvl0: Expressionlvl1
| Expressionlvl0 Operatorlvl1 Expressionlvl1
{
    expHandler();
}
;

Expressionlvl1: PrimaryExpression
| Expressionlvl1 Operatorlvl2 PrimaryExpression
{
    expHandler();
}
;

PrimaryExpression: Operatorlvl0 SingleExpression ExpCycleOfIndexFieldFunc
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
    ID* id = new ID($2, @2.first_line, @2.first_column);
    Exp* tmp = new Exp(IDEXPOLD, NULL, NULL, NULL, id);
    context->exp.push_back(tmp);
}
| Identifiers
{
    ID* id = new ID($1, @1.first_line, @1.first_column);
    Exp* tmp = new Exp(IDEXPNOOLD, NULL, NULL, NULL, id);
    context->exp.push_back(tmp);
}
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
{
    ID* id = new ID(to_string($1), @1.first_line, @1.first_column);
    Exp* tmp = new Exp(NOOP, NULL, NULL, NULL, id);
    context->exp.push_back(tmp);
}
| Key_true
{
    ID* id = new ID("1", @1.first_line, @1.first_column);
    Exp* tmp = new Exp(NOOP, NULL, NULL, NULL, id);
    context->exp.push_back(tmp);
}
| Key_false
{
    ID* id = new ID("0", @1.first_line, @1.first_column);
    Exp* tmp = new Exp(NOOP, NULL, NULL, NULL, id);
    context->exp.push_back(tmp);
}
| Key_null
{
    ID* id = new ID("0", @1.first_line, @1.first_column);
    Exp* tmp = new Exp(NOOP, NULL, NULL, NULL, id);
    context->exp.push_back(tmp);
}
;

IfExpression: Expression
Key_then Expression
Key_else Expression IfExpressionEnd
{
    Exp* op3 = context->exp.back();
    context->exp.pop_back();
    Exp* op2 = context->exp.back();
    context->exp.pop_back();
    Exp* op1 = context->exp.back();
    context->exp.pop_back();
    Exp* tmp = new Exp(ITE, op1, op2, op3);
    context->exp.push_back(tmp);
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
    ID* name = new ID($2, @2.first_line, @2.first_column);
    context->formalPar = new FormalPar(name, context->type);
}
| Identifiers
{
    ID* name = new ID($1, @1.first_line, @1.first_column);
    context->formalPar = new FormalPar(name);
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
    context->id = new ID($1, @1.first_line, @1.first_column);
    context->type = NULL;
}
| Type Identifiers
{
    context->id = new ID($2, @2.first_line, @2.first_column);
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
    Exp* exp = context->exp.back();
    context->exp.pop_back();
    VarType* var = new VarType(false, context->id, context->type, exp);
    context->varDecls.push_back(var);
}
;

OptionalMutable:
| Key_mutable
;

OptionalEqExpression:
{
    context->exp.push_back(NULL);
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
    ID* name = new ID($1, @1.first_line, @1.first_column);
    context->actionTag->qualID = new QualID;
    context->actionTag->qualID->addID(name);
}
| Identifiers
{
    context->qualID = new QualID;
    ID* name = new ID($1, @1.first_line, @1.first_column);
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
    context->id = new ID($1, @1.first_line, @1.first_column);
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
    context->repeatClause->exp = context->exp[0];
    context->exp.pop_back();    
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
