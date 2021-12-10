#ifndef STRUCTS_HPP
#define STRUCTS_HPP
#include <vector>
#include <string>

struct Position;
struct ExpContent;
struct Operator;
struct Statement;
struct Type;
struct PortDecl;
struct PortDecls;
struct TypePar;
struct SingleImport;
struct GroupImport;
struct Import;
struct SingleExp;
struct PrimaryExp;
struct Exp;
struct IfExp;
struct LetExp;
struct BracketsExp;
struct FormalPar;
struct ProcExp;
struct ActorPar;
struct VarDecl;
struct ProcDecl;
struct FuncDecl;
struct VarType;
struct Actor;
struct Statement;
struct AssignmentStmt;
struct Index;
struct CallStmt;
struct WhileStmt;
struct BlockStmt;
struct IfStmt;
struct Action;
struct ActionTag;
struct ActionHead;
struct ChannelSelector;
struct InputPattern;
struct RepeatClause;
struct OutputExp;
struct InitAction;
struct InitHead;
struct ActionSched;
struct SchedFSM;
struct StateTransition;
struct SchedRegExp;
struct RegExp;
struct PriorityOrder;
struct PriorityIneq;
struct QualID;
struct TypeAttr;
struct FuncExps;
struct IDExp;
struct ExpLiteral;

struct Position
{
    int line;
    int col;
};

struct ID
{
    std::string id;
    Position pos;
};

struct Actor
{
    ID* id;
    std::vector <TypePar*> typePars;
    std::vector <ActorPar*> actorPars;
    std::vector<PortDecl*> inPorts;
    std::vector<PortDecl*> outPorts;
    Type* timeClause;
    ActionSched* actionSched;
    std::vector <VarDecl*> vars;
    std::vector <Action*> actions;
    std::vector <InitAction*> initActions;
    std::vector <PriorityOrder*> priorOrders;
     
    void addOutPort(PortDecl * x)
    {
        outPorts.push_back(x);
    }
    void addInPort(PortDecl * x)
    {
        inPorts.push_back(x);
    }
    void addTypePar(TypePar * x)
    {
        typePars.push_back(x);
    }
    void addActorPar(ActorPar * x)
    {
        actorPars.push_back(x);
    }
    void addVarType(VarDecl * x)
    {
        vars.push_back(x);
    }
    void addAction(Action * x){
        actions.push_back(x);
    }
    void addInitAction(InitAction * x)
    {
        initActions.push_back(x);
    }
    void addPriorityOrder(PriorityOrder * x)
    {
        priorOrders.push_back(x);
    }
};

struct ActorPar
{
    
    Type* type;
    ID* id;
    Exp* exp;
};

struct TypePar
{
    
    ID* id;
    Type* type;
};

struct PortDecl
{
    
    bool multi;
    Type* type;
    ID* id;
};

struct Import
{

};

struct SingleImport : public Import
{
    QualID* qualID;
    ID* alias;
};

struct GroupImport : public Import
{
    QualID* qualID;
};

struct QualID
{
    
    std::vector<ID*> ids;

    void addID(ID* x)
    {
        ids.push_back(x);
    }
};

struct Type//todo - fixed
{
    std::vector<TypeAttr*> attrs;

    void addTypeAttr(TypeAttr * x)
    {
        attrs.push_back(x);
    }
    ID* id; 
};

struct TypeAttr//todo - fixed
{
    ID* id;
    Type* type;
    Exp* exp;
};

struct VarDecl
{
    
   ID* id;
};

struct VarType : public VarDecl
{
    bool _mutable;
    Type* type;
    Exp* exp;

    void setExp(Exp * x)
    {
        exp = x;
    }
};

struct Exp
{
    Position pos;
    std::vector<ExpContent*> exp;

    void addToExp(ExpContent * x)
    {
        exp.push_back(x);
    }
};

struct ExpContent
{
    
};

struct Operator : public ExpContent
{
    ID* oper;
};

struct PrimaryExp : public ExpContent
{
    Operator* oper;
    SingleExp* singleExp;
};

struct SingleExp
{
    
};

struct BracketsExp : public SingleExp
{
    Exp* exp;
};

struct FuncExps : public SingleExp
{
    std::vector<Exp*> exps;

    void addExp(Exp * x)
    {
        exps.push_back(x);
    }
};

struct IDExp : public SingleExp
{
    bool old;
    ID* id;
};

struct ExpLiteral : public SingleExp
{
    ID* expLiteral;
};

struct IfExp : public SingleExp
{
    Exp* ifExp;
    Exp* thenExp;
    Exp* elseExp;
};

struct LetExp : public SingleExp
{
    
    std::vector<VarDecl*> varDecls;
    Exp* exp;

    void addVarDecl(VarDecl * x)
    {
        varDecls.push_back(x);
    }
};

struct FormalPar
{
    
    Type* type;
    ID* id;
};

struct ProcExp : public SingleExp
{
    
    std::vector<FormalPar*> formalPars;
    std::vector<VarDecl*> varDecls;
    Statement* statement;

    void addFormalPar(FormalPar * x)
    {
        formalPars.push_back(x);
    }
    void addVarDecl(VarDecl * x)
    {
        varDecls.push_back(x);
    }
};

struct ProcDecl : public VarDecl
{
    std::vector<FormalPar*> formalPars;
    std::vector<VarDecl*> varDecls;
    Statement* statement;

    void addFormalPar(FormalPar * x)
    {
        formalPars.push_back(x);
    }
    void addVarDecl(VarDecl * x)
    {
        varDecls.push_back(x);
    }
};

struct FuncDecl : public VarDecl
{   
    std::vector<FormalPar*> formalPars;
    std::vector<VarDecl*> varDecls;
    Exp* exp;

    void addFormalPar(FormalPar * x)
    {
        formalPars.push_back(x);
    }
    void addVarDecl(VarDecl * x)
    {
        varDecls.push_back(x);
    }
};

struct Statement
{
    
};

struct AssignmentStmt : public Statement
{
    ID* id;
    Index* index;
    std::string fieldRef;
};

struct Index
{
    
    std::vector <Exp*> exps;

    void addExp(Exp * x)
    {
        exps.push_back(x);
    }
};

struct CallStmt : public Statement
{
    
    Exp* exp;
    std::vector<Exp*> args;

    void addArg(Exp * x)
    {
        args.push_back(x);
    }
};

struct BlockStmt : public Statement
{
    
    std::vector<VarDecl*> varDecls;
    Statement* statement;

    void addVarDecl(VarDecl * x)
    {
        varDecls.push_back(x);
    }
};

struct IfStmt : public Statement
{
    
    Exp* ifExp;
    Statement* thenStmt;
    Statement* elseStmt;
};

struct WhileStmt : public Statement
{
    
    Exp* whileExp;
    std::vector<VarDecl*> vars;
    std::vector<Statement*> doStmts;

    void addVarDecl(VarDecl * x)
    {
        vars.push_back(x);
    }
    void addDoStmt(Statement * x)
    {
        doStmts.push_back(x);
    }
};

struct Action
{
    
    ActionTag* actionTag;
    ActionHead* actionHead;
    std::vector<Statement*> doStmts;

    void addDoStmt(Statement * x)
    {
        doStmts.push_back(x);
    }
};

struct ActionTag
{
    
    QualID* qualID;
};

struct ActionHead
{
    
    std::vector<InputPattern*> inputPatterns;
    std::vector<OutputExp*> outputExps;
    std::vector<Exp*> guardExps;
    std::vector<VarDecl*> varDecls;
    Exp* delayExp;

    void addInputPattern(InputPattern * x)
    {
        inputPatterns.push_back(x);
    }
    void addOutputExp(OutputExp * x)
    {
        outputExps.push_back(x);
    }
    void addGuardExp(Exp * x)
    {
        guardExps.push_back(x);
    }
    void addVarDecl(VarDecl * x)
    {
        varDecls.push_back(x);
    }
};

struct InputPattern
{
    
    ID* id;
    std::vector<std::string> IDs;
    RepeatClause* repeatClause;
    ChannelSelector* channelSelector;

    void addID(std::string x)
    {
        IDs.push_back(x);
    }
};

struct ChannelSelector//?
{
    
};

struct RepeatClause
{
    
    Exp* exp;
};

struct OutputExp
{
    
    ID* id;
    std::vector<Exp*> exps;
    RepeatClause* repeatClause;
    ChannelSelector* channelSelector;

    void addExp(Exp * x)
    {
        exps.push_back(x);
    }
};

struct InitAction
{
    
    ActionTag* actionTag;
    InitHead* initHead;
    std::vector<Statement*> stmts;

    void addStmt(Statement * x)
    {
        stmts.push_back(x);
    }
};

struct InitHead
{
    
    std::vector<OutputExp*> outputExps;
    std::vector<Exp*> guardExps;
    std::vector<VarDecl*> varDecls;
    Exp* delayExp;

    void addOutputExp(OutputExp * x)
    {
        outputExps.push_back(x);
    }
    void addGuardExp(Exp * x)
    {
        guardExps.push_back(x);
    }
    void addVarDecl(VarDecl * x)
    {
        varDecls.push_back(x);
    }
};

struct ActionSched//todo
{
    
    SchedFSM* schedFSM;
    SchedRegExp* schedRegExp;
};

struct SchedFSM
{
    
    bool fsm;
    ID* id;
    std::vector<StateTransition*> stateTransitions;

    void addStateTransition(StateTransition * x)
    {
        stateTransitions.push_back(x);
    }
};

struct StateTransition
{
    
};

struct SchedRegExp
{
    
    RegExp* regExp;
};

struct RegExp//?
{
    
};

struct PriorityOrder
{
    
    std::vector<PriorityIneq*> priorityIneqs;

    void addPriorityIneq(PriorityIneq * x)
    {
        priorityIneqs.push_back(x);
    }
};

struct PriorityIneq
{
    
    ActionTag* firstTag;
    ActionTag* secondTag;
    std::vector<ActionTag*> extraTags;

    void addExtraTag(ActionTag * x)
    {
        extraTags.push_back(x);
    }
};

struct AST
{
    Actor* actor;
    std::vector<GroupImport*> gImports;
    std::vector<SingleImport*> sImports;
    
    void addsImport(SingleImport * x)
    {
        sImports.push_back(x);
    }
    void addgImport(GroupImport * x)
    {
        gImports.push_back(x);
    }
};

struct Context
{
    int portType;
    Actor* actor;
    std::vector<ActorPar*> actorPars;
    std::vector<TypePar*> typePars;
    PortDecl* portDecl;
    GroupImport* gImport;
    SingleImport* sImport;
    QualID* qualID;
    Type* type;
    TypeAttr* typeAttr;
    VarDecl* varDecl;
    Exp* exp;
    Operator* oper;
    std::vector<ExpContent*> expContent;
    SingleExp* singleExp;
    IfExp* ifExp;
    LetExp* letExp;
    /*otherExps?*/
    FormalPar* formalPar;
    Statement* statement;
    Index* index;
    Action* action;
    ActionTag* actionTag;
    ActionHead* actionHead;
    InputPattern* inputPattern;
    OutputExp* outputExp;
    InitAction* initAction;
    InitHead* initHead;
    ActionSched* actionSched;
    PriorityIneq* priorityIneq;
    PriorityOrder* priorityOrder;

    Context()
    {
        portType = 0;
    }
};
#endif
