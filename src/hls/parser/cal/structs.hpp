#ifndef STRUCTS_HPP
#define STRUCTS_HPP
#include <vector>
#include <string>

struct Position;
struct Statement;
struct Type;
struct PortDecl;
struct IOSig;
struct TypePar;
struct SingleImport;
struct GroupImport;
struct Import;
struct SingleExp;
struct PrimaryExp;
struct Exp;
struct IfExp;
struct LetExp;
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
    int linePos;
};

struct Actor
{
    Position pos;
    std::vector<Import*> imports;
    std::string ID;
    std::vector <TypePar*> typePars;
    std::vector <ActorPar*> actorPars;
    IOSig* IO;
    Type* timeClause;
    ActionSched* actionSched;
    std::vector <VarDecl*> vars;
    std::vector <Action*> actions;
    std::vector <InitAction*> initActions;
    std::vector <PriorityOrder*> priorOrders;
    
    void addImport(Import * x)
    {
        imports.push_back(x);
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
    Position pos;
    Type* type;
    std::string ID;
    Exp* exp;
};

struct TypePar
{
    Position pos;
    std::string ID;
    Type* type;
};


struct IOSig
{
    Position pos;
    std::vector <PortDecl*> inPorts;
    std::vector <PortDecl*> outPorts;

    void addInPorts(PortDecl * x){
        inPorts.push_back(x);
    }
    void addOutPorts(PortDecl * x){
        outPorts.push_back(x);
    }
};

struct PortDecl
{
    Position pos;
    bool multi;
    Type* type;
    std::string ID;
};

struct Import
{
    Position pos;
};

struct SingleImport : public Import
{
    std::vector <std::pair <QualID*, std::string>> IDs;

    void addIDs(QualID * x, std::string y = 0)
    {
        IDs.push_back(std::make_pair(x, y));
    }
};

struct GroupImport : public Import
{
    std::vector <QualID*> IDs;

    void addIDs(QualID * x)
    {
        IDs.push_back(x);
    }
};

struct QualID
{
    Position pos;
    std::vector<std::string> IDs;

    void addIDs(std::string x)
    {
        IDs.push_back(x);
    }
};

struct Type//todo
{
    
};

struct TypeAttr//todo
{
    std::string ID;
    Type* type;
    Exp* exp;
};

struct VarDecl
{
    Position pos;
};

struct VarType : public VarDecl
{
    bool _mutable;
    Type* type;
    std::string ID;
    Exp* value;
};

struct Exp
{
    Position pos;
    std::vector<PrimaryExp*> primaryExps;
    std::vector<std::string> operators;

    void addPrimaryExp(PrimaryExp * x)
    {
        primaryExps.push_back(x);
    }
    void addOperator(std::string x)
    {
        operators.push_back(x);
    }
};

struct PrimaryExp//todo
{
    Position pos;
    std::string oper;
    SingleExp* singleExp;
};

struct SingleExp
{
    Position pos;
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
    std::string ID;
};

struct ExpLiteral
{
    std::string expLiteral;
};

struct IfExp : public SingleExp
{
    Position pos;
    Exp* ifExp;
    Exp* thenExp;
    Exp* elseExp;
};

struct LetExp : public SingleExp
{
    Position pos;
    std::vector<VarDecl*> varDecls;
    Exp* exp;

    void addVarDecl(VarDecl * x)
    {
        varDecls.push_back(x);
    }
};

struct FormalPar
{
    Position pos;
    Type* type;
    std::string ID;
};

struct ProcExp : public SingleExp
{
    Position pos;
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
    std::string ID;
    std::vector<FormalPar*> formalPars;
    std::vector<VarDecl*> varDecls;
    Statement* statement;

    void addFormalPar(FormalPar * x)
    {
        formalPars.push_back(x);
    }
};

struct FuncDecl : public VarDecl
{   
    std::string ID;
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
    Position pos;
};

struct AssignmentStmt : public Statement
{
    std::string ID;
    Index* index;
    std::string fieldRef;
};

struct Index
{
    Position pos;
    std::vector <Exp*> exps;

    void addExp(Exp * x)
    {
        exps.push_back(x);
    }
};

struct CallStmt : public Statement
{
    Position pos;
    Exp* exp;
    std::vector<Exp*> args;

    void addArg(Exp * x)
    {
        args.push_back(x);
    }
};

struct BlockStmt : public Statement
{
    Position pos;
    std::vector<VarDecl*> varDecls;
    Statement* statement;

    void addVarDecl(VarDecl * x)
    {
        varDecls.push_back(x);
    }
};

struct IfStmt : public Statement
{
    Position pos;
    Exp* ifExp;
    Statement* thenStmt;
    Statement* elseStmt;
};

struct WhileStmt : public Statement
{
    Position pos;
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
    Position pos;
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
    Position pos;
    std::vector<std::string> IDs;

    void addID(std::string x)
    {
        IDs.push_back(x);
    }
};

struct ActionHead
{
    Position pos;
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
    Position pos;
    std::string ID;
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
    Position pos;
};

struct RepeatClause
{
    Position pos;
    Exp* exp;
};

struct OutputExp
{
    Position pos;
    std::string ID;
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
    Position pos;
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
    Position pos;
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
    Position pos;
    SchedFSM* schedFSM;
    SchedRegExp* schedRegExp;
};

struct SchedFSM
{
    Position pos;
    bool fsm;
    std::string ID;
    std::vector<StateTransition*> stateTransitions;

    void addStateTransition(StateTransition * x)
    {
        stateTransitions.push_back(x);
    }
};

struct StateTransition
{
    Position pos;
};

struct SchedRegExp
{
    Position pos;
    RegExp* regExp;
};

struct RegExp//?
{
    Position pos;
};

struct PriorityOrder
{
    Position pos;
    std::vector<PriorityIneq*> priorityIneqs;

    void addPriorityIneq(PriorityIneq * x)
    {
        priorityIneqs.push_back(x);
    }
};

struct PriorityIneq
{
    Position pos;
    ActionTag* firstTag;
    ActionTag* secondTag;
    std::vector<ActionTag*> extraTags;

    void addExtraTag(ActionTag * x)
    {
        extraTags.push_back(x);
    }
};

#endif
