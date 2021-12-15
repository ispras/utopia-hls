#ifndef STRUCTS_HPP
#define STRUCTS_HPP
#include <vector>
#include <string>
#include <iostream>

using namespace std;


struct ExpContent;
struct Operator;
struct Statement;
struct Exp;
struct BracketsExp;
struct FormalPar;
struct VarDecl;
struct ProcDecl;
struct FuncDecl;
struct VarType;
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
struct FuncExps;
struct TypeAttr;

struct Position
{
    int line;
    int col;
    void print()
    {
        cout <<"pos("<< line <<','<< col << ") ";
    }
};

struct ID
{
    std::string id;
    Position pos;
    void print()
    {
        pos.print();
        cout << id << ' ';
    }
};

struct QualID
{
    std::vector<ID*> ids;
    void addID(ID* x)
    {
        ids.push_back(x);
    }
    void print()
    {
        cout << endl << "QualID: ";
        for (int i = 0; i < ids.size(); i++)
            ids[i]->print();
    }
};

struct Type
{
    std::vector<TypeAttr*> attrs;
    void addTypeAttr(TypeAttr * x)
    {
        attrs.push_back(x);
    }
    ID* id;
    void print();
};

struct ExpContent
{
    void print()
    {
        
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
    void print()
    {
        cout << endl << "Exp: ";
        for (int i = 0; i < exp.size(); i++)
            exp[i]->print();
    }
};

struct TypeAttr
{
    ID* id;
    Type* type;
    Exp* exp;
    void print();
};

void TypeAttr::print()
{
    cout << endl << "TypeAttr: ";
    id->print();
    if(type)
        type->print();
    if(exp)
        exp->print();
};

void Type::print()
{
    cout << endl << "Type: ";
    id->print();
    for (int i = 0; i < attrs.size(); i++)
        attrs[i]->print();
};

struct TypePar
{
    ID* id;
    Type* type;
    void print()
    {
        if(type)
            type->print();
        id->print();
    }
};

struct VarDecl
{
    ID* id;
    virtual ~VarDecl(){};
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
    void print()
    {
        cout << endl << "VarDecl: ";
        cout << "mutable:" << _mutable << " ";
        id->print();
        if(type)
            type->print();
        if(exp)
            exp->print();
    }
    virtual ~VarType(){}
};

struct FormalPar
{
    Type* type;
    ID* id;
    void print()
    {
        cout << endl << "Formal Par: "; 
        if(type)
            type->print();
        id->print();
    }
};

struct SingleExp
{
    void print()
    {
        
    }
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
    ID* id;
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
    void print()
    {
        cout << endl << "FuncDecl: ";
        for (int i = 0; i < formalPars.size(); i++)
            formalPars[i]->print();
        for (int i = 0; i < varDecls.size(); i++)
            ;//varDecls[i]->print();
        if(exp)
            exp->print();
    }
};

struct Operator : public ExpContent
{
    ID* oper;
    void print()
    {
        oper->print();
    }
};

struct BracketsExp : public SingleExp
{
    Exp* exp;
    void print()
    {
        exp->print();
    }
};

struct FuncExps : public SingleExp
{
    std::vector<Exp*> exps;
    void addExp(Exp * x)
    {
        exps.push_back(x);
    }
    void print()
    {
        cout << endl << "FucExp: ";
        for (int i = 0; i < exps.size(); i++)
            exps[i]->print();
    }
};

struct IDExp : public SingleExp
{
    bool old;
    ID* id;
    void print()
    {
        cout << endl << "IDExp: ";
        cout << "old=" << old << " ";
        id->print();
    }
};

struct ExpLiteral : public SingleExp
{
    ID* expLiteral;
    void print()
    {
        cout << endl<< "Literal: ";
        expLiteral->print();
    }
};

struct IfExp : public SingleExp
{
    Exp* ifExp;
    Exp* thenExp;
    Exp* elseExp;
    void print()
    {
        cout << endl << "ifExp: ";
        ifExp->print();
        thenExp->print();
        elseExp->print();
    }
};

struct LetExp : public SingleExp
{
    std::vector<VarDecl*> varDecls;
    Exp* exp;
    void addVarDecl(VarDecl * x)
    {
        varDecls.push_back(x);
    }
    void print()
    {
        cout << endl << "LetExp: ";
        for (int i = 0; i < varDecls.size(); i++)
            ;//varDecls[i]->print();
        if(exp)
            exp->print();
    }
};


struct PrimaryExp : public ExpContent
{
    Operator* oper;
    SingleExp* singleExp;
    void print()
    {
        cout << endl << "PrimExp: ";
        if(oper)
            oper->print();
        singleExp->print();
    }
};

struct ActorPar
{
    Type* type;
    ID* id;
    Exp* exp;
    void print()
    {
        cout << endl << "ActorPar: ";
        if(type != NULL)
            type->print();
        if(id)
            id->print();
        if(exp)
            exp->print();
    }
};

struct PortDecl
{
    bool multi;
    Type* type;
    ID* id;
    PortDecl()
    {
        type = NULL;
    }
    void print()
    {
        cout << endl << "PortDecl: ";
        cout << "multi:" << multi << ' ';
        if(type)
            type->print();
        id->print();
    }
};

struct Import
{
    void print()
    {
        
    }
};

struct SingleImport : public Import
{
    QualID* qualID;
    ID* alias;
    void print()
    {
        cout << endl << "SingleImport: ";
        qualID->print();
        alias->print();
    }
};

struct GroupImport : public Import
{
    QualID* qualID;
    void print()
    {
        cout << endl << "GroupImport: ";
        qualID->print();
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

struct ActionTag
{
    QualID* qualID;
    void print()
    {
        cout << endl << "ActionTag: ";
        qualID->print();
    }
};



struct RepeatClause
{
    Exp* exp;
    void print()
    {
        cout << endl << "Exp: ";
        exp->print();
    }
};

struct ChannelSelector//?
{
    
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
    void print()
    {
        cout << endl << "OutputExp: ";
        id->print();
        for (int i = 0; i < exps.size(); i++)
            exps[i]->print();
        if(repeatClause)
            repeatClause->print();

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
    void print()
    {
        cout << endl << "InitHead ";
        for (int i = 0; i < outputExps.size(); i++)
            outputExps[i]->print();
        for (int i = 0; i < varDecls.size(); i++)
            ;//varDecls[i]->print();
        if(delayExp)
            delayExp->print();
    }
};

struct InputPattern
{
    ID* id;
    QualID * IDs;
    RepeatClause* repeatClause;
    ChannelSelector* channelSelector;
    void print()
    {
        cout << endl << "InpuPattern: ";
        id->print();
        if(IDs)
            IDs->print();
        if(repeatClause)
            repeatClause->print();

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
    void print()
    {
        cout << endl << "InitAction: ";
        if(actionTag)
            actionTag->print();
        if(initHead)
            initHead->print();
    }
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
    void print()
    {
        cout << endl << "ActionHead";
        for (int i = 0; i < inputPatterns.size(); i++)
            inputPatterns[i]->print();
        for (int i = 0; i < outputExps.size(); i++)
            outputExps[i]->print();
    }
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

struct ActionSched//todo
{
    SchedFSM* schedFSM;
    SchedRegExp* schedRegExp;
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

struct Action
{
    ActionTag* actionTag;
    ActionHead* actionHead;
    std::vector<Statement*> doStmts;

    void addDoStmt(Statement * x)
    {
        doStmts.push_back(x);
    }
    void print()
    {
        cout << endl << "Action: ";
        if(actionTag)
            actionTag->print();
        if(actionHead)
            actionHead->print();

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

struct PriorityOrder
{
    std::vector<PriorityIneq*> priorityIneqs;

    void addPriorityIneq(PriorityIneq * x)
    {
        priorityIneqs.push_back(x);
    }
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
    void print()
    {
        cout << endl << "Actor: ";
        id->print();
        cout << endl << "TypePars: ";
        for (int i = 0; i < typePars.size(); i++)
            typePars[i]->print();
        cout << endl << "Actor Pars: ";
        for (int i = 0; i < actorPars.size(); i++)
            actorPars[i]->print();
        cout << endl << "inPorts: "; 
        for (int i = 0; i < inPorts.size(); i++)
            inPorts[i]->print();
        cout << endl << "outPorts: ";
        for (int i = 0; i < outPorts.size(); i++)
            outPorts[i]->print();
        cout << endl << "vars: ";
        for(int i = 0; i < vars.size(); i++){
            VarType* tmp = dynamic_cast<VarType*>(vars[i]);
            if(tmp != NULL)
                tmp->print();
        }
        if(timeClause){
            cout << endl << "timeClause: ";
            timeClause->print();
        }
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
    void print()
    {
        cout << endl << "AST: ";
        for (int i = 0; i < gImports.size(); i++)
            gImports[i]->print();
        for (int i = 0; i < sImports.size(); i++)
            sImports[i]->print();
        actor->print();
    }
};

struct Context
{
    int portType;
    ID* id;
    Actor* actor;
    std::vector<ActorPar*> actorPars;
    std::vector<TypePar*> typePars;
    PortDecl* portDecl;
    GroupImport* gImport;
    SingleImport* sImport;
    QualID* qualID;
    Type* type;
    TypeAttr* typeAttr;
    std::vector<VarDecl*> varDecls;
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
    RepeatClause* repeatClause;
    std::vector<InputPattern*> inputPattern;
    std::vector<OutputExp*> outputExp;
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
