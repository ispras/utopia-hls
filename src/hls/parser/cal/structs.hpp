#ifndef STRUCTS_HPP
#define STRUCTS_HPP
#include <vector>
#include <string>
#include <iostream>

using namespace std;

struct TypeAttr;

enum
{
    DIV = 0,
    MUL,
    ADD,
    SUB,
    MORE,
    LESS,
    EQ,
    MOREEQ,
    LESSEQ,
    MOD,
    ITE,
    IDEXPOLD,
    IDEXPNOOLD,
    NOOP
};
struct Position
{
    int line;
    int col;
    Position(int xline = 0, int xcol = 0)
    {
        col = xcol;
        line = xline;
    }
    void print()
    {
        cout <<"("<< line <<','<< col << ") ";
    }
};

struct ID
{
    std::string id;
    Position pos;
    ID(){}
    ID(string x, int line, int col)
    {
        id = x;
        pos.line = line;
        pos.col = col;
    }
    void print()
    {
        cout <<endl << id;
        pos.print();
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
    Type(ID* xid)
    {
        id = xid;
    }
    void print();
};

struct Exp
{
    int op;
    ID* id;
    Exp* op1;
    Exp* op2;
    Exp* op3;
    Exp(int opc, Exp* p1 = NULL, Exp* p2 = NULL, Exp* p3 = NULL, ID* x = NULL)
    {
        op = opc;
        op1 = p1;
        op2 = p2;
        op3 = p3;
        id = x;
    }
    void print();
    int evaluate();
    virtual ~Exp(){}
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
    cout << endl << "    TypeAttr: ";
    id->print();
    if(type)
        type->print();
    if(exp)
        exp->print();
};

void Type::print()
{
    cout << endl << "    Type: ";
    id->print();
    for (int i = 0; i < attrs.size(); i++)
        attrs[i]->print();
};

struct TypePar
{
    ID* id;
    Type* type;
    TypePar(ID* xid, Type* xtype = NULL)
    {
        id = xid;
        type = xtype;
    }
    void print()
    {
        id->print();
        if(type)
            type->print();
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
    VarType(bool xmutable, ID* xid, Type* xtype = NULL, Exp* xexp = NULL)
    {
        id = xid;
        _mutable = xmutable;
        type = xtype;
        exp = xexp;
    }
    void print()
    {
        cout << endl << "VarDecl: ";
        if(_mutable)
            cout << endl << "mutable " << " ";
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
    FormalPar(ID* xid, Type* xtype = NULL)
    {
        id = xid;
        type = xtype;
    }
    void print()
    {
        cout << endl << "Formal Par: ";
        id->print();
        if(type)
            type->print();
    }
};

struct Statement
{
    
};

struct ProcExp : public Exp
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

struct LetExp : public Exp
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

void Exp::print()
{
    //toggle to eval exp
    //cout << evaluate() << endl;
    switch (op)
    {
    case ADD:
        op1->print();
        op2->print();
        cout << "+ ";
        break;
    case SUB:
        op1->print();
        op2->print();
        cout << "- ";
        break;
    case MUL:
        op1->print();
        op2->print();
        cout << "* ";
        break;
    case DIV:
        op1->print();
        op2->print();
        cout << "/ ";
        break;
    case NOOP:
        cout << (id->id) << " ";
        break;
    case ITE:
        if(op1->evaluate()){
            op2->print();
            break;
        }
        op3->print();
        break;
    case IDEXPOLD:
        cout << "old ";
        cout << id->id << " ";
        break;
    case IDEXPNOOLD:
        cout << "old ";
        cout << id->id << " ";
        break;
    case MORE:
        op1->print();
        op2->print();
        cout << "> ";
        break;
    case LESS:
        op1->print();
        op2->print();
        cout << "< ";
        break;
    case EQ:
        op1->print();
        op2->print();
        cout << "= ";
        break;
    default:
        cout << "fault!" << endl;
        break;
    }
}

int Exp::evaluate()
{
    switch (op)
    {
    case ADD:
        return op1->evaluate() + op2->evaluate();
    case SUB:
        return op1->evaluate() - op2->evaluate();
    case MUL:
        return op1->evaluate() * op2->evaluate();
    case DIV:
        return op1->evaluate() / op2->evaluate();
    case NOOP:
        return stoi(id->id);
    case ITE:
        if(op1->evaluate())
            return op2->evaluate();
        return op3->evaluate();
    case IDEXPOLD:
        //temporarily return 1
        return 1;
    case IDEXPNOOLD:
        //temporarily return -1
        return -1;
    case MORE:
        return op1->evaluate() > op2->evaluate();
    case LESS:
        return op1->evaluate() < op2->evaluate();
    case EQ:
        return op1->evaluate() == op2->evaluate();
    default:
        cout << "fault!" << endl;
        return -1;
    }
}
struct ActorPar
{
    Type* type;
    ID* id;
    Exp* exp;
    ActorPar(ID* xid, Exp* xexp = NULL, Type* xtype = NULL)
    {
        id = xid;
        exp = xexp;
        type = xtype;
    }
    void print()
    {
        cout << endl << "ActorPar: ";
        if(type)
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
    PortDecl(bool xmulti, ID* xid, Type* xtype = NULL)
    {
        id = xid;
        multi = xmulti;
        type = xtype;
    }
    void print()
    {
        cout << endl << "PortDecl: ";
        if(multi)
            cout << "multi" << ' ';
        if(type)
            type->print();
        id->print();
    }
};

struct SingleImport
{
    QualID* qualID;
    ID* alias;
    SingleImport(QualID* xid, ID* xalias = NULL)
    {
        qualID = xid;
        alias = xalias;
    }
    void print()
    {
        cout << endl << "    SingleImport: ";
        qualID->print();
        cout << endl << "alias: ";
        alias->print();
    }
};

struct GroupImport 
{
    QualID* qualID;
    void print()
    {
        cout << endl << "    GroupImport: ";
        qualID->print();
    }
};

struct Index
{
    std::vector <Exp*> exps;

    void addExp(Exp * x)
    {
        exps.push_back(x);
    }
};

struct AssignmentStmt : public Statement
{
    ID* id;
    Index* index;
    std::string fieldRef;
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

struct ChannelSelector//todo
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

struct StateTransition
{
    
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

struct RegExp//todo
{
    
};

struct SchedRegExp
{
    RegExp* regExp;
};

struct ActionSched//todo
{
    SchedFSM* schedFSM;
    SchedRegExp* schedRegExp;
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
    void addVarDecl(VarDecl * x)
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
        cout << endl << "    Actor: ";
        id->print();
        if(typePars.size() > 0)
            cout << endl << "    TypePars: ";
        for (int i = 0; i < typePars.size(); i++)
            typePars[i]->print();
        if(actorPars.size() > 0)
            cout << endl << "    ActorPars: ";
        for (int i = 0; i < actorPars.size(); i++)
            actorPars[i]->print();
        if(inPorts.size() > 0)
            cout << endl << "    inPorts: "; 
        for (int i = 0; i < inPorts.size(); i++)
            inPorts[i]->print();
        if(outPorts.size() > 0)
            cout << endl << "    outPorts: ";
        for (int i = 0; i < outPorts.size(); i++)
            outPorts[i]->print();
        if(vars.size() > 0)
            cout << endl << "    ActorVars: ";
        for(int i = 0; i < vars.size(); i++){
            VarType* tmp = dynamic_cast<VarType*>(vars[i]);
            if(tmp)
                tmp->print();
        }
        if(timeClause){
            cout << endl << "    timeClause: ";
            timeClause->print();
        }
        cout << endl;
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
    int _mutable;
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
    std::vector<Exp*> exp;
    std::vector<int> opers;
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
        _mutable = 0;
    }
};
#endif
