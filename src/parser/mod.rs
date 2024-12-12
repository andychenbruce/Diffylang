use parsel::{syn::Token, Parse, ToTokens};

mod kw {
    parsel::custom_keyword!(then);
    parsel::custom_keyword!(fold);
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct ProgramParseTree {
    pub declarations: parsel::ast::Punctuated<Declaration, Token![;]>,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub enum Declaration {
    FunctionDef(FunctionDefinition),
    TestCaseDef(TestCaseDefinition),
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct TestCaseDefinition {
    pub inner: parsel::ast::Brace<Expression>,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct FunctionDefinition {
    pub inner: parsel::ast::Bracket<FunctionDefinitionInner>,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct FunctionDefinitionInner {
    pub name: parsel::ast::Ident,
    pub arguments:
        parsel::ast::Paren<parsel::ast::Punctuated<parsel::ast::Paren<Argument>, Token![,]>>,
    pub right_arrow: Token![->],
    pub to_type: VarType,
    pub func_body: parsel::ast::Paren<Expression>,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct Argument {
    pub varname: parsel::ast::Ident,
    colon: Token![:],
    pub vartype: VarType,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct VarType {
    pub type_name: parsel::ast::Ident,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub enum Expression {
    #[parsel(recursive)]
    Product(ProductType),
    
    ProductProject {
        index: parsel::ast::LitInt,
        dot: Token![.],
        #[parsel(recursive)]
        value: Box<Expression>,
    },

    ListLit(parsel::ast::Brace<ListInner>),
    FoldLoop {
        fold_token: kw::fold,
        #[parsel(recursive)]
        iter_val: FoldIter,
        accumulator: parsel::ast::Paren<FoldAccumulator>,
        #[parsel(recursive)]
        body: Box<Expression>,
    },
    WhileFoldLoop {
        while_token: Token![while],
        accumulator: parsel::ast::Paren<FoldAccumulator>,
        #[parsel(recursive)]
        cond: Box<Expression>,
        #[parsel(recursive)]
        body: Box<Expression>,
        #[parsel(recursive)]
        exit_body: Box<Expression>,
    },

    FunctionApplication {
        func_name: parsel::ast::Ident,
        #[parsel(recursive)]
        args: parsel::ast::Paren<parsel::ast::Punctuated<Box<Expression>, Token![,]>>,
    },

    Variable(parsel::ast::Ident),
    IntegerLit(parsel::ast::LitInt),
    StringLit(parsel::ast::LitStr),
    FloatLit(parsel::ast::LitFloat),
    BoolLit(parsel::ast::LitBool),

    Addition(parsel::ast::Paren<BinaryOp<Token![+]>>),
    Subtraction(parsel::ast::Paren<BinaryOp<Token![-]>>),
    Multiplication(parsel::ast::Paren<BinaryOp<Token![*]>>),
    Division(parsel::ast::Paren<BinaryOp<Token![/]>>),
    Equality(parsel::ast::Paren<BinaryOp<Token![==]>>),
    GreaterThan(parsel::ast::Paren<BinaryOp<Token![>]>>),
    LessThan(parsel::ast::Paren<BinaryOp<Token![<]>>),
    And(parsel::ast::Paren<BinaryOp<Token![&&]>>),
    Or(parsel::ast::Paren<BinaryOp<Token![||]>>),

    Not {
        exclamation_mark: Token![!],
        #[parsel(recursive)]
        inner: Box<Expression>,
    },
    ExprWhere {
        bindings:
            parsel::ast::Paren<parsel::ast::Punctuated<parsel::ast::Paren<LetBind>, Token![,]>>,
        where_token: Token![in],
        #[parsel(recursive)]
        inner: Box<Expression>,
    },
    IfThenElse {
        if_token: Token![if],
        #[parsel(recursive)]
        boolean: Box<Expression>,
        then_token: kw::then,
        #[parsel(recursive)]
        true_expr: Box<Expression>,
        else_token: Token![else],
        #[parsel(recursive)]
        false_expr: Box<Expression>,
    },
    
    
    // },
    //     ListIndex{
    //     #[parsel(recursive)]
    //     list: Box<Expression>,
    //     #[parsel(recursive)]
    //     index: parsel::ast::Brace<Box<Expression>>
    // },
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct FoldAccumulator {
    pub name: parsel::ast::Ident,
    comma: Token![,],
    #[parsel(recursive)]
    pub initial_expression: Box<Expression>,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub enum FoldIter {
    Range(parsel::ast::Bracket<FoldRange>),
    #[parsel(recursive)]
    ListExpr(Box<Expression>),
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct FoldRange {
    #[parsel(recursive)]
    pub start: Box<Expression>,
    comma: Token![,],
    #[parsel(recursive)]
    pub end: Box<Expression>,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct BinaryOp<Middle> {
    #[parsel(recursive)]
    pub left_side: Box<Expression>,
    operator: Middle,
    #[parsel(recursive)]
    pub right_side: Box<Expression>,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct LetBind {
    let_token: Token![let],
    pub name: parsel::ast::Ident,
    equals_sign: Token![=],
    #[parsel(recursive)]
    pub value: Box<Expression>,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct ListInner {
    pub type_name: parsel::ast::Ident,
    pub semicolon: Token![;],
    #[parsel(recursive)]
    pub values: parsel::ast::Punctuated<Box<Expression>, Token![,]>,
}

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct ProductType {
    #[parsel(recursive)]
    pub values: parsel::ast::Paren<parsel::ast::Punctuated<Box<Expression>, Token![,]>>,
}
