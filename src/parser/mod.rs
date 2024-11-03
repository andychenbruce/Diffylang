use parsel::{syn::Token, Parse, ToTokens};

#[derive(Clone, Debug, Parse, ToTokens)]
pub struct ProgramParseTree {
    pub functions: parsel::ast::Punctuated<FunctionDefinition, Token![;]>,
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
    Variable(parsel::ast::Ident),
    IntegerLit(parsel::ast::LitInt),
    StringLit(parsel::ast::LitStr),
    FloatLit(parsel::ast::LitFloat),
    FunctionApplication {
        func_name: parsel::ast::Ident,
        #[parsel(recursive)]
        args: parsel::ast::Paren<parsel::ast::Punctuated<Box<Expression>, Token![,]>>,
    },
    Addition(parsel::ast::Paren<BinaryOp<Token![+]>>),
    Subtraction(parsel::ast::Paren<BinaryOp<Token![-]>>),
    Multiplication(parsel::ast::Paren<BinaryOp<Token![*]>>),
    Division(parsel::ast::Paren<BinaryOp<Token![/]>>),
    Equality(parsel::ast::Paren<BinaryOp<Token![==]>>),
    GreaterThan(parsel::ast::Paren<BinaryOp<Token![>]>>),
    LessThan(parsel::ast::Paren<BinaryOp<Token![<]>>),
    ExprWhere {
        bindings:
            parsel::ast::Paren<parsel::ast::Punctuated<parsel::ast::Paren<LetBind>, Token![,]>>,
        where_token: Token![in],
        #[parsel(recursive)]
        inner: Box<Expression>,
    },
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
