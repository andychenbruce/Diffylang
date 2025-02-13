//TODO: make an actual LL or LR parser generator instead whatever this is

#[derive(Debug, Clone)]
pub enum TokenNonParen {
    Name(String),
    Defun,
    Colon,
    Fold,
    If,
    Let,
}

pub enum Token {
    ParenLeft,
    ParenRight,
    Other(TokenNonParen),
}

pub struct Tokens<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

impl<'a> Tokens<'a> {
    pub fn new(code: &'a str) -> Self {
        Self {
            chars: code.chars().peekable(),
        }
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        take_whitespace(&mut self.chars);
        if let Some(c) = self.chars.peek() {
            if *c == '(' {
                self.chars.next();
                Some(Token::ParenLeft)
            } else if *c == ')' {
                self.chars.next();
                Some(Token::ParenRight)
            } else if *c == ':' {
                self.chars.next();
                Some(Token::Other(TokenNonParen::Colon))
            } else {
                Some(token_parse_string(&mut self.chars))
            }
        } else {
            None
        }
    }
}

fn take_whitespace<I: Iterator<Item = char>>(code: &mut std::iter::Peekable<I>) {
    while code.next_if(|x| x.is_whitespace()).is_some() {}
}
fn token_parse_string<I: Iterator<Item = char>>(code: &mut std::iter::Peekable<I>) -> Token {
    let mut name: String = "".to_owned();

    while let Some(c) = code.next_if(|x| !(x.is_whitespace() || *x == '(' || *x == ')')) {
        name.push(c);
    }

    Token::Other(match name.as_str() {
        ":" => TokenNonParen::Colon,
        "fold" => TokenNonParen::Fold,
        "if" => TokenNonParen::If,
        "let" => TokenNonParen::Let,
        "defun" => TokenNonParen::Defun,
        _ => TokenNonParen::Name(name),
    })
}

#[derive(Debug)]
pub enum TokenTree {
    Leaf(TokenNonParen),
    Branch(Vec<TokenTree>),
}

pub fn make_token_trees(
    tokens: &mut std::iter::Peekable<Tokens>,
) -> Result<Vec<TokenTree>, &'static str> {
    let mut trees = vec![];

    while let Some(tree) = make_token_tree(tokens)? {
        trees.push(tree);
    }

    Ok(trees)
}

pub fn make_token_tree(
    tokens: &mut std::iter::Peekable<Tokens>,
) -> Result<Option<TokenTree>, &'static str> {
    if let Some(next_token) = tokens.next() {
        if let Token::Other(tok) = next_token {
            return Ok(Some(TokenTree::Leaf(tok)));
        }
        if !matches!(next_token, Token::ParenLeft) {
            //println!("bruh = {:?}", next_token);
            return Err("expected left parenthesis");
        }
    } else {
        return Ok(None);
    }

    let mut children: Vec<TokenTree> = vec![];

    loop {
        if let Some(t) = tokens.peek() {
            match t {
                Token::ParenLeft => {
                    children.push(make_token_tree(tokens)?.ok_or("unexpected EOF")?)
                }
                Token::ParenRight => {
                    tokens.next();
                    break;
                }
                Token::Other(n) => {
                    children.push(TokenTree::Leaf(n.clone()));
                    tokens.next();
                }
            }
        } else {
            return Err("mismatched parenthesis");
        }
    }

    Ok(Some(TokenTree::Branch(children)))
}

pub fn parse_program(token_trees: Vec<TokenTree>) -> Result<ProgramParseTree, &'static str> {
    let declarations = token_trees
        .into_iter()
        .map(|x| parse_declaration(&x))
        .collect::<Result<Vec<_>, &'static str>>()?;

    Ok(ProgramParseTree { declarations })
}

pub fn parse_declaration(token_tree: &TokenTree) -> Result<Declaration, &'static str> {
    match token_tree {
        TokenTree::Leaf(_) => Err("expected declaration"),
        TokenTree::Branch(vec) => {
            if !matches!(
                vec.get(0).ok_or("declaration empty")?,
                TokenTree::Leaf(TokenNonParen::Defun)
            ) {
                return Err("declarations must start with \"defun\"");
            }
            let name = parse_name(vec.get(1).ok_or("declaration missing name")?)?;
            let args = parse_args(vec.get(2).ok_or("declaration missing args")?)?;
            let to_type = parse_name(vec.get(3).ok_or("declaration missing return type")?)?;
            let body = parse_expr(vec.get(4).ok_or("declaration missing body")?)?;

            return Ok(Declaration::FunctionDef(FunctionDefinition {
                name,
                arguments: args,
                to_type: VarType(to_type),
                func_body: body,
            }));
        }
    }
}

pub fn parse_expr(token_tree: &TokenTree) -> Result<Expression, &'static str> {
    match token_tree {
        TokenTree::Leaf(token_non_paren) => match token_non_paren {
            TokenNonParen::Name(var_name) => Ok(Expression::Variable(var_name.clone())),
            TokenNonParen::Defun => Err("uhh"),
            TokenNonParen::Colon => Err("bruh"),
            TokenNonParen::Fold => Err("mmm"),
            TokenNonParen::If => Err("idk"),
            TokenNonParen::Let => Err("bruh"),
        },
        TokenTree::Branch(_) => todo!(),
    }
}

pub fn parse_arg(token_tree: &TokenTree) -> Result<Argument, &'static str> {
    match token_tree {
        TokenTree::Branch(vec) => {
            let colon = vec.get(0).ok_or("arg empty")?;
            if !matches!(colon, TokenTree::Leaf(TokenNonParen::Colon)) {
                return Err("args must start with \":\"");
            }

            let varname = parse_name(vec.get(1).ok_or("arg missing name")?)?;
            let typename = parse_name(vec.get(2).ok_or("arg missing type")?)?;

            Ok(Argument {
                varname,
                vartype: VarType(typename),
            })
        }
        TokenTree::Leaf(_) => Err("bruh asdf"),
    }
}

pub fn parse_args(token_tree: &TokenTree) -> Result<Vec<Argument>, &'static str> {
    match token_tree {
        TokenTree::Branch(vec) => Ok(vec
            .into_iter()
            .map(parse_arg)
            .collect::<Result<Vec<_>, _>>()?),
        TokenTree::Leaf(_) => Err("bruh idk"),
    }
}

pub fn parse_name(token_tree: &TokenTree) -> Result<String, &'static str> {
    match token_tree {
        TokenTree::Leaf(TokenNonParen::Name(n)) => Ok(n.clone()),
        _ => Err("bruh"),
    }
}

#[derive(Clone, Debug)]
pub struct ProgramParseTree {
    pub declarations: Vec<Declaration>,
}

#[derive(Clone, Debug)]
pub enum Declaration {
    FunctionDef(FunctionDefinition),
    TestCaseDef(Expression),
}

#[derive(Clone, Debug)]
pub struct FunctionDefinition {
    pub name: String,
    pub arguments: Vec<Argument>,
    pub to_type: VarType,
    pub func_body: Expression,
}

#[derive(Clone, Debug)]
pub struct Argument {
    pub varname: String,
    pub vartype: VarType,
}

#[derive(Clone, Debug)]
pub struct VarType(pub String);

#[derive(Clone, Debug)]
pub enum Expression {
    Product(ProductType),
    ProductProject {
        index: i64,
        value: Box<Expression>,
    },

    ListLit(ListInner),
    FoldLoop {
        iter_val: FoldIter,
        accumulator: FoldAccumulator,
        body: Box<Expression>,
    },
    WhileFoldLoop {
        //while_token: Token![while],
        accumulator: FoldAccumulator,
        cond: Box<Expression>,
        body: Box<Expression>,
        exit_body: Box<Expression>,
    },

    FunctionApplication {
        func_name: String,
        args: Vec<Expression>,
    },

    Variable(String),
    IntegerLit(i64),
    StringLit(String),
    FloatLit(f64),
    BoolLit(bool),

    Addition(BinaryOp),
    Subtraction(BinaryOp),
    Multiplication(BinaryOp),
    Division(BinaryOp),
    Equality(BinaryOp),
    GreaterThan(BinaryOp),
    LessThan(BinaryOp),
    And(BinaryOp),
    Or(BinaryOp),

    Not(Box<Expression>),

    ExprWhere {
        bindings: Vec<LetBind>,
        inner: Box<Expression>,
    },
    IfThenElse {
        boolean: Box<Expression>,
        true_expr: Box<Expression>,
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

#[derive(Clone, Debug)]
pub struct FoldAccumulator {
    pub name: String,
    pub initial_expression: Box<Expression>,
}

#[derive(Clone, Debug)]
pub enum FoldIter {
    Range(FoldRange),
    ListExpr(Box<Expression>),
}

#[derive(Clone, Debug)]
pub struct FoldRange {
    pub start: Box<Expression>,
    pub end: Box<Expression>,
}

#[derive(Clone, Debug)]
pub struct BinaryOp {
    pub left_side: Box<Expression>,
    pub right_side: Box<Expression>,
}

#[derive(Clone, Debug)]
pub struct LetBind {
    pub name: String,
    pub value: Box<Expression>,
}

#[derive(Clone, Debug)]
pub struct ListInner {
    pub type_name: String,
    pub values: Vec<Expression>,
}

#[derive(Clone, Debug)]
pub struct ProductType {
    pub values: Vec<Expression>,
}
