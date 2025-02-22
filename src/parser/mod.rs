//TODO: make an actual LL or LR parser generator instead whatever this is

#[derive(Debug)]
pub struct ParseError {
    pub pos_start: ParsePos,
    pub pos_end: ParsePos,
    pub reason: ParseErrorReason,
}

#[derive(Debug)]
pub enum ParseErrorReason {
    MissingClosingParenthesis,
    UnexpectedEOF,
    UnexpectedRightParenthesis,
    StrayAtom,
    EmptyDeclaration,
    ExpectedName,
    ExpectedLambdaInputName,
    FunctionApplicationIsntName,
    ExpectedUniverse,
    ExpectedArgs,
    ExpectedConstructors,
    ExpectedExpression,
    ExpectedLambdaBodyExpression,
    ExpectedTestCaseExpression,
    ExpectedLetBindingExpression,
    ExpectedDependentFunctionTypeToTypeExpression,
    ExpectedDependentProductTypeSecondTypeExpression,
    ExpectedDependentFunctionTypeArgument,
    ExpectedArgTypeExpression,
    ExpectedDependentProductTypeArgument,
    ExpressionCantStartWithThis,
    ExpectedDefinitionType,
    ExpectedLetBindingBody,
    VariableNameIsKeyword,
    EmptyExpression,
    ExpectedBindings,
    BindingsAreLeaf,
    BindingIsLeaf,
    ArgsAreLeaf,
    ArgIsLeaf,
    ArgDoesntStartWithColon,
    EmptyArg,
    TopLevelNotDeclaration,
}

#[derive(Debug, Clone)]
enum TokenNonParen {
    NonKeyword(String),
    Def,
    Defgadt,
    Deftest,
    Colon,
    Let,
    Lambda,
    Pi,
    Sigma,
}

enum TokenVal {
    ParenLeft,
    ParenRight,
    Other(TokenNonParen),
}

#[derive(Debug, Clone, Copy)]
pub struct ParsePos {
    pub col: usize,
    pub line: usize,
}

struct Token {
    val: TokenVal,
    pos_start: ParsePos,
    pos_end: ParsePos,
}

struct CharIter<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    pub curr_pos: ParsePos,
}

struct Tokens<'a> {
    char_iter: CharIter<'a>,
}

impl<'a> Tokens<'a> {
    pub fn new(code: &'a str) -> Self {
        Self {
            char_iter: CharIter {
                chars: code.chars().peekable(),
                curr_pos: ParsePos { col: 0, line: 0 },
            },
        }
    }
    fn take_whitespace(&mut self) {
        while self.char_iter.next_if(|x| x.is_whitespace()).is_some() {}
    }

    fn token_parse_string(&mut self) -> TokenVal {
        let mut name: String = "".to_owned();

        while let Some(c) = self
            .char_iter
            .next_if(|x| !(x.is_whitespace() || *x == '(' || *x == ')'))
        {
            name.push(c);
        }

        TokenVal::Other(match name.as_str() {
            ":" => TokenNonParen::Colon,
            "λ" => TokenNonParen::Lambda,
            "Π" => TokenNonParen::Pi,
            "Σ" => TokenNonParen::Sigma,
            "let" => TokenNonParen::Let,
            "def" => TokenNonParen::Def,
            "deftest" => TokenNonParen::Deftest,
            "defgadt" => TokenNonParen::Defgadt,
            _ => TokenNonParen::NonKeyword(name),
        })
    }
}

impl<'a> Iterator for CharIter<'a> {
    type Item = char;
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.chars.next();
        if let Some(next) = next {
            self.curr_pos.col += 1;
            if next == '\n' {
                self.curr_pos.col = 0;
                self.curr_pos.line += 1;
            }
        }
        next
    }
}

impl<'a> CharIter<'a> {
    fn peek(&mut self) -> Option<char> {
        self.chars.peek().map(|x| *x)
    }
    pub fn next_if(&mut self, func: impl FnOnce(&char) -> bool) -> Option<char> {
        match self.peek() {
            Some(matched) if func(&matched) => {
                let next = self.next();
                assert!(next.unwrap() == matched);
                Some(matched)
            }
            _ => None,
        }
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        self.take_whitespace();
        if let Some(c) = self.char_iter.peek() {
            let start_pos = self.char_iter.curr_pos;
            let token_val = if c == '(' {
                self.char_iter.next();
                TokenVal::ParenLeft
            } else if c == ')' {
                self.char_iter.next();
                TokenVal::ParenRight
            } else if c == ':' {
                self.char_iter.next();
                TokenVal::Other(TokenNonParen::Colon)
            } else {
                self.token_parse_string()
            };

            return Some(Token {
                val: token_val,
                pos_start: start_pos,
                pos_end: self.char_iter.curr_pos,
            });
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct TokenTree {
    val: TokenTreeVal,
    start_pos: ParsePos,
    end_pos: ParsePos,
}

#[derive(Debug)]
enum TokenTreeVal {
    Leaf(TokenNonParen),
    Branch(Vec<TokenTree>),
}

fn make_token_trees(
    tokens: &mut std::iter::Peekable<Tokens>,
) -> Result<Vec<TokenTree>, ParseError> {
    let mut trees = vec![];

    while let Some(tree) = make_token_tree(tokens)? {
        trees.push(tree);
    }

    Ok(trees)
}

fn make_token_tree(
    tokens: &mut std::iter::Peekable<Tokens>,
) -> Result<Option<TokenTree>, ParseError> {
    if let Some(next_token) = tokens.next() {
        match next_token.val {
            TokenVal::Other(tok) => {
                return Ok(Some(TokenTree {
                    val: TokenTreeVal::Leaf(tok),
                    start_pos: next_token.pos_start,
                    end_pos: next_token.pos_end,
                }));
            }
            TokenVal::ParenLeft => {
                let mut children: Vec<TokenTree> = vec![];

                loop {
                    if let Some(t) = tokens.peek() {
                        let t_start = t.pos_start;
                        let t_end = t.pos_end;
                        match &t.val {
                            TokenVal::ParenLeft => {
                                children.push(make_token_tree(tokens)?.ok_or(ParseError {
                                    pos_start: t_start,
                                    pos_end: t_end,
                                    reason: ParseErrorReason::UnexpectedEOF,
                                })?)
                            }
                            TokenVal::ParenRight => {
                                let out = TokenTree {
                                    val: TokenTreeVal::Branch(children),
                                    start_pos: t.pos_start,
                                    end_pos: t.pos_end,
                                };
                                tokens.next();
                                return Ok(Some(out));
                            }
                            TokenVal::Other(n) => {
                                children.push(TokenTree {
                                    val: TokenTreeVal::Leaf(n.clone()),
                                    start_pos: t.pos_start,
                                    end_pos: t.pos_end,
                                });
                                tokens.next();
                            }
                        }
                    } else {
                        return Err(ParseError {
                            pos_start: next_token.pos_start,
                            pos_end: next_token.pos_end,
                            reason: ParseErrorReason::MissingClosingParenthesis,
                        });
                    }
                }
            }

            _ => {
                return Err(ParseError {
                    pos_start: next_token.pos_start,
                    pos_end: next_token.pos_end,
                    reason: ParseErrorReason::UnexpectedRightParenthesis,
                })
            }
        }
    } else {
        return Ok(None);
    }
}

fn parse_program(token_trees: Vec<TokenTree>) -> Result<ProgramParseTree, ParseError> {
    let declarations = token_trees
        .into_iter()
        .map(|x| parse_top(&x))
        .collect::<Result<Vec<_>, ParseError>>()?;

    Ok(ProgramParseTree { declarations })
}

fn parse_top(token_tree: &TokenTree) -> Result<Declaration, ParseError> {
    match &token_tree.val {
        TokenTreeVal::Leaf(_) => Err(ParseError {
            pos_start: token_tree.start_pos,
            pos_end: token_tree.end_pos,
            reason: ParseErrorReason::StrayAtom,
        }),
        TokenTreeVal::Branch(vec) => match &vec
            .get(0)
            .ok_or(ParseError {
                pos_start: token_tree.start_pos,
                pos_end: token_tree.end_pos,
                reason: ParseErrorReason::EmptyDeclaration,
            })?
            .val
        {
            TokenTreeVal::Leaf(token_non_paren) => match token_non_paren {
                TokenNonParen::Def => {
                    parse_def(token_tree.start_pos, token_tree.end_pos, &vec[1..])
                }
                TokenNonParen::Defgadt => {
                    parse_defgadt(token_tree.start_pos, token_tree.end_pos, &vec[1..])
                }
                TokenNonParen::Deftest => {
                    parse_deftest(token_tree.start_pos, token_tree.end_pos, &vec[1..])
                }
                _ => Err(ParseError {
                    pos_start: token_tree.start_pos,
                    pos_end: token_tree.end_pos,
                    reason: ParseErrorReason::TopLevelNotDeclaration,
                }),
            },
            TokenTreeVal::Branch(_) => Err(ParseError {
                pos_start: token_tree.start_pos,
                pos_end: token_tree.end_pos,
                reason: ParseErrorReason::TopLevelNotDeclaration,
            }),
        },
    }
}

fn parse_defgadt(
    start_pos: ParsePos,
    end_pos: ParsePos,
    vec: &[TokenTree],
) -> Result<Declaration, ParseError> {
    let name = parse_name(vec.get(0).ok_or(ParseError {
        pos_start: start_pos,
        pos_end: end_pos,
        reason: ParseErrorReason::ExpectedName,
    })?)?;
    let universe: u64 = parse_universe(vec.get(1).ok_or(ParseError {
        pos_start: start_pos,
        pos_end: end_pos,
        reason: ParseErrorReason::ExpectedUniverse,
    })?)?;
    let args = parse_args(vec.get(2).ok_or(ParseError {
        pos_start: start_pos,
        pos_end: end_pos,
        reason: ParseErrorReason::ExpectedArgs,
    })?)?;
    let constructors = parse_args(vec.get(3).ok_or(ParseError {
        pos_start: start_pos,
        pos_end: end_pos,
        reason: ParseErrorReason::ExpectedConstructors,
    })?)?;

    return Ok(Declaration::GadtDef(GadtDefinition {
        name,
        universe,
        arguments: args,
        constructors,
    }));
}

fn parse_def(
    start_pos: ParsePos,
    end_pos: ParsePos,
    vec: &[TokenTree],
) -> Result<Declaration, ParseError> {
    let name = parse_name(vec.get(0).ok_or(ParseError {
        pos_start: start_pos,
        pos_end: end_pos,
        reason: ParseErrorReason::ExpectedName,
    })?)?;
    let binding_type = parse_expr(vec.get(1).ok_or(ParseError {
        pos_start: start_pos,
        pos_end: end_pos,
        reason: ParseErrorReason::ExpectedDefinitionType,
    })?)?;
    let body = parse_expr(vec.get(2).ok_or(ParseError {
        pos_start: start_pos,
        pos_end: end_pos,
        reason: ParseErrorReason::ExpectedExpression,
    })?)?;
    return Ok(Declaration::BindingDef(BindingDefinition {
        name,
        binding_type,
        func_body: body,
    }));
}

fn parse_deftest(
    start_pos: ParsePos,
    end_pos: ParsePos,
    vec: &[TokenTree],
) -> Result<Declaration, ParseError> {
    let body = parse_expr(vec.get(0).ok_or(ParseError {
        pos_start: start_pos,
        pos_end: end_pos,
        reason: ParseErrorReason::ExpectedTestCaseExpression,
    })?)?;

    Ok(Declaration::TestCaseDef(body))
}

fn parse_universe(token_tree: &TokenTree) -> Result<u64, ParseError> {
    match &token_tree.val {
        TokenTreeVal::Leaf(x) => match x {
            TokenNonParen::NonKeyword(universe_number) => match parse_leaf_name(universe_number) {
                Expression::UniverseLit(x) => Ok(x),
                _ => Err(ParseError {
                    pos_start: token_tree.start_pos,
                    pos_end: token_tree.end_pos,
                    reason: ParseErrorReason::ExpectedUniverse,
                }),
            },
            _ => Err(ParseError {
                pos_start: token_tree.start_pos,
                pos_end: token_tree.end_pos,
                reason: ParseErrorReason::ExpectedUniverse,
            }),
        },
        TokenTreeVal::Branch(_) => Err(ParseError {
            pos_start: token_tree.start_pos,
            pos_end: token_tree.end_pos,
            reason: ParseErrorReason::ExpectedUniverse,
        }),
    }
}

fn parse_leaf_name(s: &str) -> Expression {
    if s == "true" {
        return Expression::BoolLit(true);
    }
    if s == "false" {
        return Expression::BoolLit(false);
    }

    if s.chars().next().unwrap().is_ascii_digit() {
        let (indicie, _) = s.char_indices().last().unwrap();
        match s.split_at(indicie) {
            (num, "i") => Expression::IntegerLit(num.parse().unwrap()),
            (num, "f") => Expression::FloatLit(num.parse().unwrap()),
            (num, "u") => Expression::UniverseLit(num.parse().unwrap()),
            _ => panic!("bad integer"),
        }
    } else {
        return Expression::Variable(s.to_owned());
    }
}

fn parse_expr(token_tree: &TokenTree) -> Result<Expression, ParseError> {
    match &token_tree.val {
        TokenTreeVal::Leaf(token_non_paren) => match token_non_paren {
            TokenNonParen::NonKeyword(var_name) => Ok(parse_leaf_name(var_name)),
            _ => Err(ParseError {
                pos_start: token_tree.start_pos,
                pos_end: token_tree.end_pos,
                reason: ParseErrorReason::VariableNameIsKeyword,
            }),
        },
        TokenTreeVal::Branch(sub_trees) => {
            let first = sub_trees.get(0).ok_or(ParseError {
                pos_start: token_tree.start_pos,
                pos_end: token_tree.end_pos,
                reason: ParseErrorReason::EmptyExpression,
            })?;
            match &first.val {
                TokenTreeVal::Leaf(token_non_paren) => match token_non_paren {
                    TokenNonParen::NonKeyword(func_name) => {
                        let args = sub_trees[1..]
                            .iter()
                            .map(parse_expr)
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(Expression::FunctionApplicationMultipleArgs {
                            func: Box::new(Expression::Variable(func_name.clone())),
                            args,
                        })
                    }
                    TokenNonParen::Let => {
                        let let_bindings: Vec<LetBind> =
                            parse_let_bindings(sub_trees.get(1).ok_or(ParseError {
                                pos_start: token_tree.start_pos,
                                pos_end: token_tree.end_pos,
                                reason: ParseErrorReason::ExpectedBindings,
                            })?)?;
                        let inner: Expression =
                            parse_expr(sub_trees.get(2).ok_or(ParseError {
                                pos_start: token_tree.start_pos,
                                pos_end: token_tree.end_pos,
                                reason: ParseErrorReason::ExpectedLetBindingExpression,
                            })?)?;

                        Ok(Expression::LetBinds {
                            bindings: let_bindings,
                            inner: Box::new(inner),
                        })
                    }
                    TokenNonParen::Lambda => {
                        let input_name: String =
                            parse_name(sub_trees.get(1).ok_or(ParseError {
                                pos_start: token_tree.start_pos,
                                pos_end: token_tree.end_pos,
                                reason: ParseErrorReason::ExpectedLambdaInputName,
                            })?)?;
                        let body: Expression = parse_expr(sub_trees.get(2).ok_or(ParseError {
                            pos_start: token_tree.start_pos,
                            pos_end: token_tree.end_pos,
                            reason: ParseErrorReason::ExpectedLambdaBodyExpression,
                        })?)?;

                        Ok(Expression::Lambda {
                            input: input_name,
                            body: Box::new(body),
                        })
                    }
                    TokenNonParen::Pi => {
                        let from_type: Argument =
                            parse_arg(sub_trees.get(1).ok_or(ParseError {
                                pos_start: token_tree.start_pos,
                                pos_end: token_tree.end_pos,
                                reason: ParseErrorReason::ExpectedDependentFunctionTypeArgument,
                            })?)?;
                        let to_type: Expression =
                            parse_expr(sub_trees.get(2).ok_or(ParseError {
                                pos_start: token_tree.start_pos,
                                pos_end: token_tree.end_pos,
                                reason: ParseErrorReason::ExpectedDependentFunctionTypeToTypeExpression,
                            })?)?;

                        Ok(Expression::DependentFunctionType {
                            type_from: Box::new(from_type),
                            type_to: Box::new(to_type),
                        })
                    }
                    TokenNonParen::Sigma => {
                        let first_type: Argument =
                            parse_arg(sub_trees.get(1).ok_or(ParseError {
                                pos_start: token_tree.start_pos,
                                pos_end: token_tree.end_pos,
                                reason: ParseErrorReason::ExpectedDependentProductTypeArgument,
                            })?)?;
                        let second_type: Expression =
                            parse_expr(sub_trees.get(2).ok_or(ParseError {
                                pos_start: token_tree.start_pos,
                                pos_end: token_tree.end_pos,
                                reason: ParseErrorReason::ExpectedDependentProductTypeSecondTypeExpression,
                            })?)?;

                        Ok(Expression::DependentProductType {
                            type_first: Box::new(first_type),
                            type_second: Box::new(second_type),
                        })
                    }
                    _ => Err(ParseError {
                        pos_start: token_tree.start_pos,
                        pos_end: token_tree.end_pos,
                        reason: ParseErrorReason::ExpressionCantStartWithThis,
                    }),
                },
                TokenTreeVal::Branch(_) => Err(ParseError {
                    pos_start: token_tree.start_pos,
                    pos_end: token_tree.end_pos,
                    reason: ParseErrorReason::FunctionApplicationIsntName,
                }),
            }
        }
    }
}

fn parse_let_bindings(token_tree: &TokenTree) -> Result<Vec<LetBind>, ParseError> {
    match &token_tree.val {
        TokenTreeVal::Leaf(_) => Err(ParseError {
            pos_start: token_tree.start_pos,
            pos_end: token_tree.end_pos,
            reason: ParseErrorReason::BindingsAreLeaf,
        }),
        TokenTreeVal::Branch(vec) => vec.into_iter().map(parse_let_binding).collect(),
    }
}

fn parse_let_binding(token_tree: &TokenTree) -> Result<LetBind, ParseError> {
    match &token_tree.val {
        TokenTreeVal::Leaf(_) => Err(ParseError {
            pos_start: token_tree.start_pos,
            pos_end: token_tree.end_pos,
            reason: ParseErrorReason::BindingIsLeaf,
        }),
        TokenTreeVal::Branch(vec) => {
            let name = parse_name(vec.get(0).ok_or(ParseError {
                pos_start: token_tree.start_pos,
                pos_end: token_tree.end_pos,
                reason: ParseErrorReason::ExpectedName,
            })?)?;
            let body = parse_expr(vec.get(1).ok_or(ParseError {
                pos_start: token_tree.start_pos,
                pos_end: token_tree.end_pos,
                reason: ParseErrorReason::ExpectedLetBindingBody,
            })?)?;

            return Ok(LetBind {
                name,
                value: Box::new(body),
            });
        }
    }
}

fn parse_arg(token_tree: &TokenTree) -> Result<Argument, ParseError> {
    match &token_tree.val {
        TokenTreeVal::Branch(vec) => {
            let colon = vec.get(0).ok_or(ParseError {
                pos_start: token_tree.start_pos,
                pos_end: token_tree.end_pos,
                reason: ParseErrorReason::EmptyArg,
            })?;
            if !matches!(colon.val, TokenTreeVal::Leaf(TokenNonParen::Colon)) {
                return Err(ParseError {
                    pos_start: token_tree.start_pos,
                    pos_end: token_tree.end_pos,
                    reason: ParseErrorReason::ArgDoesntStartWithColon,
                });
            }

            let varname = parse_name(vec.get(1).ok_or(ParseError {
                pos_start: token_tree.start_pos,
                pos_end: token_tree.end_pos,
                reason: ParseErrorReason::ExpectedName,
            })?)?;
            let vartype = parse_expr(vec.get(2).ok_or(ParseError {
                pos_start: token_tree.start_pos,
                pos_end: token_tree.end_pos,
                reason: ParseErrorReason::ExpectedArgTypeExpression,
            })?)?;

            Ok(Argument {
                varname,
                vartype,
            })
        }
        TokenTreeVal::Leaf(_) => Err(ParseError {
            pos_start: token_tree.start_pos,
            pos_end: token_tree.end_pos,
            reason: ParseErrorReason::ArgsAreLeaf,
        }),
    }
}

fn parse_args(token_tree: &TokenTree) -> Result<Vec<Argument>, ParseError> {
    match &token_tree.val {
        TokenTreeVal::Branch(vec) => Ok(vec
            .into_iter()
            .map(parse_arg)
            .collect::<Result<Vec<_>, _>>()?),
        TokenTreeVal::Leaf(_) => Err(ParseError {
            pos_start: token_tree.start_pos,
            pos_end: token_tree.end_pos,
            reason: ParseErrorReason::ArgIsLeaf,
        }),
    }
}

fn parse_name(token_tree: &TokenTree) -> Result<String, ParseError> {
    match &token_tree.val {
        TokenTreeVal::Leaf(TokenNonParen::NonKeyword(n)) => Ok(n.clone()),
        _ => Err(ParseError {
            pos_start: token_tree.start_pos,
            pos_end: token_tree.end_pos,
            reason: ParseErrorReason::ExpectedName,
        }),
    }
}

#[derive(Clone, Debug)]
pub struct ProgramParseTree {
    pub declarations: Vec<Declaration>,
}

#[derive(Clone, Debug)]
pub enum Declaration {
    BindingDef(BindingDefinition),
    GadtDef(GadtDefinition),
    TestCaseDef(Expression),
}

#[derive(Clone, Debug)]
pub struct GadtDefinition {
    pub name: String,
    pub universe: u64,
    pub arguments: Vec<Argument>,
    pub constructors: Vec<Argument>,
}

#[derive(Clone, Debug)]
pub struct BindingDefinition {
    pub name: String,
    pub binding_type: Expression,
    pub func_body: Expression,
}

#[derive(Clone, Debug)]
pub struct Argument {
    pub varname: String,
    pub vartype: Expression,
}

#[derive(Clone, Debug)]
pub enum Expression {
    DependentProductType {
        type_first: Box<Argument>,
        type_second: Box<Expression>,
    },
    DependentFunctionType {
        type_from: Box<Argument>,
        type_to: Box<Expression>,
    },
    FunctionApplicationMultipleArgs {
        func: Box<Expression>,
        args: Vec<Expression>,
    },

    Lambda {
        input: String,
        body: Box<Expression>,
    },

    Variable(String),
    IntegerLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    UniverseLit(u64),
    LetBinds {
        bindings: Vec<LetBind>,
        inner: Box<Expression>,
    },
}

#[derive(Clone, Debug)]
pub struct LetBind {
    pub name: String,
    pub value: Box<Expression>,
}

pub fn parse_program_from_file<P: AsRef<std::path::Path>>(
    filename: P,
) -> Result<ProgramParseTree, ParseError> {
    let code = &std::fs::read_to_string(filename).unwrap();
    let tokens = Tokens::new(code);
    let trees = make_token_trees(&mut tokens.peekable())?;
    return parse_program(trees);
}
