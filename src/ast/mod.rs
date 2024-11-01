use parsel::Spanned;

#[derive(serde::Serialize, Clone, Debug, PartialEq)]
pub struct Identifier(pub String);

#[derive(serde::Serialize, Clone, Debug)]
pub struct TypeName(pub String);

#[derive(serde::Serialize)]
pub struct Program {
    pub functions: Vec<FunctionDefinition>,
    #[serde(skip)]
    pub span: parsel::Span,
}

impl Program {
    pub fn find_func(&self, name: &str) -> &FunctionDefinition {
        for func in self.functions.iter() {
            if func.name.0 == name {
                return func;
            }
        }
        unreachable!()
    }
}

#[derive(serde::Serialize)]
pub struct FunctionDefinition {
    pub name: Identifier,
    pub arguments: Vec<(Identifier, TypeName)>,
    pub to_type: TypeName,
    pub body: Expression,
}

#[derive(serde::Serialize)]
pub enum Expression {
    Variable {
        ident: Identifier,
        #[serde(skip)]
        span: parsel::Span,
    },
    Integer(i64),
    Float(f64),
    Str(String),
    FuncApplication {
        func_name: Identifier,
        args: Vec<Expression>,
        #[serde(skip)]
        span: parsel::Span,
    },
    ExprWhere {
        bindings: Vec<LetBind>,
        inner: Box<Expression>,
    },
}

#[derive(serde::Serialize)]
pub struct LetBind {
    pub ident: Identifier,
    pub value: Expression,
}

impl From<crate::parser::ProgramParseTree> for Program {
    fn from(value: crate::parser::ProgramParseTree) -> Self {
        Self {
            span: value.functions.span(),
            functions: value.functions.into_iter().map(|x| x.into()).collect(),
        }
    }
}

impl From<crate::parser::FunctionDefinition> for FunctionDefinition {
    fn from(value: crate::parser::FunctionDefinition) -> FunctionDefinition {
        FunctionDefinition {
            name: Identifier(value.inner.name.to_string()),
            to_type: TypeName(value.inner.to_type.type_name.to_string()),
            arguments: value
                .inner
                .arguments
                .clone()
                .into_inner()
                .into_iter()
                .map(|x| x.into_inner())
                .map(|x: crate::parser::Argument| {
                    (
                        Identifier(x.varname.to_string()),
                        TypeName(x.vartype.type_name.to_string()),
                    )
                })
                .collect(),
            body: value.inner.func_body.clone().into_inner().into(),
        }
    }
}

impl From<crate::parser::Expression> for Expression {
    fn from(value: crate::parser::Expression) -> Expression {
        match value {
            crate::parser::Expression::Variable(ref x) => Expression::Variable {
                span: value.span(),
                ident: Identifier(x.to_string()),
            },
            crate::parser::Expression::IntegerLit(x) => Expression::Integer(x.into_inner()),
            crate::parser::Expression::StringLit(x) => Expression::Str(x.into_inner()),
            crate::parser::Expression::FloatLit(x) => Expression::Float(x.into_inner().into()),
            crate::parser::Expression::Addition(ref x) => Expression::FuncApplication {
                func_name: Identifier("__add".to_owned()),
                args: vec![
                    (*x.left_side.clone()).into(),
                    (*x.right_side.clone()).into(),
                ],
                span: value.span(),
            },
            crate::parser::Expression::Subtraction(ref x) => Expression::FuncApplication {
                func_name: Identifier("__sub".to_owned()),
                args: vec![
                    (*x.left_side.clone()).into(),
                    (*x.right_side.clone()).into(),
                ],
                span: value.span(),
            },
            crate::parser::Expression::Multiplication(ref x) => Expression::FuncApplication {
                func_name: Identifier("__mul".to_owned()),
                args: vec![
                    (*x.left_side.clone()).into(),
                    (*x.right_side.clone()).into(),
                ],
                span: value.span(),
            },
            crate::parser::Expression::Division(ref x) => Expression::FuncApplication {
                func_name: Identifier("__div".to_owned()),
                args: vec![
                    (*x.left_side.clone()).into(),
                    (*x.right_side.clone()).into(),
                ],
                span: value.span(),
            },
            crate::parser::Expression::Equality(ref x) => Expression::FuncApplication {
                func_name: Identifier("__eq".to_owned()),
                args: vec![
                    (*x.left_side.clone()).into(),
                    (*x.right_side.clone()).into(),
                ],
                span: value.span(),
            },
            crate::parser::Expression::GreaterThan(ref x) => Expression::FuncApplication {
                func_name: Identifier("__gt".to_owned()),
                args: vec![
                    (*x.left_side.clone()).into(),
                    (*x.right_side.clone()).into(),
                ],
                span: value.span(),
            },
            crate::parser::Expression::LessThan(ref x) => Expression::FuncApplication {
                func_name: Identifier("__lt".to_owned()),
                args: vec![
                    (*x.left_side.clone()).into(),
                    (*x.right_side.clone()).into(),
                ],
                span: value.span(),
            },
            crate::parser::Expression::ExprWhere {
                bindings,
                where_token: _,
                inner,
            } => {
                let bindings = bindings
                    .clone()
                    .into_inner()
                    .into_iter()
                    .map(|x| x.into_inner())
                    .map(|x| LetBind {
                        ident: Identifier(x.name.to_string()),
                        value: (*x.value).into(),
                    })
                    .collect();

                Expression::ExprWhere {
                    bindings,
                    inner: Box::new((*inner).into()),
                }
            }
        }
    }
}
