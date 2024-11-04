use parsel::Spanned;

#[derive(serde::Serialize, Clone, Debug, PartialEq)]
pub struct Identifier(pub String);

#[derive(serde::Serialize, Clone, Debug)]
pub struct TypeName(pub String);

#[derive(serde::Serialize)]
pub struct Program {
    pub functions: Vec<FunctionDefinition>,
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

struct AstConversionState {
    next_lit_id: LitId,
}

#[derive(serde::Serialize)]
pub struct FunctionDefinition {
    pub name: Identifier,
    pub arguments: Vec<(Identifier, TypeName)>,
    pub to_type: TypeName,
    pub body: Expression,
}

#[derive(serde::Serialize, Copy, Clone)]
pub struct LitId(usize);

#[derive(serde::Serialize)]
pub enum Expression {
    Variable {
        ident: Identifier,
        #[serde(skip)]
        span: parsel::Span,
    },
    Integer(i64, LitId),
    Float(f64, LitId),
    Str(String, LitId),
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

pub fn make_program(parse_tree: crate::parser::ProgramParseTree) -> Program {
    let mut state = AstConversionState {
        next_lit_id: LitId(0),
    };

    Program {
        functions: parse_tree
            .functions
            .into_iter()
            .map(|x| make_function_definition(&mut state, x))
            .collect(),
    }
}

fn make_function_definition(
    state: &mut AstConversionState,
    value: crate::parser::FunctionDefinition,
) -> FunctionDefinition {
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
        body: make_expression(state, value.inner.func_body.clone().into_inner()),
    }
}

fn make_expression(state: &mut AstConversionState, value: crate::parser::Expression) -> Expression {
    match value {
        crate::parser::Expression::Variable(ref x) => Expression::Variable {
            span: value.span(),
            ident: Identifier(x.to_string()),
        },
        crate::parser::Expression::IntegerLit(x) => {
            let output = Expression::Integer(x.into_inner(), state.next_lit_id);
            state.next_lit_id.0 += 1;
            output
        }
        crate::parser::Expression::StringLit(x) => {
            let output = Expression::Str(x.into_inner(), state.next_lit_id);
            state.next_lit_id.0 += 1;
            output
        }
        crate::parser::Expression::FloatLit(x) => {
            let output = Expression::Float(*x.into_inner(), state.next_lit_id);
            state.next_lit_id.0 += 1;
            output
        }
        crate::parser::Expression::Addition(ref x) => Expression::FuncApplication {
            func_name: Identifier("__add".to_owned()),
            args: vec![
                make_expression(state, *x.left_side.clone()),
                make_expression(state, *x.right_side.clone()),
            ],
            span: value.span(),
        },
        crate::parser::Expression::Subtraction(ref x) => Expression::FuncApplication {
            func_name: Identifier("__sub".to_owned()),
            args: vec![
                make_expression(state, *x.left_side.clone()),
                make_expression(state, *x.right_side.clone()),
            ],
            span: value.span(),
        },
        crate::parser::Expression::Multiplication(ref x) => Expression::FuncApplication {
            func_name: Identifier("__mul".to_owned()),
            args: vec![
                make_expression(state, *x.left_side.clone()),
                make_expression(state, *x.right_side.clone()),
            ],
            span: value.span(),
        },
        crate::parser::Expression::Division(ref x) => Expression::FuncApplication {
            func_name: Identifier("__div".to_owned()),
            args: vec![
                make_expression(state, *x.left_side.clone()),
                make_expression(state, *x.right_side.clone()),
            ],
            span: value.span(),
        },
        crate::parser::Expression::Equality(ref x) => Expression::FuncApplication {
            func_name: Identifier("__eq".to_owned()),
            args: vec![
                make_expression(state, *x.left_side.clone()),
                make_expression(state, *x.right_side.clone()),
            ],
            span: value.span(),
        },
        crate::parser::Expression::GreaterThan(ref x) => Expression::FuncApplication {
            func_name: Identifier("__gt".to_owned()),
            args: vec![
                make_expression(state, *x.left_side.clone()),
                make_expression(state, *x.right_side.clone()),
            ],
            span: value.span(),
        },
        crate::parser::Expression::LessThan(ref x) => Expression::FuncApplication {
            func_name: Identifier("__lt".to_owned()),
            args: vec![
                make_expression(state, *x.left_side.clone()),
                make_expression(state, *x.right_side.clone()),
            ],
            span: value.span(),
        },
        crate::parser::Expression::FunctionApplication {
            ref func_name,
            ref args,
        } => Expression::FuncApplication {
            span: value.span(),
            func_name: Identifier(func_name.to_string()),
            args: args
                .clone()
                .into_inner()
                .into_iter()
                .map(|x| make_expression(state, *x))
                .collect(),
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
                    value: make_expression(state, *x.value),
                })
                .collect();

            Expression::ExprWhere {
                bindings,
                inner: Box::new(make_expression(state, *inner)),
            }
        }
    }
}
