use parsel::Display;

use crate::ast;

#[derive(Debug)]
pub enum TypeError {
    UnknownType(ast::TypeName),
    FuncTypeMismatch {
        body_type_expected: SimpleType,
        body_type_found: SimpleType,
    },
    UnknownVariable {
        span: parsel::Span,
        ident: ast::Identifier,
    },
    BadComparison {
        span: parsel::Span,
        left: SimpleType,
        right: SimpleType,
    },
    BadArithmetic {
        left: SimpleType,
        right: SimpleType,
        span: parsel::Span,
    },
    WrongNumArgs {
        expected: usize,
        got: usize,
        span: parsel::Span,
    },
}

impl Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::UnknownType(typename) => write!(f, "unknown type {}", typename.0),
            TypeError::FuncTypeMismatch {
                body_type_expected,
                body_type_found,
            } => write!(
                f,
                "function type mismatch, expected {:?}, found {:?}",
                body_type_expected, body_type_found
            ),
            TypeError::UnknownVariable { span, ident } => {
                let location = span.start();
                write!(
                    f,
                    "Unknown variable on line {} col {}: {}",
                    location.line, location.column, ident.0
                )
            }
            TypeError::BadComparison { span, left, right } => {
                let location = span.start();
                write!(
                    f,
                    "line {} col {}: could not compare types {:?} and {:?}",
                    location.line, location.column, left, right
                )
            }
            TypeError::BadArithmetic { left, right, span } => {
                let location = span.start();
                write!(
                    f,
                    "line {} col {}: could not apply arithmetic on types {:?} and {:?}",
                    location.line, location.column, left, right
                )
            }
            TypeError::WrongNumArgs {
                expected,
                got,
                span,
            } => {
                let location = span.start();
                write!(f,
                    "line {} col {}: wrong number of arguments for function, expected {:?} but got {:?}",
                    location.line, location.column, expected, got
                )
            }
        }
    }
}

type Res<A> = Result<A, TypeError>;

#[derive(Clone, Debug)]
pub enum Type {
    Expr(SimpleType),
    Function(FunctionType),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionType {
    from: Vec<SimpleType>,
    to: SimpleType,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SimpleType {
    Int,
    Float,
    Bool,
    String, // List(u32),
            // Dict
}

#[derive(Clone)]
pub enum TypeEnv {
    End,
    Rest {
        first: (ast::Identifier, Type),
        rest: Box<TypeEnv>,
    },
}

impl TypeEnv {
    pub fn empty() -> Self {
        Self::End
    }
    fn add_type(&self, ident: ast::Identifier, type_v: Type) -> Self {
        TypeEnv::Rest {
            first: (ident, type_v),
            rest: Box::new(self.clone()),
        }
    }
    fn find_var_type(&self, ident: &ast::Identifier) -> Option<SimpleType> {
        match self {
            TypeEnv::End => None,
            TypeEnv::Rest { first, rest } => {
                if &first.0 == ident {
                    match first.1 {
                        Type::Expr(e) => Some(e),
                        Type::Function(_) => rest.find_var_type(ident),
                    }
                } else {
                    rest.find_var_type(ident)
                }
            }
        }
    }
    fn find_function_type(&self, ident: &ast::Identifier) -> Option<FunctionType> {
        match self {
            TypeEnv::End => None,
            TypeEnv::Rest { first, rest } => {
                if &first.0 == ident {
                    match &first.1 {
                        Type::Expr(_) => rest.find_function_type(ident),
                        Type::Function(func_type) => Some(func_type.clone()),
                    }
                } else {
                    rest.find_function_type(ident)
                }
            }
        }
    }
}

impl TryFrom<&ast::TypeName> for SimpleType {
    type Error = TypeError;

    fn try_from(value: &ast::TypeName) -> Res<Self> {
        match value.0.as_str() {
            "int" => Ok(SimpleType::Int),
            "float" => Ok(SimpleType::Float),
            "bool" => Ok(SimpleType::Bool),
            _ => Err(TypeError::UnknownType(value.clone())),
        }
    }
}

pub fn type_check_program(program: &ast::Program) -> Res<TypeEnv> {
    Ok(program
        .functions
        .iter()
        .fold(TypeEnv::empty(), |acc, function| {
            type_check_func(acc, function).unwrap()
        }))
}

fn type_check_func(env: TypeEnv, func: &ast::FunctionDefinition) -> Res<TypeEnv> {
    let env_with_arguments = func
        .arguments
        .iter()
        .map(|x| (x.0.clone(), Type::Expr((&x.1).try_into().unwrap())))
        .fold(env.clone(), |acc, (identifier, type_v)| {
            acc.add_type(identifier, type_v)
        });
    let expr_type = find_expr_type(env_with_arguments, &func.body)?;

    let expected_to_type = (&func.to_type).try_into()?;

    if expr_type != expected_to_type {
        return Err(TypeError::FuncTypeMismatch {
            body_type_expected: expected_to_type,
            body_type_found: expr_type,
        });
    }
    Ok(TypeEnv::Rest {
        first: (
            func.name.clone(),
            Type::Function(FunctionType {
                from: func
                    .arguments
                    .iter()
                    .map(|x| (&x.1).try_into().unwrap())
                    .collect(),
                to: (&func.to_type).try_into().unwrap(),
            }),
        ),
        rest: Box::new(env),
    })
}

fn find_expr_type(env: TypeEnv, expr: &ast::Expression) -> Res<SimpleType> {
    match expr {
        ast::Expression::Variable { ident, span } => {
            Ok(env.find_var_type(ident).ok_or(TypeError::UnknownVariable {
                span: *span,
                ident: ident.clone(),
            })?)
        }
        ast::Expression::Integer(_) => Ok(SimpleType::Int),
        ast::Expression::Str(_) => Ok(SimpleType::String),
        ast::Expression::Float(_) => Ok(SimpleType::Float),
        ast::Expression::FuncApplication {
            func_name,
            args,
            span,
        } => {
            if ["__add", "__sub"].contains(&func_name.0.as_str()) {
                assert!(args.len() == 2);
                return find_arithmtic_type(env, &args[0], &args[1], *span);
            }
            if ["__gt", "__lt"].contains(&func_name.0.as_str()) {
                assert!(args.len() == 2);
                validate_comparison(env, &args[0], &args[1], *span).unwrap();
                return Ok(SimpleType::Bool);
            }

            let function_type = env.find_function_type(func_name).unwrap();
            if args.len() != function_type.from.len() {
                return Err(TypeError::WrongNumArgs {
                    expected: function_type.from.len(),
                    got: args.len(),
                    span: *span,
                });
            }

            for (expr, expected) in args.iter().zip(function_type.from.into_iter()) {
                assert!(find_expr_type(env.clone(), expr)? == expected);
            }

            Ok(function_type.to)
        }
        ast::Expression::ExprWhere { bindings, inner } => {
            let env_with_bindings = bindings
                .iter()
                .map(|x| {
                    (
                        &x.ident,
                        Type::Expr(find_expr_type(env.clone(), &x.value).unwrap()),
                    )
                })
                .fold(env.clone(), |acc, (identifier, type_v)| {
                    acc.add_type(identifier.clone(), type_v)
                });

            find_expr_type(env_with_bindings, inner)
        }
    }
}

fn validate_comparison(
    env: TypeEnv,
    left: &ast::Expression,
    right: &ast::Expression,
    span: parsel::Span,
) -> Res<()> {
    match (
        find_expr_type(env.clone(), left)?,
        find_expr_type(env.clone(), right)?,
    ) {
        (SimpleType::Int, SimpleType::Int) => Ok(()),
        (SimpleType::Int, SimpleType::Float) => Ok(()),
        (SimpleType::Float, SimpleType::Int) => Ok(()),
        (SimpleType::Float, SimpleType::Float) => Ok(()),
        (left, right) => Err(TypeError::BadComparison { left, right, span }),
    }
}

fn find_arithmtic_type(
    env: TypeEnv,
    left: &ast::Expression,
    right: &ast::Expression,
    span: parsel::Span,
) -> Res<SimpleType> {
    match (
        find_expr_type(env.clone(), left)?,
        find_expr_type(env.clone(), right)?,
    ) {
        (SimpleType::Int, SimpleType::Int) => Ok(SimpleType::Int),
        (SimpleType::Int, SimpleType::Float) => Ok(SimpleType::Float),
        (SimpleType::Float, SimpleType::Int) => Ok(SimpleType::Float),
        (SimpleType::Float, SimpleType::Float) => Ok(SimpleType::Float),
        (left, right) => Err(TypeError::BadArithmetic { left, right, span }),
    }
}
