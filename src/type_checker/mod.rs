use crate::parser;

type Res<A> = Result<A, Box<dyn std::error::Error>>;

#[derive(Clone)]
pub enum Type {
    Expr(SimpleType),
    Function {
        from: Vec<SimpleType>,
        to: SimpleType,
    },
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
        first: (parsel::ast::Ident, Type),
        rest: Box<TypeEnv>,
    },
}

impl TypeEnv {
    pub fn empty() -> Self {
        Self::End
    }
    fn add_type(&self, ident: parsel::ast::Ident, type_v: Type) -> Self {
        TypeEnv::Rest {
            first: (ident, type_v),
            rest: Box::new(self.clone()),
        }
    }
    fn find_var_type(&self, ident: &parsel::ast::Ident) -> Option<SimpleType> {
        match self {
            TypeEnv::End => None,
            TypeEnv::Rest { first, rest } => {
                if &first.0 == ident {
                    match first.1 {
                        Type::Expr(e) => Some(e),
                        Type::Function { from: _, to: _ } => rest.find_var_type(ident),
                    }
                } else {
                    rest.find_var_type(ident)
                }
            }
        }
    }
}

impl TryFrom<parser::VarType> for SimpleType {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: parser::VarType) -> Res<Self> {
        match value.type_name.to_string().as_str() {
            "int" => Ok(SimpleType::Int),
            "float" => Ok(SimpleType::Float),
            "bool" => Ok(SimpleType::Bool),
            a => Err(format!("unknown type: {}", a).into()),
        }
    }
}

pub fn type_check_program(program: &parser::Program) -> Res<TypeEnv> {
    Ok(program
        .functions
        .clone()
        .into_iter()
        .fold(TypeEnv::empty(), |acc, function| {
            type_check_func(acc, &function).unwrap()
        }))
}

fn type_check_func(env: TypeEnv, func: &parser::FunctionDefinition) -> Res<TypeEnv> {
    let env_with_arguments = func
        .inner
        .arguments
        .clone()
        .into_inner()
        .clone()
        .into_iter()
        .map(|x| x.into_inner())
        .map(|x| Ok((x.varname, Type::Expr(x.vartype.try_into()?))))
        .collect::<Res<Vec<_>>>()?
        .into_iter()
        .fold(env.clone(), |acc, (identifier, type_v)| {
            acc.add_type(identifier, type_v)
        });
    let expr_type = find_expr_type(
        env_with_arguments,
        func.inner.func_body.clone().into_inner(),
    )?;

    if expr_type != func.inner.to_type.clone().try_into()? {
        return Err("func type signature mismatch".into());
    }
    Ok(TypeEnv::Rest {
        first: (
            func.clone().inner.name.clone(),
            Type::Function {
                from: vec![],
                to: func
                    .clone()
                    .inner
                    .to_type
                    .clone()
                    .try_into()
                    .unwrap(),
            },
        ),
        rest: Box::new(env),
    })
}

fn find_expr_type(env: TypeEnv, expr: parser::Expression) -> Res<SimpleType> {
    match expr {
        parser::Expression::Variable(var) => Ok(env.find_var_type(&var).ok_or::<Box<
            dyn std::error::Error,
        >>(
            format!("unknown variable type: {}", var).into(),
        )?),
        parser::Expression::IntegerLit(_) => Ok(SimpleType::Int),
        parser::Expression::StringLit(_) => Ok(SimpleType::String),
        parser::Expression::FloatLit(_) => Ok(SimpleType::Float),
        parser::Expression::Addition(x) => {
            let thing = x.into_inner();
            find_arithmtic_type(env, *thing.left_side, *thing.right_side)
        }
        parser::Expression::Subtraction(x) => {
            let thing = x.into_inner();
            find_arithmtic_type(env, *thing.left_side, *thing.right_side)
        }
        parser::Expression::Multiplication(x) => {
            let thing = x.into_inner();
            find_arithmtic_type(env, *thing.left_side, *thing.right_side)
        }
        parser::Expression::Division(x) => {
            let thing = x.into_inner();
            find_arithmtic_type(env, *thing.left_side, *thing.right_side)
        }
        parser::Expression::Equality(x) => {
            let thing = x.into_inner();
            validate_comparison(env, *thing.left_side, *thing.right_side)?;
            Ok(SimpleType::Bool)
        }
        parser::Expression::GreaterThan(x) => {
            let thing = x.into_inner();
            validate_comparison(env, *thing.left_side, *thing.right_side)?;
            Ok(SimpleType::Bool)
        }
        parser::Expression::LessThan(x) => {
            let thing = x.into_inner();
            validate_comparison(env, *thing.left_side, *thing.right_side)?;
            Ok(SimpleType::Bool)
        }
        parser::Expression::ExprWhere {
            bindings,
            where_token: _,
            inner,
        } => {
            let env_with_bindings = bindings
                .into_inner()
                .clone()
                .into_iter()
                .map(|x| x.into_inner())
                .map(|x| Ok((x.name, Type::Expr(find_expr_type(env.clone(), *x.value)?))))
                .collect::<Res<Vec<_>>>()?
                .into_iter()
                .fold(env, |acc, (identifier, type_v)| {
                    acc.add_type(identifier, type_v)
                });

            find_expr_type(env_with_bindings, *inner)
        }
    }
}

fn validate_comparison(
    env: TypeEnv,
    left: parser::Expression,
    right: parser::Expression,
) -> Res<()> {
    match (
        find_expr_type(env.clone(), left)?,
        find_expr_type(env.clone(), right)?,
    ) {
        (SimpleType::Int, SimpleType::Int) => Ok(()),
        (SimpleType::Int, SimpleType::Float) => Ok(()),
        (SimpleType::Float, SimpleType::Int) => Ok(()),
        (SimpleType::Float, SimpleType::Float) => Ok(()),
        (left_type, right_type) => Err(format!(
            "cannot perform comparison on types {:?} and {:?}",
            left_type, right_type
        )
        .into()),
    }
}

fn find_arithmtic_type(
    env: TypeEnv,
    left: parser::Expression,
    right: parser::Expression,
) -> Res<SimpleType> {
    match (
        find_expr_type(env.clone(), left)?,
        find_expr_type(env.clone(), right)?,
    ) {
        (SimpleType::Int, SimpleType::Int) => Ok(SimpleType::Int),
        (SimpleType::Int, SimpleType::Float) => Ok(SimpleType::Float),
        (SimpleType::Float, SimpleType::Int) => Ok(SimpleType::Float),
        (SimpleType::Float, SimpleType::Float) => Ok(SimpleType::Float),
        (left_type, right_type) => Err(format!(
            "cannot perform arithmetic on types {:?} and {:?}",
            left_type, right_type
        )
        .into()),
    }
}
