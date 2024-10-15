#[derive(serde::Serialize)]
pub struct FunctionDefinitionJson {
    pub name: String,
    pub func_type: FuncTypeJson,
    pub arguments: Vec<(String, String)>,
    pub body: ExpressionJson,
}

#[derive(serde::Serialize)]
pub struct FuncTypeJson {
    pub from: String,
    pub to: String,
}

#[derive(serde::Serialize)]
pub enum ExpressionJson {
    Variable(String),
    Integer(i64),
    Float(f64),
    Str(String),
    Addition(Box<ExpressionJson>, Box<ExpressionJson>),
    Subtraction(Box<ExpressionJson>, Box<ExpressionJson>),
    Multiplication(Box<ExpressionJson>, Box<ExpressionJson>),
    Division(Box<ExpressionJson>, Box<ExpressionJson>),
    Equality(Box<ExpressionJson>, Box<ExpressionJson>),
    GreaterThan(Box<ExpressionJson>, Box<ExpressionJson>),
    LessThan(Box<ExpressionJson>, Box<ExpressionJson>),
    ExprWhere {
        bindings: Vec<LetBindJson>,
        inner: Box<ExpressionJson>,
    },
}

#[derive(serde::Serialize)]
pub struct LetBindJson {
    pub ident: String,
    pub value: ExpressionJson,
}

impl From<crate::parser::FunctionDefinition> for FunctionDefinitionJson {
    fn from(value: crate::parser::FunctionDefinition) -> FunctionDefinitionJson {
        FunctionDefinitionJson {
            name: value.inner.name.to_string(),
            func_type: value.inner.func_type.clone().into_inner().into(),
            arguments: value
                .inner
                .arguments
                .clone()
                .into_inner()
                .into_iter()
                .map(|x| x.into_inner())
                .map(|x: crate::parser::Argument| {
                    (x.varname.to_string(), x.vartype.name.to_string())
                })
                .collect(),
            body: value.inner.func_body.clone().into_inner().into(),
        }
    }
}

impl From<crate::parser::FuncType> for FuncTypeJson {
    fn from(value: crate::parser::FuncType) -> FuncTypeJson {
        FuncTypeJson {
            from: value.from_type.name.to_string(),
            to: value.to_type.name.to_string(),
        }
    }
}

impl From<crate::parser::Expression> for ExpressionJson {
    fn from(value: crate::parser::Expression) -> ExpressionJson {
        match value {
            crate::parser::Expression::Variable(x) => ExpressionJson::Variable(x.name.to_string()),
            crate::parser::Expression::IntegerLit(x) => ExpressionJson::Integer(x.into_inner()),
            crate::parser::Expression::StringLit(x) => ExpressionJson::Str(x.into_inner()),
            crate::parser::Expression::FloatLit(x) => ExpressionJson::Float(x.into_inner().into()),
            crate::parser::Expression::Addition(x) => ExpressionJson::Addition(
                Box::new((*x.left_side.clone()).into()),
                Box::new((*x.right_side.clone()).into()),
            ),
            crate::parser::Expression::Subtraction(x) => ExpressionJson::Subtraction(
                Box::new((*x.left_side.clone()).into()),
                Box::new((*x.right_side.clone()).into()),
            ),
            crate::parser::Expression::Multiplication(x) => ExpressionJson::Multiplication(
                Box::new((*x.left_side.clone()).into()),
                Box::new((*x.right_side.clone()).into()),
            ),
            crate::parser::Expression::Division(x) => ExpressionJson::Division(
                Box::new((*x.left_side.clone()).into()),
                Box::new((*x.right_side.clone()).into()),
            ),
            crate::parser::Expression::Equality(x) => ExpressionJson::Equality(
                Box::new((*x.left_side.clone()).into()),
                Box::new((*x.right_side.clone()).into()),
            ),
            crate::parser::Expression::GreaterThan(x) => ExpressionJson::GreaterThan(
                Box::new((*x.left_side.clone()).into()),
                Box::new((*x.right_side.clone()).into()),
            ),
            crate::parser::Expression::LessThan(x) => ExpressionJson::LessThan(
                Box::new((*x.left_side.clone()).into()),
                Box::new((*x.right_side.clone()).into()),
            ),
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
                    .map(|x| LetBindJson {
                        ident: x.name.to_string(),
                        value: (*x.value).into(),
                    })
                    .collect();

                ExpressionJson::ExprWhere {
                    bindings,
                    inner: Box::new((*inner).into()),
                }
            }
        }
    }
}
