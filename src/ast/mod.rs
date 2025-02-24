pub mod eval;

#[derive(serde::Serialize, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Identifier(pub String);

#[derive(serde::Serialize, Clone, Debug)]
pub struct TypeName(pub String);

#[derive(serde::Serialize, Clone, Debug)]
pub struct Argument<IntType, FloatType, BoolType> {
    pub name: Identifier,
    pub arg_type: Expression<IntType, FloatType, BoolType>,
}

#[derive(serde::Serialize, Clone)]
pub struct Program<IntType, FloatType, BoolType> {
    pub global_bindings: Vec<Binding<IntType, FloatType, BoolType>>,
    pub test_cases: Vec<Expression<IntType, FloatType, BoolType>>,
    pub gadts: Vec<GadtDefinition<IntType, FloatType, BoolType>>,
    pub num_ids: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct ProgramInitFunctions<IntType, FloatType, BoolType> {
    pub make_int: fn(i64, LitId, usize) -> IntType,
    pub make_float: fn(f64, LitId, usize) -> FloatType,
    pub make_bool: fn(bool, LitId, usize) -> BoolType,
}

struct AstConversionState {
    next_lit_id: LitId,
    total: usize,
}

#[derive(serde::Serialize, Clone)]
pub struct GadtDefinition<IntType, FloatType, BoolType> {
    pub name: Identifier,
    pub gadt_type: Expression<IntType, FloatType, BoolType>,
    pub constructors: Vec<Argument<IntType, FloatType, BoolType>>,
}

#[derive(serde::Serialize, Clone)]
pub struct Binding<IntType, FloatType, BoolType> {
    pub name: Identifier,
    pub elem_type: Expression<IntType, FloatType, BoolType>,
    pub value: Expression<IntType, FloatType, BoolType>,
}

#[derive(serde::Serialize, Clone)]
pub struct DependentFunctionType<IntType, FloatType, BoolType> {
    pub from_type: (Identifier, Expression<IntType, FloatType, BoolType>),
    pub to_type: Expression<IntType, FloatType, BoolType>,
}

#[derive(serde::Serialize, Copy, Clone, Debug)]
pub struct LitId(pub Option<usize>);

#[derive(serde::Serialize, Clone, Debug)]
pub enum Expression<IntType, FloatType, BoolType> {
    Intrinsic,
    Variable {
        ident: Identifier,
    },
    DependentProductType {
        type_first: Box<Argument<IntType, FloatType, BoolType>>,
        type_second: Box<Expression<IntType, FloatType, BoolType>>,
    },
    DependentFunctionType {
        type_from: Box<Argument<IntType, FloatType, BoolType>>,
        type_to: Box<Expression<IntType, FloatType, BoolType>>,
    },

    Integer(IntType, LitId),
    Float(FloatType, LitId),
    Bool(BoolType, LitId),
    Universe(u64),
    FuncApplicationMultipleArgs {
        func: Box<Expression<IntType, FloatType, BoolType>>,
        args: Vec<Expression<IntType, FloatType, BoolType>>,
    },
    ExprLetBinding {
        bindings: Vec<LetBind<IntType, FloatType, BoolType>>,
        inner: Box<Expression<IntType, FloatType, BoolType>>,
    },

    Lambda {
        input: Box<Argument<IntType, FloatType, BoolType>>,
        body: Box<Expression<IntType, FloatType, BoolType>>,
    },
}

enum TypeEnv<'a, IntType, FloatType, BoolType> {
    Next {
        first: (&'a Identifier, &'a Expression<IntType, FloatType, BoolType>),
        rest: &'a TypeEnv<'a, IntType, FloatType, BoolType>,
    },
    End,
}

impl<'a, IntType, FloatType, BoolType> TypeEnv<'a, IntType, FloatType, BoolType> {
    pub fn lookup_type(&self, name: &Identifier) -> Expression<IntType, FloatType, BoolType>
    where
        BoolType: Clone,
        FloatType: Clone,
        IntType: Clone,
    {
        match name.0.as_str() {
            "__add_int" => Expression::DependentFunctionType {
                type_from: Box::new(Argument {
                    name: Identifier("__unused".to_owned()),
                    arg_type: Expression::Variable {
                        ident: Identifier("int".to_owned()),
                    },
                }),
                type_to: Box::new(Expression::DependentFunctionType {
                    type_from: Box::new(Argument {
                        name: Identifier("__unused".to_owned()),
                        arg_type: Expression::Variable {
                            ident: Identifier("int".to_owned()),
                        },
                    }),
                    type_to: Box::new(Expression::Variable {
                        ident: Identifier("int".to_owned()),
                    }),
                }),
            },
            "int" => Expression::Universe(0),
            _ => match self {
                TypeEnv::Next { first, rest } => {
                    if first.0 == name {
                        return first.1.clone();
                    }
                    return rest.lookup_type(name);
                }
                TypeEnv::End => panic!("unknown type: {}", name.0),
            },
        }
    }
}

impl<IntType, FloatType, BoolType> Expression<IntType, FloatType, BoolType> {
    fn get_type(&self, env: &TypeEnv<IntType, FloatType, BoolType>) -> Self
    where
        BoolType: Clone + std::fmt::Debug,
        FloatType: Clone + std::fmt::Debug,
        IntType: Clone + std::fmt::Debug,
    {
        match &self {
            Expression::Intrinsic => todo!(),
            Expression::Variable { ident } => env.lookup_type(ident),
            Expression::DependentProductType {
                type_first,
                type_second,
            } => {
                let first_type_type = type_first.arg_type.get_type(env);

                let new_env = TypeEnv::Next {
                    first: (&type_first.name, &first_type_type),
                    rest: env,
                };

                let second_type_type = type_second.get_type(&new_env);

                match (first_type_type, second_type_type) {
                    (Expression::Universe(x), Expression::Universe(y)) => {
                        Expression::Universe(std::cmp::max(x, y) + 1)
                    }
                    _ => todo!(),
                }
            }
            Expression::DependentFunctionType { type_from, type_to } => {
                let first_type_type = type_from.arg_type.get_type(env);

                let new_env = TypeEnv::Next {
                    first: (&type_from.name, &first_type_type),
                    rest: env,
                };

                let second_type_type = type_to.get_type(&new_env);

                match (first_type_type, second_type_type) {
                    (Expression::Universe(x), Expression::Universe(y)) => {
                        Expression::Universe(std::cmp::max(x, y) + 1)
                    }
                    _ => todo!(),
                }
            },
            Expression::Integer(_, _) => Expression::Variable {
                ident: Identifier("int".to_owned()),
            },
            Expression::Float(_, _) => todo!(),
            Expression::Bool(_, _) => todo!(),
            Expression::Universe(x) => Expression::Universe(x + 1),
            Expression::FuncApplicationMultipleArgs { func, args } => {
                find_function_application_type(&func.get_type(env), args, env)
            }
            Expression::ExprLetBinding { bindings, inner } => todo!(),
            Expression::Lambda { input, body } => {
                let new_env = TypeEnv::Next {
                    first: (&input.name, &input.arg_type),
                    rest: env,
                };

                let body_type = body.get_type(&new_env);

                Expression::DependentFunctionType {
                    type_from: input.clone(),
                    type_to: Box::new(body_type),
                }
            }
        }
    }
}

#[derive(serde::Serialize, Clone, Debug)]
pub struct LetBind<IntType, FloatType, BoolType> {
    pub ident: Identifier,
    pub value: Expression<IntType, FloatType, BoolType>,
}

pub fn make_program<IntType, FloatType, BoolType>(
    parse_tree: &crate::parser::ProgramParseTree,
    funcs: ProgramInitFunctions<IntType, FloatType, BoolType>,
) -> Program<IntType, FloatType, BoolType>
where
    BoolType: Clone + std::fmt::Debug,
    FloatType: Clone + std::fmt::Debug,
    IntType: Clone + std::fmt::Debug,
{
    let state = AstConversionState {
        next_lit_id: LitId(Some(0)),
        total: 0,
    };

    let counter = make_program_inner(parse_tree, NULL_AST_INIT, state);

    make_program_inner(
        parse_tree,
        funcs,
        AstConversionState {
            next_lit_id: LitId(Some(0)),
            total: counter.num_ids,
        },
    )
}

pub const NULL_AST_INIT: ProgramInitFunctions<i64, f64, bool> = ProgramInitFunctions {
    make_int,
    make_float,
    make_bool,
};

fn make_int(x: i64, _: LitId, _: usize) -> i64 {
    x
}

fn make_float(x: f64, _: LitId, _: usize) -> f64 {
    x
}
fn make_bool(x: bool, _: LitId, _: usize) -> bool {
    x
}

fn make_program_inner<IntType, FloatType, BoolType>(
    parse_tree: &crate::parser::ProgramParseTree,
    funcs: ProgramInitFunctions<IntType, FloatType, BoolType>,
    mut state: AstConversionState,
) -> Program<IntType, FloatType, BoolType>
where
    BoolType: Clone + std::fmt::Debug,
    FloatType: Clone + std::fmt::Debug,
    IntType: Clone + std::fmt::Debug,
{
    let bindings: Vec<Binding<IntType, FloatType, BoolType>> = parse_tree
        .declarations
        .iter()
        .filter_map(|x| match x {
            crate::parser::Declaration::BindingDef(f) => {
                Some(make_definition(&mut state, &funcs, f))
            }
            crate::parser::Declaration::TestCaseDef(_) => None,
            crate::parser::Declaration::GadtDef(_) => None,
        })
        .collect();

    for binding in bindings.iter() {
        let env = TypeEnv::End;
        let _ = binding.value.get_type(&env);
    }

    let test_cases = parse_tree
        .declarations
        .iter()
        .filter_map(|x| match x {
            crate::parser::Declaration::TestCaseDef(t) => Some(make_expression(
                &mut AstConversionState {
                    next_lit_id: LitId(None),
                    total: state.total,
                },
                &funcs,
                &t,
                false,
            )),
            crate::parser::Declaration::BindingDef(_) => None,
            crate::parser::Declaration::GadtDef(_) => None,
        })
        .collect();

    let gadts: Vec<GadtDefinition<IntType, FloatType, BoolType>> = parse_tree
        .declarations
        .iter()
        .filter_map(|x| match x {
            crate::parser::Declaration::TestCaseDef(_) => None,
            crate::parser::Declaration::BindingDef(_) => None,
            crate::parser::Declaration::GadtDef(gadt_def) => Some(make_gadt(
                &mut AstConversionState {
                    next_lit_id: LitId(None),
                    total: state.total,
                },
                &funcs,
                gadt_def,
                true,
            )),
        })
        .collect();

    let constructors: Vec<Binding<IntType, FloatType, BoolType>> = gadts
        .iter()
        .flat_map(|gadt| {
            gadt.constructors.iter().map(|x| Binding {
                name: x.name.clone(),
                elem_type: x.arg_type.clone(),
                value: Expression::Intrinsic,
            })
        })
        .collect();

    Program {
        global_bindings: bindings.into_iter().chain(constructors).collect(),
        test_cases,
        gadts,
        num_ids: state.next_lit_id.0.unwrap(),
    }
}

fn make_definition<IntType, FloatType, BoolType>(
    state: &mut AstConversionState,
    funcs: &ProgramInitFunctions<IntType, FloatType, BoolType>,
    value: &crate::parser::BindingDefinition,
) -> Binding<IntType, FloatType, BoolType> {
    Binding {
        name: Identifier(value.name.clone()),
        elem_type: make_expression(state, funcs, &value.binding_type, true),
        value: make_expression(state, funcs, &value.func_body, true),
    }
}

fn make_expression<IntType, FloatType, BoolType>(
    state: &mut AstConversionState,
    funcs: &ProgramInitFunctions<IntType, FloatType, BoolType>,
    value: &crate::parser::Expression,
    differentiable: bool,
) -> Expression<IntType, FloatType, BoolType> {
    match value {
        crate::parser::Expression::Variable(ref x) => {
            if !differentiable {
                panic!()
            }
            Expression::Variable {
                ident: Identifier(x.to_string()),
            }
        }
        crate::parser::Expression::IntegerLit(x) => {
            let out = Expression::Integer(
                (funcs.make_int)(*x, state.next_lit_id, state.total),
                state.next_lit_id,
            );
            if let Some(x) = state.next_lit_id.0.as_mut() {
                *x += 1
            };
            out
        }
        crate::parser::Expression::FloatLit(x) => {
            let output = Expression::Float(
                (funcs.make_float)(*x, state.next_lit_id, state.total),
                state.next_lit_id,
            );
            if let Some(x) = state.next_lit_id.0.as_mut() {
                *x += 1
            };
            output
        }
        crate::parser::Expression::BoolLit(x) => {
            let output = Expression::Bool(
                (funcs.make_bool)(*x, state.next_lit_id, state.total),
                state.next_lit_id,
            );
            if let Some(x) = state.next_lit_id.0.as_mut() {
                *x += 1
            };
            output
        }
        crate::parser::Expression::FunctionApplicationMultipleArgs { ref func, ref args } => {
            Expression::FuncApplicationMultipleArgs {
                func: Box::new(make_expression(state, funcs, &func, true)),
                args: args
                    .clone()
                    .into_iter()
                    .map(|x| make_expression(state, funcs, &x, true))
                    .collect(),
            }
        }
        crate::parser::Expression::LetBinds { bindings, inner } => {
            let bindings = bindings
                .clone()
                .into_iter()
                .map(|x| LetBind {
                    ident: Identifier(x.name.to_string()),
                    value: make_expression(state, funcs, &x.value, differentiable),
                })
                .collect();

            Expression::ExprLetBinding {
                bindings,
                inner: Box::new(make_expression(state, funcs, &inner, differentiable)),
            }
        }
        crate::parser::Expression::DependentProductType {
            type_first,
            type_second,
        } => Expression::DependentProductType {
            type_first: Box::new(make_argument(state, funcs, type_first, differentiable)),
            type_second: Box::new(make_expression(state, funcs, type_second, differentiable)),
        },
        crate::parser::Expression::DependentFunctionType { type_from, type_to } => {
            Expression::DependentFunctionType {
                type_from: Box::new(make_argument(state, funcs, type_from, differentiable)),
                type_to: Box::new(make_expression(state, funcs, type_to, differentiable)),
            }
        }
        crate::parser::Expression::UniverseLit(x) => Expression::Universe(*x),
        crate::parser::Expression::Lambda { input, body } => Expression::Lambda {
            input: Box::new(make_argument(state, funcs, input, differentiable)),
            body: Box::new(make_expression(state, funcs, body, differentiable)),
        },
    }
}

fn make_argument<IntType, FloatType, BoolType>(
    state: &mut AstConversionState,
    funcs: &ProgramInitFunctions<IntType, FloatType, BoolType>,
    value: &crate::parser::Argument,
    differentiable: bool,
) -> Argument<IntType, FloatType, BoolType> {
    Argument {
        name: Identifier(value.varname.clone()),
        arg_type: make_expression(state, funcs, &value.vartype, differentiable),
    }
}

fn make_gadt<IntType, FloatType, BoolType>(
    state: &mut AstConversionState,
    funcs: &ProgramInitFunctions<IntType, FloatType, BoolType>,
    value: &crate::parser::GadtDefinition,
    differentiable: bool,
) -> GadtDefinition<IntType, FloatType, BoolType> {
    GadtDefinition {
        name: Identifier(value.name.clone()),
        gadt_type: make_expression(state, funcs, &value.gadt_type, differentiable),
        constructors: value
            .constructors
            .iter()
            .map(|x: &crate::parser::Argument| Argument {
                name: Identifier(x.varname.to_string()),
                arg_type: make_expression(state, funcs, &x.vartype, differentiable),
            })
            .collect(),
    }
}

fn find_function_application_type<IntType, FloatType, BoolType>(
    func_type: &Expression<IntType, FloatType, BoolType>,
    args: &[Expression<IntType, FloatType, BoolType>],
    env: &TypeEnv<IntType, FloatType, BoolType>,
) -> Expression<IntType, FloatType, BoolType>
where
    BoolType: Clone + std::fmt::Debug,
    FloatType: Clone + std::fmt::Debug,
    IntType: Clone + std::fmt::Debug,
{
    match args.split_first() {
        Some((first_arg, rest_args)) => {
            match func_type {
                Expression::DependentFunctionType { type_from, type_to } => {
                    assert!(types_are_equal(
                        &type_from.arg_type,
                        &first_arg.get_type(env),
                        env,
                    ));

                    find_function_application_type(&type_to, rest_args, env)
                }
                x => panic!("invalid function application: {:?}", x),
            }
        }
        None => func_type.clone(),
    }
}

fn types_are_equal<IntType, FloatType, BoolType>(
    a: &Expression<IntType, FloatType, BoolType>,
    b: &Expression<IntType, FloatType, BoolType>,
    env: &TypeEnv<IntType, FloatType, BoolType>,
) -> bool
where
    BoolType: Clone + std::fmt::Debug,
    FloatType: Clone + std::fmt::Debug,
    IntType: Clone + std::fmt::Debug,
{
    match (a, b) {
        (
            Expression::DependentFunctionType {
                type_from: a1,
                type_to: a2,
            },
            Expression::DependentFunctionType {
                type_from: b1,
                type_to: b2,
            },
        ) => types_are_equal(&a1.arg_type, &b1.arg_type, env) && types_are_equal(a2, b2, env),
        (
            Expression::DependentProductType {
                type_first: a1,
                type_second: a2,
            },
            Expression::DependentProductType {
                type_first: b1,
                type_second: b2,
            },
        ) => types_are_equal(&a1.arg_type, &b1.arg_type, env) && types_are_equal(a2, b2, env),
        (Expression::Universe(ua), Expression::Universe(ub)) => ua == ub,
        (Expression::Variable { ident: _ }, Expression::Variable { ident: _ }) => {
            types_are_equal(&a.get_type(env), &b.get_type(env), env)
        }
        (x, y) => panic!("unequal types {:?} vs {:?}", x, y),
    }
}
