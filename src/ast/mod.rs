pub mod eval;
#[derive(serde::Serialize, Clone, Debug, PartialEq)]
pub struct Identifier(pub String);

#[derive(serde::Serialize, Clone, Debug)]
pub struct TypeName(pub String);

#[derive(serde::Serialize, Clone)]
pub struct Program<IntType, FloatType, BoolType> {
    pub functions: Vec<FunctionDefinition<IntType, FloatType, BoolType>>,
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

impl<IntType, FloatType, BoolType> Program<IntType, FloatType, BoolType> {
    pub fn find_func(&self, name: &str) -> &FunctionDefinition<IntType, FloatType, BoolType> {
        for func in self.functions.iter() {
            if func.name.0 == name {
                return func;
            }
        }
        eprintln!("coulnd't find func {}", name);
        unreachable!()
    }
}

struct AstConversionState {
    next_lit_id: LitId,
    total: usize,
}

#[derive(serde::Serialize, Clone)]
pub struct GadtDefinition<IntType, FloatType, BoolType> {
    pub name: Identifier,
    pub universe: u64,
    pub arguments: Vec<(Identifier, Expression<IntType, FloatType, BoolType>)>,
    pub constructors: Vec<(Identifier, Expression<IntType, FloatType, BoolType>)>,
}

#[derive(serde::Serialize, Clone)]
pub struct FunctionDefinition<IntType, FloatType, BoolType> {
    pub name: Identifier,
    pub universe: u64,
    pub arguments: Vec<(Identifier, Expression<IntType, FloatType, BoolType>)>,
    pub to_type: Expression<IntType, FloatType, BoolType>,
    pub body: Expression<IntType, FloatType, BoolType>,
}

#[derive(serde::Serialize, Copy, Clone)]
pub struct LitId(pub Option<usize>);

#[derive(serde::Serialize, Clone)]
pub enum Expression<IntType, FloatType, BoolType> {
    Variable {
        ident: Identifier,
    },
    Product(Vec<Expression<IntType, FloatType, BoolType>>),

    Integer(IntType, LitId),
    Float(FloatType, LitId),
    Bool(BoolType, LitId),
    Universe(u64),
    FuncApplication {
        func_name: Identifier,
        args: Vec<Expression<IntType, FloatType, BoolType>>,
    },
    ExprWhere {
        bindings: Vec<LetBind<IntType, FloatType, BoolType>>,
        inner: Box<Expression<IntType, FloatType, BoolType>>,
    },
    IfThenElse {
        boolean: Box<Expression<IntType, FloatType, BoolType>>,
        true_expr: Box<Expression<IntType, FloatType, BoolType>>,
        false_expr: Box<Expression<IntType, FloatType, BoolType>>,
    },
}

#[derive(serde::Serialize, Clone)]
pub struct LetBind<IntType, FloatType, BoolType> {
    pub ident: Identifier,
    pub value: Expression<IntType, FloatType, BoolType>,
}

pub fn make_program<IntType, FloatType, BoolType>(
    parse_tree: crate::parser::ProgramParseTree,
    funcs: ProgramInitFunctions<IntType, FloatType, BoolType>,
) -> Program<IntType, FloatType, BoolType> {
    let state = AstConversionState {
        next_lit_id: LitId(Some(0)),
        total: 0,
    };

    let counter = make_program_inner(parse_tree.clone(), NULL_AST_INIT, state);

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
    parse_tree: crate::parser::ProgramParseTree,
    funcs: ProgramInitFunctions<IntType, FloatType, BoolType>,
    mut state: AstConversionState,
) -> Program<IntType, FloatType, BoolType> {
    let functions = parse_tree
        .declarations
        .iter()
        .filter_map(|x| match x {
            crate::parser::Declaration::FunctionDef(f) => {
                Some(make_function_definition(&mut state, &funcs, f))
            }
            crate::parser::Declaration::TestCaseDef(_) => None,
            crate::parser::Declaration::GadtDef(_) => None,
        })
        .collect();

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
                t.clone(),
                false,
            )),
            crate::parser::Declaration::FunctionDef(_) => None,
            crate::parser::Declaration::GadtDef(_) => None,
        })
        .collect();

    let gadts = parse_tree
        .declarations
        .iter()
        .filter_map(|x| match x {
            crate::parser::Declaration::TestCaseDef(_) => None,
            crate::parser::Declaration::FunctionDef(_) => None,
            crate::parser::Declaration::GadtDef(gadt_def) => Some(make_gadt(
                &mut AstConversionState {
                    next_lit_id: LitId(None),
                    total: state.total,
                },
                &funcs,
                gadt_def.clone(),
                true,
            )),
        })
        .collect();

    Program {
        functions,
        test_cases,
        gadts,
        num_ids: state.next_lit_id.0.unwrap(),
    }
}

fn make_function_definition<IntType, FloatType, BoolType>(
    state: &mut AstConversionState,
    funcs: &ProgramInitFunctions<IntType, FloatType, BoolType>,
    value: &crate::parser::FunctionDefinition,
) -> FunctionDefinition<IntType, FloatType, BoolType> {
    FunctionDefinition {
        name: Identifier(value.name.clone()),
        universe: value.universe,
        to_type: make_expression(state, funcs, value.to_type.clone(), true),
        arguments: value
            .clone()
            .arguments
            .into_iter()
            .map(|x: crate::parser::Argument| {
                (
                    Identifier(x.varname.to_string()),
                    make_expression(state, funcs, x.vartype, true),
                )
            })
            .collect(),
        body: make_expression(state, funcs, value.func_body.clone(), true),
    }
}

fn make_expression<IntType, FloatType, BoolType>(
    state: &mut AstConversionState,
    funcs: &ProgramInitFunctions<IntType, FloatType, BoolType>,
    value: crate::parser::Expression,
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
                (funcs.make_int)(x, state.next_lit_id, state.total),
                state.next_lit_id,
            );
            if let Some(x) = state.next_lit_id.0.as_mut() {
                *x += 1
            };
            out
        }
        crate::parser::Expression::FloatLit(x) => {
            let output = Expression::Float(
                (funcs.make_float)(x, state.next_lit_id, state.total),
                state.next_lit_id,
            );
            if let Some(x) = state.next_lit_id.0.as_mut() {
                *x += 1
            };
            output
        }
        crate::parser::Expression::BoolLit(x) => {
            let output = Expression::Bool(
                (funcs.make_bool)(x, state.next_lit_id, state.total),
                state.next_lit_id,
            );
            if let Some(x) = state.next_lit_id.0.as_mut() {
                *x += 1
            };
            output
        }
        crate::parser::Expression::FunctionApplication {
            ref func_name,
            ref args,
        } => {
            let func_name = func_name.to_string();
            let can_diff = func_name == "__len" || differentiable;

            Expression::FuncApplication {
                func_name: Identifier(func_name),
                args: args
                    .clone()
                    .into_iter()
                    .map(|x| make_expression(state, funcs, x, can_diff))
                    .collect(),
            }
        }
        crate::parser::Expression::LetBinds { bindings, inner } => {
            let bindings = bindings
                .clone()
                .into_iter()
                .map(|x| LetBind {
                    ident: Identifier(x.name.to_string()),
                    value: make_expression(state, funcs, *x.value, differentiable),
                })
                .collect();

            Expression::ExprWhere {
                bindings,
                inner: Box::new(make_expression(state, funcs, *inner, differentiable)),
            }
        }
        crate::parser::Expression::IfThenElse {
            boolean,
            true_expr,
            false_expr,
        } => Expression::IfThenElse {
            boolean: Box::new(make_expression(state, funcs, *boolean, differentiable)),
            true_expr: Box::new(make_expression(state, funcs, *true_expr, differentiable)),
            false_expr: Box::new(make_expression(state, funcs, *false_expr, differentiable)),
        },
        crate::parser::Expression::Product(x) => Expression::Product(
            x.values
                .into_iter()
                .map(|x| make_expression(state, funcs, x, differentiable))
                .collect(),
        ),
        crate::parser::Expression::UniverseLit(x) => Expression::Universe(x),
    }
}

fn make_gadt<IntType, FloatType, BoolType>(
    state: &mut AstConversionState,
    funcs: &ProgramInitFunctions<IntType, FloatType, BoolType>,
    value: crate::parser::GadtDefinition,
    differentiable: bool,
) -> GadtDefinition<IntType, FloatType, BoolType> {
    GadtDefinition {
        name: Identifier(value.name),
        universe: value.universe,
        arguments: value
            .arguments
            .into_iter()
            .map(|x: crate::parser::Argument| {
                (
                    Identifier(x.varname.to_string()),
                    make_expression(state, funcs, x.vartype, differentiable),
                )
            })
            .collect(),
        constructors: value
            .constructors
            .into_iter()
            .map(|x: crate::parser::Argument| {
                (
                    Identifier(x.varname.to_string()),
                    make_expression(state, funcs, x.vartype, differentiable),
                )
            })
            .collect(),
    }
}
