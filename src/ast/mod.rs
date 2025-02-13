pub mod eval;
#[derive(serde::Serialize, Clone, Debug, PartialEq)]
pub struct Identifier(pub String);

#[derive(serde::Serialize, Clone, Debug)]
pub struct TypeName(pub String);

#[derive(serde::Serialize, Clone)]
pub struct Program<IntType, FloatType, BoolType, HardType> {
    pub functions: Vec<FunctionDefinition<IntType, FloatType, BoolType, HardType>>,
    pub test_cases: Vec<Expression<IntType, FloatType, BoolType, HardType>>,
    pub num_ids: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct ProgramInitFunctions<IntType, FloatType, BoolType, HardType> {
    pub make_int: fn(i64, LitId, usize) -> IntType,
    pub make_float: fn(f64, LitId, usize) -> FloatType,
    pub make_bool: fn(bool, LitId, usize) -> BoolType,
    pub make_hard: fn(i64) -> HardType,
}

impl<IntType, FloatType, BoolType, HardType> Program<IntType, FloatType, BoolType, HardType> {
    pub fn find_func(
        &self,
        name: &str,
    ) -> &FunctionDefinition<IntType, FloatType, BoolType, HardType> {
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
pub struct FunctionDefinition<IntType, FloatType, BoolType, HardType> {
    pub name: Identifier,
    pub arguments: Vec<(Identifier, TypeName)>,
    pub to_type: TypeName,
    pub body: Expression<IntType, FloatType, BoolType, HardType>,
}

#[derive(serde::Serialize, Copy, Clone)]
pub struct LitId(pub Option<usize>);

#[derive(serde::Serialize, Clone)]
pub enum Expression<IntType, FloatType, BoolType, HardType> {
    Variable {
        ident: Identifier,
    },
    Product(Vec<Expression<IntType, FloatType, BoolType, HardType>>),
    ProductProject {
        value: Box<Expression<IntType, FloatType, BoolType, HardType>>,
        index: HardType,
    },

    HardInt(HardType),
    Integer(IntType, LitId),
    Float(FloatType, LitId),
    Str(String, LitId),
    Bool(BoolType, LitId),
    List {
        type_name: TypeName,
        values: Vec<Expression<IntType, FloatType, BoolType, HardType>>,
    },
    FuncApplication {
        func_name: Identifier,
        args: Vec<Expression<IntType, FloatType, BoolType, HardType>>,
    },
    ExprWhere {
        bindings: Vec<LetBind<IntType, FloatType, BoolType, HardType>>,
        inner: Box<Expression<IntType, FloatType, BoolType, HardType>>,
    },
    IfThenElse {
        boolean: Box<Expression<IntType, FloatType, BoolType, HardType>>,
        true_expr: Box<Expression<IntType, FloatType, BoolType, HardType>>,
        false_expr: Box<Expression<IntType, FloatType, BoolType, HardType>>,
    },
    FoldLoop {
        fold_iter: Box<FoldIter<IntType, FloatType, BoolType, HardType>>,
        accumulator: (
            Identifier,
            Box<Expression<IntType, FloatType, BoolType, HardType>>,
        ),
        body: Box<Expression<IntType, FloatType, BoolType, HardType>>,
    },
    WhileLoop {
        accumulator: (
            Identifier,
            Box<Expression<IntType, FloatType, BoolType, HardType>>,
        ),
        cond: Box<Expression<IntType, FloatType, BoolType, HardType>>,
        body: Box<Expression<IntType, FloatType, BoolType, HardType>>,
        exit_body: Box<Expression<IntType, FloatType, BoolType, HardType>>,
    },
}

#[derive(serde::Serialize, Clone)]
pub enum FoldIter<IntType, FloatType, BoolType, HardType> {
    ExprList(Expression<IntType, FloatType, BoolType, HardType>),
    Range(
        Expression<IntType, FloatType, BoolType, HardType>,
        Expression<IntType, FloatType, BoolType, HardType>,
    ),
}

#[derive(serde::Serialize, Clone)]
pub struct LetBind<IntType, FloatType, BoolType, HardType> {
    pub ident: Identifier,
    pub value: Expression<IntType, FloatType, BoolType, HardType>,
}

pub fn make_program<IntType, FloatType, BoolType, HardType>(
    parse_tree: crate::parser::ProgramParseTree,
    funcs: ProgramInitFunctions<IntType, FloatType, BoolType, HardType>,
) -> Program<IntType, FloatType, BoolType, HardType> {
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

pub const NULL_AST_INIT: ProgramInitFunctions<i64, f64, bool, i64> = ProgramInitFunctions {
    make_int,
    make_float,
    make_bool,
    make_hard,
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
fn make_hard(x: i64) -> i64 {
    x
}

fn make_program_inner<IntType, FloatType, BoolType, HardType>(
    parse_tree: crate::parser::ProgramParseTree,
    funcs: ProgramInitFunctions<IntType, FloatType, BoolType, HardType>,
    mut state: AstConversionState,
) -> Program<IntType, FloatType, BoolType, HardType> {
    let functions = parse_tree
        .declarations
        .iter()
        .filter_map(|x| match x {
            crate::parser::Declaration::FunctionDef(f) => {
                Some(make_function_definition(&mut state, &funcs, f))
            }
            crate::parser::Declaration::TestCaseDef(_) => None,
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
                true,
            )),
            crate::parser::Declaration::FunctionDef(_) => None,
        })
        .collect();
    Program {
        functions,
        test_cases,
        num_ids: state.next_lit_id.0.unwrap(),
    }
}

fn make_function_definition<IntType, FloatType, BoolType, HardType>(
    state: &mut AstConversionState,
    funcs: &ProgramInitFunctions<IntType, FloatType, BoolType, HardType>,
    value: &crate::parser::FunctionDefinition,
) -> FunctionDefinition<IntType, FloatType, BoolType, HardType> {
    FunctionDefinition {
        name: Identifier(value.name.clone()),
        to_type: TypeName(value.to_type.0.to_string()),
        arguments: value.clone()
            .arguments
            .into_iter()
            .map(|x: crate::parser::Argument| {
                (
                    Identifier(x.varname.to_string()),
                    TypeName(x.vartype.0.to_string()),
                )
            })
            .collect(),
        body: make_expression(
            state,
            funcs,
            value.func_body.clone(),
            true,
        ),
    }
}

fn make_expression<IntType, FloatType, BoolType, HardType>(
    state: &mut AstConversionState,
    funcs: &ProgramInitFunctions<IntType, FloatType, BoolType, HardType>,
    value: crate::parser::Expression,
    differentiable: bool,
) -> Expression<IntType, FloatType, BoolType, HardType> {
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
            let output = if differentiable {
                let out = Expression::Integer(
                    (funcs.make_int)(x, state.next_lit_id, state.total),
                    state.next_lit_id,
                );
                if let Some(x) = state.next_lit_id.0.as_mut() {
                    *x += 1
                };
                out
            } else {
                Expression::HardInt((funcs.make_hard)(x))
            };

            output
        }
        crate::parser::Expression::StringLit(x) => {
            let output = Expression::Str(x, state.next_lit_id);
            if let Some(x) = state.next_lit_id.0.as_mut() {
                *x += 1
            };
            output
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
        crate::parser::Expression::Addition(ref x) => Expression::FuncApplication {
            func_name: Identifier("__add".to_owned()),
            args: vec![
                make_expression(state, funcs, *x.left_side.clone(), differentiable),
                make_expression(state, funcs, *x.right_side.clone(), differentiable),
            ],
        },
        crate::parser::Expression::Subtraction(ref x) => Expression::FuncApplication {
            func_name: Identifier("__sub".to_owned()),
            args: vec![
                make_expression(state, funcs, *x.left_side.clone(), differentiable),
                make_expression(state, funcs, *x.right_side.clone(), differentiable),
            ],
        },
        crate::parser::Expression::Multiplication(ref x) => Expression::FuncApplication {
            func_name: Identifier("__mul".to_owned()),
            args: vec![
                make_expression(state, funcs, *x.left_side.clone(), differentiable),
                make_expression(state, funcs, *x.right_side.clone(), differentiable),
            ],
        },
        crate::parser::Expression::Division(ref x) => Expression::FuncApplication {
            func_name: Identifier("__div".to_owned()),
            args: vec![
                make_expression(state, funcs, *x.left_side.clone(), differentiable),
                make_expression(state, funcs, *x.right_side.clone(), differentiable),
            ],
        },
        crate::parser::Expression::Equality(ref x) => Expression::FuncApplication {
            func_name: Identifier("__eq".to_owned()),
            args: vec![
                make_expression(state, funcs, *x.left_side.clone(), differentiable),
                make_expression(state, funcs, *x.right_side.clone(), differentiable),
            ],
        },
        crate::parser::Expression::GreaterThan(ref x) => Expression::FuncApplication {
            func_name: Identifier("__gt".to_owned()),
            args: vec![
                make_expression(state, funcs, *x.left_side.clone(), differentiable),
                make_expression(state, funcs, *x.right_side.clone(), differentiable),
            ],
        },
        crate::parser::Expression::LessThan(ref x) => Expression::FuncApplication {
            func_name: Identifier("__lt".to_owned()),
            args: vec![
                make_expression(state, funcs, *x.left_side.clone(), differentiable),
                make_expression(state, funcs, *x.right_side.clone(), differentiable),
            ],
        },
        crate::parser::Expression::And(ref x) => Expression::FuncApplication {
            func_name: Identifier("__and".to_owned()),
            args: vec![
                make_expression(state, funcs, *x.left_side.clone(), differentiable),
                make_expression(state, funcs, *x.right_side.clone(), differentiable),
            ],
        },
        crate::parser::Expression::Or(ref x) => Expression::FuncApplication {
            func_name: Identifier("__or".to_owned()),
            args: vec![
                make_expression(state, funcs, *x.left_side.clone(), differentiable),
                make_expression(state, funcs, *x.right_side.clone(), differentiable),
            ],
        },
        crate::parser::Expression::Not(inner)

     => Expression::FuncApplication {
            func_name: Identifier("__not".to_owned()),
            args: vec![make_expression(
                state,
                funcs,
                *inner.clone(),
                differentiable,
            )],
        },
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
        crate::parser::Expression::ExprWhere {
            bindings,
            inner,
        } => {
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
        crate::parser::Expression::FoldLoop {
            accumulator,
            body,
            iter_val,
        } => {
            
            let fold_iter = match iter_val {
                crate::parser::FoldIter::Range(range) => {
                    FoldIter::Range(
                        make_expression(state, funcs, *range.start, false),
                        make_expression(state, funcs, *range.end, false),
                    )
                }
                crate::parser::FoldIter::ListExpr(expr) => {
                    FoldIter::ExprList(make_expression(state, funcs, *expr, differentiable))
                }
            };
            Expression::FoldLoop {
                accumulator: (
                    Identifier(accumulator.name.to_string()),
                    Box::new(make_expression(
                        state,
                        funcs,
                        *accumulator.initial_expression,
                        differentiable,
                    )),
                ),
                body: Box::new(make_expression(state, funcs, *body, differentiable)),
                fold_iter: Box::new(fold_iter),
            }
        }
        crate::parser::Expression::WhileFoldLoop {
            accumulator,
            cond,
            body,
            exit_body,
        } => {
            Expression::WhileLoop {
                accumulator: (
                    Identifier(accumulator.name.to_string()),
                    Box::new(make_expression(
                        state,
                        funcs,
                        *accumulator.initial_expression,
                        differentiable,
                    )),
                ),
                cond: Box::new(make_expression(state, funcs, *cond, differentiable)),
                body: Box::new(make_expression(state, funcs, *body, differentiable)),
                exit_body: Box::new(make_expression(state, funcs, *exit_body, differentiable)),
            }
        }
        crate::parser::Expression::ListLit(list_inner) => {
            Expression::List {
                type_name: TypeName(list_inner.type_name.to_string()),
                values: list_inner
                    .values
                    .into_iter()
                    .map(|value| make_expression(state, funcs, value, differentiable))
                    .collect(),
            }
        }
        crate::parser::Expression::Product(x) => Expression::Product(
            x.values
                .into_iter()
                .map(|x| make_expression(state, funcs, x, differentiable))
                .collect(),
        ),
        crate::parser::Expression::ProductProject {
            index,
            value,
        } => Expression::ProductProject {
            value: Box::new(make_expression(state, funcs, *value, differentiable)),
            index: (funcs.make_hard)(index),
        },
    }
}
