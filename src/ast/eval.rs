#[derive(Clone, Debug)]
pub enum EvalVal<IntType, FloatType, BoolType, HardType> {
    Int(IntType),
    Float(FloatType),
    Bool(BoolType),
    Hard(HardType),
    List(Vec<EvalVal<IntType, FloatType, BoolType, HardType>>),
    Product(Vec<EvalVal<IntType, FloatType, BoolType, HardType>>),
}

#[derive(Clone)]
struct Env<'a, IntType, FloatType, BoolType, HardType> {
    program: &'a super::Program<IntType, FloatType, BoolType, HardType>,
    vars: EnvVars<IntType, FloatType, BoolType, HardType>,
}

#[derive(Clone)]
enum EnvVars<IntType, FloatType, BoolType, HardType> {
    End,
    Rest {
        first: (
            super::Identifier,
            EvalVal<IntType, FloatType, BoolType, HardType>,
        ),
        rest: Box<EnvVars<IntType, FloatType, BoolType, HardType>>,
    },
}

pub trait Evaluator<IntType, FloatType, BoolType, HardType> {
    fn eval_addition_ints(&self, a: IntType, b: IntType) -> IntType;
    fn eval_addition_hards(&self, a: HardType, b: HardType) -> HardType;
    fn eval_addition_floats(&self, a: FloatType, b: FloatType) -> FloatType;
    fn eval_multiplication_int(&self, a: IntType, b: IntType) -> IntType;
    fn eval_multiplication_floats(&self, a: FloatType, b: FloatType) -> FloatType;
    fn eval_negation_int(&self, a: IntType) -> IntType;
    fn eval_negation_hard(&self, a: HardType) -> HardType;
    fn eval_negation_float(&self, a: FloatType) -> FloatType;
    fn eval_equality_ints(&self, a: IntType, b: IntType) -> BoolType;
    fn eval_equality_floats(&self, a: FloatType, b: FloatType) -> BoolType;
    fn eval_less_than_ints(&self, a: IntType, b: IntType) -> BoolType;
    fn eval_less_than_floats(&self, a: FloatType, b: FloatType) -> BoolType;
    fn eval_not(&self, a: BoolType) -> BoolType;

    fn eval_and(&self, a: BoolType, b: BoolType) -> BoolType;
    fn eval_or(&self, a: BoolType, b: BoolType) -> BoolType;
    fn eval_index(
        &self,
        l: Vec<EvalVal<IntType, FloatType, BoolType, HardType>>,
        i: IntType,
    ) -> EvalVal<IntType, FloatType, BoolType, HardType>;
    fn eval_set_index(
        &self,
        l: Vec<EvalVal<IntType, FloatType, BoolType, HardType>>,
        i: IntType,
        v: EvalVal<IntType, FloatType, BoolType, HardType>,
    ) -> Vec<EvalVal<IntType, FloatType, BoolType, HardType>>;
    fn eval_len(&self, l: Vec<EvalVal<IntType, FloatType, BoolType, HardType>>) -> HardType;
    fn eval_if(
        &self,
        cond: BoolType,
        true_branch: EvalVal<IntType, FloatType, BoolType, HardType>,
        false_branch: EvalVal<IntType, FloatType, BoolType, HardType>,
    ) -> EvalVal<IntType, FloatType, BoolType, HardType>;
    fn stop_while_eval(&self, cond: BoolType) -> bool;
    fn make_range(&self, start: HardType, end: HardType, num_ids: usize) -> Vec<IntType>;
    fn eval_product_index(
        &self,
        p: Vec<EvalVal<IntType, FloatType, BoolType, HardType>>,
        i: HardType,
    ) -> EvalVal<IntType, FloatType, BoolType, HardType>;
}

pub fn run_function<
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    HardType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    program: &super::Program<IntType, FloatType, BoolType, HardType>,
    func_name: &str,
    arguments: Vec<EvalVal<IntType, FloatType, BoolType, HardType>>,
) -> EvalVal<IntType, FloatType, BoolType, HardType>
where
    E: Evaluator<IntType, FloatType, BoolType, HardType>,
{
    apply_function::<IntType, FloatType, BoolType, HardType, E>(
        evaluator,
        Env {
            program,
            vars: EnvVars::End,
        },
        func_name,
        arguments,
    )
}

impl<
        IntType: Clone + core::fmt::Debug,
        FloatType: Clone + core::fmt::Debug,
        BoolType: Clone + core::fmt::Debug,
        HardType: Clone + core::fmt::Debug,
    > EnvVars<IntType, FloatType, BoolType, HardType>
{
    fn lookup_var(
        &self,
        var_name: &super::Identifier,
    ) -> EvalVal<IntType, FloatType, BoolType, HardType> {
        match self {
            EnvVars::End => unreachable!("couldn't find var {:?}", var_name),
            EnvVars::Rest { first, rest } => {
                if first.0 == *var_name {
                    first.1.clone()
                } else {
                    rest.lookup_var(var_name)
                }
            }
        }
    }
}

fn eval<
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    HardType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    env: Env<IntType, FloatType, BoolType, HardType>,
    expr: &super::Expression<IntType, FloatType, BoolType, HardType>,
) -> EvalVal<IntType, FloatType, BoolType, HardType>
where
    E: Evaluator<IntType, FloatType, BoolType, HardType>,
{
    match expr {
        super::Expression::Variable { ident } => env.vars.lookup_var(ident),
        super::Expression::Integer(x, _) => EvalVal::Int(x.clone()),
        super::Expression::HardInt(x) => EvalVal::Hard(x.clone()),
        super::Expression::Str(_, _) => todo!(),
        super::Expression::Float(x, _) => EvalVal::Float(x.clone()),
        super::Expression::Bool(x, _) => EvalVal::Bool(x.clone()),
        super::Expression::FuncApplication {
            func_name,
            args,
        } => match func_name.0.as_str() {
            "__add" | "__sub" | "__mul" | "__div" | "__eq" | "__lt" | "__gt" | "__and" | "__or" => {
                match (
                    eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator,
                        env.clone(),
                        &args[0],
                    ),
                    eval::<IntType, FloatType, BoolType, HardType, E>(evaluator, env, &args[1]),
                ) {
                    (EvalVal::Int(a), EvalVal::Int(b)) => match func_name.0.as_str() {
                        "__add" => EvalVal::Int(evaluator.eval_addition_ints(a, b)),
                        "__sub" => EvalVal::Int(
                            evaluator.eval_addition_ints(a, evaluator.eval_negation_int(b)),
                        ),
                        "__mul" => EvalVal::Int(evaluator.eval_multiplication_int(a, b)),
                        "__div" => todo!(),
                        "__eq" => EvalVal::Bool(evaluator.eval_equality_ints(a, b)),
                        "__lt" => EvalVal::Bool(evaluator.eval_less_than_ints(a, b)),
                        "__gt" => EvalVal::Bool(evaluator.eval_less_than_ints(b, a)),
                        _ => todo!(),
                    },
                    (EvalVal::Float(a), EvalVal::Float(b)) => match func_name.0.as_str() {
                        "__add" => EvalVal::Float(evaluator.eval_addition_floats(a, b)),
                        "__sub" => EvalVal::Float(
                            evaluator.eval_addition_floats(a, evaluator.eval_negation_float(b)),
                        ),
                        "__mul" => EvalVal::Float(evaluator.eval_multiplication_floats(a, b)),
                        "__div" => todo!(),
                        "__eq" => EvalVal::Bool(evaluator.eval_equality_floats(a, b)),
                        "__lt" => EvalVal::Bool(evaluator.eval_less_than_floats(a, b)),
                        "__gt" => EvalVal::Bool(evaluator.eval_less_than_floats(b, a)),
                        _ => todo!(),
                    },
                    (EvalVal::Hard(a), EvalVal::Hard(b)) => match func_name.0.as_str() {
                        "__add" => EvalVal::Hard(evaluator.eval_addition_hards(a, b)),
                        "__sub" => EvalVal::Hard(
                            evaluator.eval_addition_hards(a, evaluator.eval_negation_hard(b)),
                        ),
                        _ => todo!(),
                    },

                    (EvalVal::Bool(a), EvalVal::Bool(b)) => match func_name.0.as_str() {
                        "__and" => EvalVal::Bool(evaluator.eval_and(a, b)),
                        "__or" => EvalVal::Bool(evaluator.eval_or(a, b)),
                        _ => todo!(),
                    },
                    (x, y) => todo!("THING = {}({:?}, {:?})", func_name.0.as_str(), x, y),
                }
            }
            "__neg" => {
                match eval::<IntType, FloatType, BoolType, HardType, E>(
                    evaluator,
                    env.clone(),
                    &args[0],
                ) {
                    EvalVal::Int(x) => EvalVal::Int(evaluator.eval_negation_int(x)),
                    EvalVal::Float(x) => EvalVal::Float(evaluator.eval_negation_float(x)),
                    _ => todo!(),
                }
            }
            "__not" => {
                match eval::<IntType, FloatType, BoolType, HardType, E>(
                    evaluator,
                    env.clone(),
                    &args[0],
                ) {
                    EvalVal::Bool(x) => EvalVal::Bool(evaluator.eval_not(x)),
                    _ => todo!(),
                }
            }

            "__index" => {
                match (
                    eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator,
                        env.clone(),
                        &args[0],
                    ),
                    eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator,
                        env.clone(),
                        &args[1],
                    ),
                ) {
                    (EvalVal::List(l), EvalVal::Int(i)) => evaluator.eval_index(l, i),
                    _ => todo!(),
                }
            }
            "__set_index" => {
                match (
                    eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator,
                        env.clone(),
                        &args[0],
                    ),
                    eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator,
                        env.clone(),
                        &args[1],
                    ),
                    eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator,
                        env.clone(),
                        &args[2],
                    ),
                ) {
                    (EvalVal::List(l), EvalVal::Int(i), v) => {
                        EvalVal::List(evaluator.eval_set_index(l, i, v))
                    }
                    _ => todo!(),
                }
            }
            "__len" => {
                match eval::<IntType, FloatType, BoolType, HardType, E>(
                    evaluator,
                    env.clone(),
                    &args[0],
                ) {
                    EvalVal::List(l) => EvalVal::Hard(evaluator.eval_len(l)),
                    _ => todo!(),
                }
            }

            func_name => apply_function::<IntType, FloatType, BoolType, HardType, E>(
                evaluator,
                env.clone(),
                func_name,
                args.iter()
                    .map(|x| {
                        eval::<IntType, FloatType, BoolType, HardType, E>(evaluator, env.clone(), x)
                    })
                    .collect(),
            ),
        },
        super::Expression::ExprWhere { bindings, inner } => {
            let new_env = bindings.iter().fold(env.vars, |acc, x| EnvVars::Rest {
                first: (
                    x.ident.clone(),
                    eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator,
                        Env {
                            program: env.program,
                            vars: acc.clone(),
                        },
                        &x.value,
                    ),
                ),
                rest: Box::new(acc),
            });

            eval::<IntType, FloatType, BoolType, HardType, E>(
                evaluator,
                Env {
                    program: env.program,
                    vars: new_env,
                },
                inner,
            )
        }
        super::Expression::IfThenElse {
            boolean,
            true_expr,
            false_expr,
        } => {
            match eval::<IntType, FloatType, BoolType, HardType, E>(evaluator, env.clone(), boolean)
            {
                EvalVal::Bool(x) => evaluator.eval_if(
                    x,
                    eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator,
                        env.clone(),
                        true_expr,
                    ),
                    eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator,
                        env.clone(),
                        false_expr,
                    ),
                ),
                _ => todo!(),
            }
        }
        super::Expression::FoldLoop {
            fold_iter,
            accumulator,
            body,
        } => {
            let fold_iter_val = eval::<IntType, FloatType, BoolType, HardType, E>(
                evaluator,
                env.clone(),
                &fold_iter
            );
            
            let iter = match fold_iter_val {
                EvalVal::Int(_) => todo!(),
                EvalVal::Float(_) => todo!(),
                EvalVal::Bool(_) => todo!(),
                EvalVal::Hard(_) => todo!(),
                EvalVal::List(vec) => {
                    vec
                },
                EvalVal::Product(_) => todo!(),
            };
            iter.into_iter().fold(
                eval::<IntType, FloatType, BoolType, HardType, E>(
                    evaluator,
                    env.clone(),
                    &accumulator.1,
                ),
                |acc, _| {
                    let new_vars = EnvVars::Rest {
                        first: (accumulator.0.clone(), acc),
                        rest: Box::new(env.vars.clone()),
                    };
                    let new_env = Env {
                        program: env.program,
                        vars: new_vars,
                    };

                    eval::<IntType, FloatType, BoolType, HardType, E>(evaluator, new_env, body)
                },
            )
        }
        super::Expression::WhileLoop {
            accumulator,
            cond,
            body,
            exit_body,
        } => {
            let mut acc = eval::<IntType, FloatType, BoolType, HardType, E>(
                evaluator,
                env.clone(),
                &accumulator.1,
            );

            let mut exits = vec![];

            let mut iters = 0;
            loop {
                let new_vars = EnvVars::Rest {
                    first: (accumulator.0.clone(), acc.clone()),
                    rest: Box::new(env.vars.clone()),
                };
                let new_env = Env {
                    program: env.program,
                    vars: new_vars,
                };

                let cond_val = {
                    if let EvalVal::Bool(x) = eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator,
                        new_env.clone(),
                        cond,
                    ) {
                        x
                    } else {
                        panic!()
                    }
                };

                exits.push((cond_val.clone(), acc.clone()));
                if evaluator.stop_while_eval(cond_val) {
                    break;
                }
                if iters > 500 {
                    break;
                }

                acc = eval::<IntType, FloatType, BoolType, HardType, E>(evaluator, new_env, body);

                iters += 1;
            }

            let last_body = {
                let new_vars = EnvVars::Rest {
                    first: (accumulator.0.clone(), exits.last().unwrap().1.clone()),
                    rest: Box::new(env.vars.clone()),
                };
                let new_env = Env {
                    program: env.program,
                    vars: new_vars,
                };

                eval::<IntType, FloatType, BoolType, HardType, E>(evaluator, new_env, exit_body)
            };

            exits[..exits.len() - 1].iter().fold(
                last_body,
                |acc: EvalVal<_, _, _, _>, (cond, val)| {
                    let new_vars = EnvVars::Rest {
                        first: (accumulator.0.clone(), val.clone()),
                        rest: Box::new(env.vars.clone()),
                    };
                    let new_env = Env {
                        program: env.program,
                        vars: new_vars,
                    };

                    let exit_val = eval::<IntType, FloatType, BoolType, HardType, E>(
                        evaluator, new_env, exit_body,
                    );

                    evaluator.eval_if(cond.clone(), acc, exit_val)
                },
            )
        }
        super::Expression::List {
            type_name: _,
            values,
        } => EvalVal::List(
            values
                .iter()
                .map(|x| {
                    eval::<IntType, FloatType, BoolType, HardType, E>(evaluator, env.clone(), x)
                })
                .collect(),
        ),
        super::Expression::Product(value) => EvalVal::Product(
            value
                .iter()
                .map(|x| {
                    eval::<IntType, FloatType, BoolType, HardType, E>(evaluator, env.clone(), x)
                })
                .collect(),
        ),
        super::Expression::ProductProject { value, index } => {
            match eval::<IntType, FloatType, BoolType, HardType, E>(evaluator, env.clone(), value) {
                EvalVal::Product(vals) => evaluator.eval_product_index(vals, index.clone()),
                _ => unreachable!(),
            }
        }
    }
}

fn apply_function<
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    HardType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    env: Env<IntType, FloatType, BoolType, HardType>,
    func_name: &str,
    arguments: Vec<EvalVal<IntType, FloatType, BoolType, HardType>>,
) -> EvalVal<IntType, FloatType, BoolType, HardType>
where
    E: Evaluator<IntType, FloatType, BoolType, HardType>,
{
    let func = env.program.find_func(func_name);

    assert!(arguments.len() == func.arguments.len());

    let env_with_args = arguments
        .into_iter()
        .zip(func.arguments.iter())
        .map(|(arg_val, arg)| (arg.0.clone(), arg_val))
        .fold(EnvVars::End, |acc, x| EnvVars::Rest {
            first: x,
            rest: Box::new(acc),
        });

    eval::<IntType, FloatType, BoolType, HardType, E>(
        evaluator,
        Env {
            program: env.program,
            vars: env_with_args,
        },
        &func.body,
    )
}

pub fn eval_test_cases<
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    HardType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    program: &super::Program<IntType, FloatType, BoolType, HardType>,
) -> Vec<EvalVal<IntType, FloatType, BoolType, HardType>>
where
    E: Evaluator<IntType, FloatType, BoolType, HardType>,
{
    program
        .test_cases
        .iter()
        .map(|test_case| {
            crate::ast::eval::eval::<IntType, FloatType, BoolType, HardType, E>(
                evaluator,
                Env {
                    program,
                    vars: EnvVars::End,
                },
                test_case,
            )
        })
        .collect()
}
