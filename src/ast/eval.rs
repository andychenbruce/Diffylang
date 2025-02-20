#[derive(Clone, Debug)]
pub enum EvalVal<IntType, FloatType, BoolType> {
    Int(IntType),
    Float(FloatType),
    Bool(BoolType),
    Product(Vec<EvalVal<IntType, FloatType, BoolType>>),
    Universe(u64),
}

#[derive(Clone)]
struct Env<'a, IntType, FloatType, BoolType> {
    program: &'a super::Program<IntType, FloatType, BoolType>,
    vars: EnvVars<IntType, FloatType, BoolType>,
}

#[derive(Clone)]
enum EnvVars<IntType, FloatType, BoolType> {
    End,
    Rest {
        first: (super::Identifier, EvalVal<IntType, FloatType, BoolType>),
        rest: Box<EnvVars<IntType, FloatType, BoolType>>,
    },
}

pub trait Evaluator<IntType, FloatType, BoolType> {
    fn eval_addition_ints(&self, a: IntType, b: IntType) -> IntType;
    fn eval_addition_floats(&self, a: FloatType, b: FloatType) -> FloatType;
    fn eval_multiplication_int(&self, a: IntType, b: IntType) -> IntType;
    fn eval_multiplication_floats(&self, a: FloatType, b: FloatType) -> FloatType;
    fn eval_negation_int(&self, a: IntType) -> IntType;
    fn eval_negation_float(&self, a: FloatType) -> FloatType;
    fn eval_equality_ints(&self, a: IntType, b: IntType) -> BoolType;
    fn eval_equality_floats(&self, a: FloatType, b: FloatType) -> BoolType;
    fn eval_less_than_ints(&self, a: IntType, b: IntType) -> BoolType;
    fn eval_less_than_floats(&self, a: FloatType, b: FloatType) -> BoolType;
    fn eval_not(&self, a: BoolType) -> BoolType;

    fn eval_and(&self, a: BoolType, b: BoolType) -> BoolType;
    fn eval_or(&self, a: BoolType, b: BoolType) -> BoolType;
    fn eval_if(
        &self,
        cond: BoolType,
        true_branch: EvalVal<IntType, FloatType, BoolType>,
        false_branch: EvalVal<IntType, FloatType, BoolType>,
    ) -> EvalVal<IntType, FloatType, BoolType>;
}

pub fn run_function<
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    program: &super::Program<IntType, FloatType, BoolType>,
    func_name: &str,
    arguments: Vec<EvalVal<IntType, FloatType, BoolType>>,
) -> EvalVal<IntType, FloatType, BoolType>
where
    E: Evaluator<IntType, FloatType, BoolType>,
{
    apply_function::<IntType, FloatType, BoolType, E>(
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
    > EnvVars<IntType, FloatType, BoolType>
{
    fn lookup_var(&self, var_name: &super::Identifier) -> EvalVal<IntType, FloatType, BoolType> {
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
    E,
>(
    evaluator: &E,
    env: Env<IntType, FloatType, BoolType>,
    expr: &super::Expression<IntType, FloatType, BoolType>,
) -> EvalVal<IntType, FloatType, BoolType>
where
    E: Evaluator<IntType, FloatType, BoolType>,
{
    match expr {
        super::Expression::Variable { ident } => env.vars.lookup_var(ident),
        super::Expression::Integer(x, _) => EvalVal::Int(x.clone()),
        super::Expression::Float(x, _) => EvalVal::Float(x.clone()),
        super::Expression::Bool(x, _) => EvalVal::Bool(x.clone()),
        super::Expression::Universe(x) => EvalVal::Universe(*x),

        super::Expression::FuncApplication { func_name, args } => match func_name.0.as_str() {
            "__add_int" | "__sub_int" | "__mul_int" | "__div_int" | "__eq_int" | "__lt_int"
            | "__gt_int" => {
                match (
                    eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), &args[0]),
                    eval::<IntType, FloatType, BoolType, E>(evaluator, env, &args[1]),
                ) {
                    (EvalVal::Int(a), EvalVal::Int(b)) => match func_name.0.as_str() {
                        "__add_int" => EvalVal::Int(evaluator.eval_addition_ints(a, b)),
                        "__sub_int" => EvalVal::Int(
                            evaluator.eval_addition_ints(a, evaluator.eval_negation_int(b)),
                        ),
                        "__mul_int" => EvalVal::Int(evaluator.eval_multiplication_int(a, b)),
                        "__div_int" => todo!(),
                        "__eq_int" => EvalVal::Bool(evaluator.eval_equality_ints(a, b)),
                        "__lt_int" => EvalVal::Bool(evaluator.eval_less_than_ints(a, b)),
                        "__gt_int" => EvalVal::Bool(evaluator.eval_less_than_ints(b, a)),
                        _ => todo!(),
                    },
                    _ => todo!(),
                }
            }
            "__add_flt" | "__sub_flt" | "__mul_flt" | "__div_flt" | "__eq_flt" | "__lt_flt"
            | "__gt_flt" => {
                match (
                    eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), &args[0]),
                    eval::<IntType, FloatType, BoolType, E>(evaluator, env, &args[1]),
                ) {
                    (EvalVal::Float(a), EvalVal::Float(b)) => match func_name.0.as_str() {
                        "__add_flt" => EvalVal::Float(evaluator.eval_addition_floats(a, b)),
                        "__sub_flt" => EvalVal::Float(
                            evaluator.eval_addition_floats(a, evaluator.eval_negation_float(b)),
                        ),
                        "__mul_flt" => EvalVal::Float(evaluator.eval_multiplication_floats(a, b)),
                        "__div_flt" => todo!(),
                        "__eq_flt" => EvalVal::Bool(evaluator.eval_equality_floats(a, b)),
                        "__lt_flt" => EvalVal::Bool(evaluator.eval_less_than_floats(a, b)),
                        "__gt_flt" => EvalVal::Bool(evaluator.eval_less_than_floats(b, a)),
                        _ => todo!(),
                    },
                    _ => todo!(),
                }
            }
            "__and" | "__or" => {
                match (
                    eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), &args[0]),
                    eval::<IntType, FloatType, BoolType, E>(evaluator, env, &args[1]),
                ) {
                    (EvalVal::Bool(a), EvalVal::Bool(b)) => match func_name.0.as_str() {
                        "__and" => EvalVal::Bool(evaluator.eval_and(a, b)),
                        "__or" => EvalVal::Bool(evaluator.eval_or(a, b)),
                        _ => todo!(),
                    },
                    _ => todo!(),
                }
            }
            "__neg_int" => {
                match eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), &args[0]) {
                    EvalVal::Int(x) => EvalVal::Int(evaluator.eval_negation_int(x)),
                    _ => todo!(),
                }
            }
            "__neg_flt" => {
                match eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), &args[0]) {
                    EvalVal::Float(x) => EvalVal::Float(evaluator.eval_negation_float(x)),
                    _ => todo!(),
                }
            }
            "__not" => {
                match eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), &args[0]) {
                    EvalVal::Bool(x) => EvalVal::Bool(evaluator.eval_not(x)),
                    _ => todo!(),
                }
            }
            func_name => apply_function::<IntType, FloatType, BoolType, E>(
                evaluator,
                env.clone(),
                func_name,
                args.iter()
                    .map(|x| eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), x))
                    .collect(),
            ),
        },
        super::Expression::ExprWhere { bindings, inner } => {
            let new_env = bindings.iter().fold(env.vars, |acc, x| EnvVars::Rest {
                first: (
                    x.ident.clone(),
                    eval::<IntType, FloatType, BoolType, E>(
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

            eval::<IntType, FloatType, BoolType, E>(
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
        } => match eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), boolean) {
            EvalVal::Bool(x) => evaluator.eval_if(
                x,
                eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), true_expr),
                eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), false_expr),
            ),
            _ => todo!(),
        },
        super::Expression::Product(value) => EvalVal::Product(
            value
                .iter()
                .map(|x| eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), x))
                .collect(),
        ),
    }
}

fn apply_function<
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    env: Env<IntType, FloatType, BoolType>,
    func_name: &str,
    arguments: Vec<EvalVal<IntType, FloatType, BoolType>>,
) -> EvalVal<IntType, FloatType, BoolType>
where
    E: Evaluator<IntType, FloatType, BoolType>,
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

    eval::<IntType, FloatType, BoolType, E>(
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
    E,
>(
    evaluator: &E,
    program: &super::Program<IntType, FloatType, BoolType>,
) -> Vec<EvalVal<IntType, FloatType, BoolType>>
where
    E: Evaluator<IntType, FloatType, BoolType>,
{
    program
        .test_cases
        .iter()
        .map(|test_case| {
            crate::ast::eval::eval::<IntType, FloatType, BoolType, E>(
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
