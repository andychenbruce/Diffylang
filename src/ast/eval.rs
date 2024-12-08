use super::Program;

#[derive(Clone)]
enum EvalVal<IntType, FloatType, BoolType, HardType> {
    Int(IntType),
    Float(FloatType),
    Bool(BoolType),
    Hard(HardType),
    List(Vec<EvalVal<IntType, FloatType, BoolType, HardType>>),
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

trait Evaluator<IntType, FloatType, BoolType, HardType> {
    fn eval_addition_ints(a: IntType, b: IntType) -> IntType;
    fn eval_addition_floats(a: FloatType, b: FloatType) -> FloatType;
    fn eval_multiplication_intfn(a: IntType, b: IntType) -> IntType;
    fn eval_multiplication_floats(a: FloatType, b: FloatType) -> FloatType;
    fn eval_negation_int(a: IntType) -> IntType;
    fn eval_negation_float(a: FloatType) -> FloatType;
    fn eval_equality_ints(a: IntType, b: IntType) -> BoolType;
    fn eval_equality_floats(a: FloatType, b: FloatType) -> BoolType;
    fn eval_not(a: BoolType) -> BoolType;
    fn eval_if(
        cond: BoolType,
        true_branch: EvalVal<IntType, FloatType, BoolType, HardType>,
        false_branch: EvalVal<IntType, FloatType, BoolType, HardType>,
    ) -> EvalVal<IntType, FloatType, BoolType, HardType>;

    fn make_range(start: IntType, end: IntType) -> Vec<IntType>;
}

struct FuncTable {}

fn traverse_eval<IntType: Clone, FloatType: Clone, BoolType: Clone, HardType: Clone, E>(
    program: Program<IntType, FloatType, BoolType, HardType>,
    evaluator: E,
) -> EvalVal<IntType, FloatType, BoolType, HardType>
where
    E: Evaluator<IntType, FloatType, BoolType, HardType>,
{
    todo!()
}

impl<IntType: Clone, FloatType: Clone, BoolType: Clone, HardType: Clone>
    EnvVars<IntType, FloatType, BoolType, HardType>
{
    fn lookup_var(
        &self,
        var_name: &super::Identifier,
    ) -> EvalVal<IntType, FloatType, BoolType, HardType> {
        match self {
            EnvVars::End => unreachable!(),
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

fn eval<IntType: Clone, FloatType: Clone, BoolType: Clone, HardType: Clone, E>(
    env: Env<IntType, FloatType, BoolType, HardType>,
    expr: &super::Expression<IntType, FloatType, BoolType, HardType>,
) -> EvalVal<IntType, FloatType, BoolType, HardType>
where
    E: Evaluator<IntType, FloatType, BoolType, HardType>,
{
    match expr {
        super::Expression::Variable { ident, span: _ } => env.vars.lookup_var(ident),
        super::Expression::Integer(x, _) => EvalVal::Int(x.clone()),
        super::Expression::HardInt(x) => EvalVal::Hard(x.clone()),
        super::Expression::Str(_, _) => todo!(),
        super::Expression::Float(x, _) => EvalVal::Float(x.clone()),
        super::Expression::Bool(x, _) => EvalVal::Bool(x.clone()),
        super::Expression::FuncApplication {
            func_name,
            args,
            span: _, // eval_subtraction(env, &args[0], &args[1]),
        } => match func_name.0.as_str() {
            "__add" | "__sub" | "__mul" | "__div" | "__eq" | "__gt" | "__lt" | "__and" | "__or" => {
                match (
                    eval::<IntType, FloatType, BoolType, HardType, E>(env.clone(), &args[0]),
                    eval::<IntType, FloatType, BoolType, HardType, E>(env, &args[1]),
                ) {
                    (EvalVal::Int(a), EvalVal::Int(b)) => match func_name.0.as_str() {
                        "__add" => EvalVal::Int(E::eval_addition_ints(a, b)),
                        "__sub" => EvalVal::Int(E::eval_addition_ints(a, E::eval_negation_int(b))),
                        "__mul" => EvalVal::Int(E::eval_multiplication_intfn(a, b)),
                        "__div" => todo!(),
                        "__eq" => EvalVal::Bool(E::eval_equality_ints(a, b)),
                        _ => todo!(),
                    },
                    _ => todo!(),
                }
            }
            "__not" => {
                match eval::<IntType, FloatType, BoolType, HardType, E>(env.clone(), &args[0]) {
                    EvalVal::Bool(x) => EvalVal::Bool(E::eval_not(x)),
                    _ => todo!(),
                }
            }

            func_name => apply_function::<IntType, FloatType, BoolType, HardType, E>(
                env.clone(),
                func_name,
                args.iter()
                    .map(|x| eval::<IntType, FloatType, BoolType, HardType, E>(env.clone(), x))
                    .collect(),
            ),
        },
        super::Expression::ExprWhere { bindings, inner } => {
            let new_env = bindings.iter().fold(env.vars, |acc, x| EnvVars::Rest {
                first: (
                    x.ident.clone(),
                    eval::<IntType, FloatType, BoolType, HardType, E>(
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
        } => match eval::<IntType, FloatType, BoolType, HardType, E>(env.clone(), boolean) {
            EvalVal::Bool(x) => E::eval_if(
                x,
                eval::<IntType, FloatType, BoolType, HardType, E>(env.clone(), true_expr),
                eval::<IntType, FloatType, BoolType, HardType, E>(env.clone(), false_expr),
            ),
            _ => todo!(),
        },
        super::Expression::FoldLoop {
            fold_iter,
            accumulator,
            body,
        } => {
            let iter = match **fold_iter {
                super::FoldIter::Range(ref start, ref end) => {
                    let start = match eval::<IntType, FloatType, BoolType, HardType, E>(
                        env.clone(),
                        &start,
                    ) {
                        EvalVal::Int(x) => x,
                        _ => unreachable!(),
                    };

                    let end = match eval::<IntType, FloatType, BoolType, HardType, E>(
                        env.clone(),
                        &end,
                    ) {
                        EvalVal::Int(x) => x,
                        _ => unreachable!(),
                    };

                    E::make_range(start, end)
                        .into_iter()
                        .map(|x| EvalVal::Int(x))
                        .collect()
                }
                super::FoldIter::ExprList(ref list) => {
                    match eval::<IntType, FloatType, BoolType, HardType, E>(env.clone(), &list) {
                        EvalVal::List(l) => l,
                        _ => unreachable!(),
                    }
                }
            };
            iter.into_iter().fold(
                eval::<IntType, FloatType, BoolType, HardType, E>(env.clone(), &accumulator.1),
                |acc, _| {
                    let new_vars = EnvVars::Rest {
                        first: (accumulator.0.clone(), acc),
                        rest: Box::new(env.vars.clone()),
                    };
                    let new_env = Env {
                        program: env.program,
                        vars: new_vars,
                    };

                    eval::<IntType, FloatType, BoolType, HardType, E>(new_env, body)
                },
            )
        }
        super::Expression::List {
            type_name: _,
            values,
        } => EvalVal::List(values.into_iter().map(|x| eval::<IntType, FloatType, BoolType, HardType, E>(env.clone(), x)).collect()),
    }
}

fn apply_function<IntType: Clone, FloatType: Clone, BoolType: Clone, HardType: Clone, E>(
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
        Env {
            program: env.program,
            vars: env_with_args,
        },
        &func.body,
    )
}
