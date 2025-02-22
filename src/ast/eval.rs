#[derive(Clone, Debug)]
pub enum EvalVal<IntType, FloatType, BoolType> {
    Int(IntType),
    Float(FloatType),
    Bool(BoolType),
    Universe(u64),
    Lambda {
        captured_environment: Box<EnvVars<IntType, FloatType, BoolType>>,
        input: super::Identifier,
        expr: super::Expression<IntType, FloatType, BoolType>,
    },
    DependentProduct {
        first: Box<EvalVal<IntType, FloatType, BoolType>>,
        second: Box<EvalVal<IntType, FloatType, BoolType>>,
    },
    DependentProductType {
        first_type: Box<EvalVal<IntType, FloatType, BoolType>>,
        second_type: Box<EvalVal<IntType, FloatType, BoolType>>,
    },
    DependentFunctionType {
        type_to: Box<EvalVal<IntType, FloatType, BoolType>>,
        type_from: Box<EvalVal<IntType, FloatType, BoolType>>,
    },
    BuiltinFunc(BuiltinFunc<IntType, FloatType, BoolType>),
}

#[derive(Clone, Debug)]
pub enum BuiltinFunc<IntType, FloatType, BoolType> {
    AddInt,
    SubInt,
    MulInt,
    NegInt,
    EqInt,
    LtInt,

    AddFlt,
    SubFlt,
    MulFlt,
    NegFlt,
    EqFlt,
    LtFlt,

    BoolAnd,
    BoolOr,
    BoolNot,

    PartialAppAddInt(IntType),
    PartialAppSubInt(IntType),
    PartialAppMulInt(IntType),
    PartialAppEqInt(IntType),
    PartialAppLtInt(IntType),

    PartialAppAddFlt(FloatType),
    PartialAppSubFlt(FloatType),
    PartialAppMulFlt(FloatType),
    PartialAppEqFlt(FloatType),
    PartialAppLtFlt(FloatType),

    PartialAppBoolAnd(BoolType),
    PartialAppBoolOr(BoolType),

    IfThenElse,
    PartialAppIfThenElse1(BoolType),
    PartialAppIfThenElse2(BoolType, Box<EvalVal<IntType, FloatType, BoolType>>),
}

#[derive(Clone)]
struct Env<'a, IntType, FloatType, BoolType> {
    program: &'a super::Program<IntType, FloatType, BoolType>,
    vars: EnvVars<IntType, FloatType, BoolType>,
}

#[derive(Clone, Debug)]
pub enum EnvVars<IntType, FloatType, BoolType> {
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
    'a,
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    program: &'a super::Program<IntType, FloatType, BoolType>,
    func_name: &str,
    arguments: Vec<EvalVal<IntType, FloatType, BoolType>>,
) -> EvalVal<IntType, FloatType, BoolType>
where
    E: Evaluator<IntType, FloatType, BoolType>,
{
    let env = Env {
        program,
        vars: EnvVars::End,
    };

    let func = env.lookup_expr(&super::Identifier(func_name.to_owned()));
    apply_func_multiple_args::<IntType, FloatType, BoolType, E>(evaluator, env, func, arguments)
}

impl<
        'a,
        IntType: Clone + core::fmt::Debug,
        FloatType: Clone + core::fmt::Debug,
        BoolType: Clone + core::fmt::Debug,
    > Env<'a, IntType, FloatType, BoolType>
{
    fn lookup_expr(&self, var_name: &super::Identifier) -> EvalVal<IntType, FloatType, BoolType> {
        match self.vars.lookup(var_name) {
            Some(x) => x,
            None => todo!(),
        }
    }
}
impl<
        IntType: Clone + core::fmt::Debug,
        FloatType: Clone + core::fmt::Debug,
        BoolType: Clone + core::fmt::Debug,
    > EnvVars<IntType, FloatType, BoolType>
{
    fn lookup(
        &self,
        var_name: &super::Identifier,
    ) -> Option<EvalVal<IntType, FloatType, BoolType>> {
        match var_name.0.as_str() {
            "__add_int" => Some(EvalVal::BuiltinFunc(BuiltinFunc::AddInt)),
            "__sub_int" => Some(EvalVal::BuiltinFunc(BuiltinFunc::SubInt)),
            "__mul_int" => Some(EvalVal::BuiltinFunc(BuiltinFunc::MulInt)),
            "__neg_int" => Some(EvalVal::BuiltinFunc(BuiltinFunc::NegInt)),
            "__eq_int" => Some(EvalVal::BuiltinFunc(BuiltinFunc::EqInt)),
            "__lt_int" => Some(EvalVal::BuiltinFunc(BuiltinFunc::LtInt)),
            "__add_flt" => Some(EvalVal::BuiltinFunc(BuiltinFunc::AddFlt)),
            "__sub_flt" => Some(EvalVal::BuiltinFunc(BuiltinFunc::SubFlt)),
            "__mul_flt" => Some(EvalVal::BuiltinFunc(BuiltinFunc::MulFlt)),
            "__neg_flt" => Some(EvalVal::BuiltinFunc(BuiltinFunc::NegFlt)),
            "__eq_flt" => Some(EvalVal::BuiltinFunc(BuiltinFunc::EqFlt)),
            "__lt_flt" => Some(EvalVal::BuiltinFunc(BuiltinFunc::LtFlt)),
            "__and" => Some(EvalVal::BuiltinFunc(BuiltinFunc::BoolAnd)),
            "__or" => Some(EvalVal::BuiltinFunc(BuiltinFunc::BoolOr)),
            "__not" => Some(EvalVal::BuiltinFunc(BuiltinFunc::BoolNot)),
            "__if" => Some(EvalVal::BuiltinFunc(BuiltinFunc::IfThenElse)),

            var_str => match self {
                EnvVars::End => None,
                EnvVars::Rest { first, rest } => {
                    if first.0 .0 == var_str {
                        Some(first.1.clone())
                    } else {
                        rest.lookup(var_name)
                    }
                }
            },
        }
    }
}

fn eval<
    'a,
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    env: Env<'a, IntType, FloatType, BoolType>,
    expr: &super::Expression<IntType, FloatType, BoolType>,
) -> EvalVal<IntType, FloatType, BoolType>
where
    E: Evaluator<IntType, FloatType, BoolType>,
{
    match expr {
        super::Expression::Variable { ident } => env.lookup_expr(ident),
        super::Expression::Integer(x, _) => EvalVal::Int(x.clone()),
        super::Expression::Float(x, _) => EvalVal::Float(x.clone()),
        super::Expression::Bool(x, _) => EvalVal::Bool(x.clone()),
        super::Expression::Universe(x) => EvalVal::Universe(*x),

        super::Expression::FuncApplicationMultipleArgs { func, args } => {
            let func_val = eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), func);

            let args_val = args
                .iter()
                .map(|x| eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), x))
                .collect();

            apply_func_multiple_args(evaluator, env, func_val, args_val)
        }
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
        super::Expression::DependentProductType {
            type_first,
            type_second,
        } => {
            let first_type =
                eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), &type_first.arg_type);

            let new_env: Env<IntType, FloatType, BoolType> = Env {
                program: env.program,
                vars: EnvVars::Rest {
                    first: (type_first.name.clone(), first_type.clone()),
                    rest: Box::new(env.vars.clone()),
                },
            };

            let second_type =
                eval::<IntType, FloatType, BoolType, E>(evaluator, new_env, type_second);

            EvalVal::DependentProductType {
                first_type: Box::new(first_type),
                second_type: Box::new(second_type),
            }
        }
        super::Expression::DependentFunctionType { type_from, type_to } => {
            let from_type =
                eval::<IntType, FloatType, BoolType, E>(evaluator, env.clone(), &type_from.arg_type);

            let new_env: Env<IntType, FloatType, BoolType> = Env {
                program: env.program,
                vars: EnvVars::Rest {
                    first: (type_from.name.clone(), from_type.clone()),
                    rest: Box::new(env.vars.clone()),
                },
            };

            let to_type = eval::<IntType, FloatType, BoolType, E>(evaluator, new_env, type_to);

            EvalVal::DependentFunctionType {
                type_from: Box::new(from_type),
                type_to: Box::new(to_type),
            }
        }
        super::Expression::Lambda { input: _, body: _ } => todo!(),
    }
}

pub fn eval_test_cases<
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    program: &super::Program<IntType, FloatType, BoolType>,
) -> Vec<BoolType>
where
    E: Evaluator<IntType, FloatType, BoolType>,
{
    program
        .test_cases
        .iter()
        .map(|test_case| {
            match crate::ast::eval::eval::<IntType, FloatType, BoolType, E>(
                evaluator,
                Env {
                    program,
                    vars: EnvVars::End,
                },
                test_case,
            ) {
                EvalVal::Bool(x) => x,
                _ => unreachable!(),
            }
        })
        .collect()
}

fn apply_func<
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    env: Env<IntType, FloatType, BoolType>,
    func: EvalVal<IntType, FloatType, BoolType>,
    arg: EvalVal<IntType, FloatType, BoolType>,
) -> EvalVal<IntType, FloatType, BoolType>
where
    E: Evaluator<IntType, FloatType, BoolType>,
{
    match func {
        EvalVal::Lambda {
            captured_environment: _,
            input: _,
            expr: _,
        } => {
            todo!()
        }
        EvalVal::BuiltinFunc(builtin_func) => match builtin_func {
            BuiltinFunc::AddInt => match arg {
                EvalVal::Int(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppAddInt(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::SubInt => match arg {
                EvalVal::Int(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppSubInt(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::MulInt => match arg {
                EvalVal::Int(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppMulInt(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::NegInt => match arg {
                EvalVal::Int(x) => EvalVal::Int(evaluator.eval_negation_int(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::EqInt => match arg {
                EvalVal::Int(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppEqInt(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::LtInt => match arg {
                EvalVal::Int(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppLtInt(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::AddFlt => match arg {
                EvalVal::Float(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppAddFlt(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::SubFlt => match arg {
                EvalVal::Float(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppSubFlt(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::MulFlt => match arg {
                EvalVal::Float(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppMulFlt(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::NegFlt => match arg {
                EvalVal::Float(x) => EvalVal::Float(evaluator.eval_negation_float(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::EqFlt => match arg {
                EvalVal::Float(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppEqFlt(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::LtFlt => match arg {
                EvalVal::Float(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppLtFlt(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::BoolAnd => match arg {
                EvalVal::Bool(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppBoolAnd(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::BoolOr => match arg {
                EvalVal::Bool(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppBoolOr(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::BoolNot => match arg {
                EvalVal::Bool(x) => EvalVal::Bool(evaluator.eval_not(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppAddInt(x) => match arg {
                EvalVal::Int(y) => EvalVal::Int(evaluator.eval_addition_ints(x, y)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppSubInt(x) => match arg {
                EvalVal::Int(y) => {
                    EvalVal::Int(evaluator.eval_addition_ints(x, evaluator.eval_negation_int(y)))
                }
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppMulInt(x) => match arg {
                EvalVal::Int(y) => EvalVal::Int(evaluator.eval_multiplication_int(x, y)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppEqInt(x) => match arg {
                EvalVal::Int(y) => EvalVal::Bool(evaluator.eval_equality_ints(x, y)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppLtInt(x) => match arg {
                EvalVal::Int(y) => EvalVal::Bool(evaluator.eval_less_than_ints(x, y)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppAddFlt(x) => match arg {
                EvalVal::Float(y) => EvalVal::Float(evaluator.eval_addition_floats(x, y)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppSubFlt(x) => match arg {
                EvalVal::Float(y) => EvalVal::Float(
                    evaluator.eval_addition_floats(x, evaluator.eval_negation_float(y)),
                ),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppMulFlt(x) => match arg {
                EvalVal::Float(y) => EvalVal::Float(evaluator.eval_multiplication_floats(x, y)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppEqFlt(x) => match arg {
                EvalVal::Float(y) => EvalVal::Bool(evaluator.eval_equality_floats(x, y)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppLtFlt(x) => match arg {
                EvalVal::Float(y) => EvalVal::Bool(evaluator.eval_less_than_floats(x, y)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppBoolAnd(x) => match arg {
                EvalVal::Bool(y) => EvalVal::Bool(evaluator.eval_and(x, y)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppBoolOr(x) => match arg {
                EvalVal::Bool(y) => EvalVal::Bool(evaluator.eval_or(x, y)),
                _ => unreachable!(),
            },
            BuiltinFunc::IfThenElse => match arg {
                EvalVal::Bool(x) => EvalVal::BuiltinFunc(BuiltinFunc::PartialAppIfThenElse1(x)),
                _ => unreachable!(),
            },
            BuiltinFunc::PartialAppIfThenElse1(x) => {
                EvalVal::BuiltinFunc(BuiltinFunc::PartialAppIfThenElse2(x, Box::new(arg)))
            }

            BuiltinFunc::PartialAppIfThenElse2(x, true_expr) => {
                evaluator.eval_if(x, *true_expr, arg)
            }
        },

        _ => unreachable!(),
    }
}

fn apply_func_multiple_args<
    IntType: Clone + core::fmt::Debug,
    FloatType: Clone + core::fmt::Debug,
    BoolType: Clone + core::fmt::Debug,
    E,
>(
    evaluator: &E,
    env: Env<IntType, FloatType, BoolType>,
    func: EvalVal<IntType, FloatType, BoolType>,
    args: Vec<EvalVal<IntType, FloatType, BoolType>>,
) -> EvalVal<IntType, FloatType, BoolType>
where
    E: Evaluator<IntType, FloatType, BoolType>,
{
    let first_app = apply_func(evaluator, env.clone(), func, args[0].clone());
    args.into_iter()
        .skip(1)
        .fold(first_app, |func_curried, next_arg| {
            apply_func(evaluator, env.clone(), func_curried, next_arg)
        })
}
