use crate::ast;

#[derive(Copy, Clone, Debug)]
pub enum SoftValue {
    Int(f64),
    Float(f64),
    Bool(f64),
}

#[derive(Clone, Debug)]
enum Gradient {
    Int(Vec<f64>),
    Float(Vec<f64>),
    Bool(Vec<f64>),
}

#[derive(Clone)]
struct SoftEnv<'a> {
    program: &'a ast::Program,
    vars: SoftEnvVars,
}

#[derive(Clone)]
enum SoftEnvVars {
    End,
    Rest {
        first: (ast::Identifier, SoftValue),
        rest: Box<SoftEnvVars>,
    },
}

impl SoftEnvVars {
    fn lookup_var(&self, var_name: &ast::Identifier) -> SoftValue {
        match self {
            SoftEnvVars::End => unreachable!(),
            SoftEnvVars::Rest { first, rest } => {
                if first.0 == *var_name {
                    first.1
                } else {
                    rest.lookup_var(var_name)
                }
            }
        }
    }
}

pub fn soft_run_function(
    program: &ast::Program,
    func_name: &str,
    arguments: Vec<SoftValue>,
) -> SoftValue {
    let _type_env: crate::type_checker::TypeEnv =
        crate::type_checker::type_check_program(program).unwrap();

    soft_apply_function(
        SoftEnv {
            program,
            vars: SoftEnvVars::End,
        },
        func_name,
        arguments,
    )
}

fn soft_apply_function(env: SoftEnv, func_name: &str, arguments: Vec<SoftValue>) -> SoftValue {
    let func = env.program.find_func(func_name);

    assert!(arguments.len() == func.arguments.len());

    let env_with_args = arguments
        .into_iter()
        .zip(func.arguments.iter())
        .map(|(arg_val, arg)| (arg.0.clone(), arg_val))
        .fold(SoftEnvVars::End, |acc, x| SoftEnvVars::Rest {
            first: x,
            rest: Box::new(acc),
        });

    soft_eval(
        SoftEnv {
            program: env.program,
            vars: env_with_args,
        },
        &func.body,
    )
}

fn soft_eval(env: SoftEnv, expr: &ast::Expression) -> SoftValue {
    match expr {
        ast::Expression::Variable { ident, span: _ } => env.vars.lookup_var(ident),
        ast::Expression::Integer(x, _) => SoftValue::Int(*x as f64),
        ast::Expression::Str(_, _) => todo!(),
        ast::Expression::Float(x, _) => SoftValue::Float(*x),
        ast::Expression::FuncApplication {
            func_name,
            args,
            span: _,
        } => match func_name.0.as_str() {
            "__add" => eval_soft_addition(env, &args[0], &args[1]),
            "__sub" => eval_soft_subtraction(env, &args[0], &args[1]),
            "__mul" => todo!(),
            "__div" => todo!(),
            "__eq" => todo!(),
            "__gt" => eval_soft_greater_than(env, &args[0], &args[1]),
            "__lt" => todo!(),
            func_name => soft_apply_function(
                env.clone(),
                func_name,
                args.iter().map(|x| soft_eval(env.clone(), x)).collect(),
            ),
        },
        ast::Expression::ExprWhere { bindings, inner } => {
            let new_env = bindings.iter().fold(env.vars, |acc, x| SoftEnvVars::Rest {
                first: (
                    x.ident.clone(),
                    soft_eval(
                        SoftEnv {
                            program: env.program,
                            vars: acc.clone(),
                        },
                        &x.value,
                    ),
                ),
                rest: Box::new(acc),
            });

            soft_eval(
                SoftEnv {
                    program: env.program,
                    vars: new_env,
                },
                inner,
            )
        }
    }
}

fn eval_soft_addition(env: SoftEnv, left: &ast::Expression, right: &ast::Expression) -> SoftValue {
    let left_val = soft_eval(env.clone(), left);
    let right_val = soft_eval(env.clone(), right);

    match (left_val, right_val) {
        (SoftValue::Int(a), SoftValue::Int(b)) => SoftValue::Int(a + b),
        (SoftValue::Float(a), SoftValue::Int(b)) => SoftValue::Float(a + b),
        (SoftValue::Int(a), SoftValue::Float(b)) => SoftValue::Float(a + b),
        (SoftValue::Float(a), SoftValue::Float(b)) => SoftValue::Float(a + b),
        _ => unreachable!(),
    }
}

fn eval_soft_subtraction(
    env: SoftEnv,
    left: &ast::Expression,
    right: &ast::Expression,
) -> SoftValue {
    let left_val = soft_eval(env.clone(), left);
    let right_val = soft_eval(env.clone(), right);

    match (left_val, right_val) {
        (SoftValue::Int(a), SoftValue::Int(b)) => SoftValue::Int(a - b),
        (SoftValue::Float(a), SoftValue::Int(b)) => SoftValue::Float(a - b),
        (SoftValue::Int(a), SoftValue::Float(b)) => SoftValue::Float(a - b),
        (SoftValue::Float(a), SoftValue::Float(b)) => SoftValue::Float(a - b),
        _ => unreachable!(),
    }
}

fn eval_soft_greater_than(
    env: SoftEnv,
    left: &ast::Expression,
    right: &ast::Expression,
) -> SoftValue {
    let left_val = soft_eval(env.clone(), left);
    let right_val = soft_eval(env.clone(), right);

    match (left_val, right_val) {
        (SoftValue::Float(a), SoftValue::Float(b)) => SoftValue::Bool(softgt(a, b)),
        (SoftValue::Int(a), SoftValue::Float(b)) => SoftValue::Bool(softgt(a, b)),
        (SoftValue::Float(a), SoftValue::Int(b)) => SoftValue::Bool(softgt(a, b)),
        (SoftValue::Int(a), SoftValue::Int(b)) => SoftValue::Bool(softgt(a, b)),
        _ => unreachable!(),
    }
}

pub fn softgt(x: f64, c: f64) -> f64 {
    1.0 / (1.0 + (-1.0 * (x - c)).exp())
}
