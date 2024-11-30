use crate::ast;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone)]
struct Env<'a> {
    program: &'a ast::Program,
    vars: EnvVars,
}

#[derive(Clone)]
enum EnvVars {
    End,
    Rest {
        first: (ast::Identifier, Value),
        rest: Box<EnvVars>,
    },
}

impl EnvVars {
    fn lookup_var(&self, var_name: &ast::Identifier) -> Value {
        match self {
            EnvVars::End => unreachable!(),
            EnvVars::Rest { first, rest } => {
                if first.0 == *var_name {
                    first.1
                } else {
                    rest.lookup_var(var_name)
                }
            }
        }
    }
}

pub fn run_function(program: &ast::Program, func_name: &str, arguments: Vec<Value>) -> Value {
    let _type_env: crate::type_checker::TypeEnv =
        crate::type_checker::type_check_program(program).unwrap();

    apply_function(
        Env {
            program,
            vars: EnvVars::End,
        },
        func_name,
        arguments,
    )
}

pub fn eval_test_cases(program: &ast::Program) -> Vec<Value> {
    program
        .test_cases
        .iter()
        .map(|test_case| {
            eval(
                Env {
                    program,
                    vars: EnvVars::End,
                },
                test_case,
            )
        })
        .collect()
}

fn apply_function(env: Env, func_name: &str, arguments: Vec<Value>) -> Value {
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

    eval(
        Env {
            program: env.program,
            vars: env_with_args,
        },
        &func.body,
    )
}

fn eval(env: Env, expr: &ast::Expression) -> Value {
    match expr {
        ast::Expression::Variable { ident, span: _ } => env.vars.lookup_var(ident),
        ast::Expression::Integer(x, _) => Value::Int(*x),
        ast::Expression::Str(_, _) => todo!(),
        ast::Expression::Float(x, _) => Value::Float(*x),
        ast::Expression::Bool(x, _) => Value::Bool(*x),
        ast::Expression::FuncApplication {
            func_name,
            args,
            span: _,
        } => match func_name.0.as_str() {
            "__add" => eval_addition(env, &args[0], &args[1]),
            "__sub" => eval_subtraction(env, &args[0], &args[1]),
            "__mul" => eval_multiplication(env, &args[0], &args[1]),
            "__div" => todo!(),
            "__eq" => todo!(),
            "__gt" => eval_greater_than(env, &args[0], &args[1]),
            "__lt" => eval_less_than(env, &args[0], &args[1]),
            "__and" => todo!(),
            "__or" => todo!(),
            "__not" => eval_not(env, &args[0]),
            func_name => apply_function(
                env.clone(),
                func_name,
                args.iter().map(|x| eval(env.clone(), x)).collect(),
            ),
        },
        ast::Expression::ExprWhere { bindings, inner } => {
            let new_env = bindings.iter().fold(env.vars, |acc, x| EnvVars::Rest {
                first: (
                    x.ident.clone(),
                    eval(
                        Env {
                            program: env.program,
                            vars: acc.clone(),
                        },
                        &x.value,
                    ),
                ),
                rest: Box::new(acc),
            });

            eval(
                Env {
                    program: env.program,
                    vars: new_env,
                },
                inner,
            )
        }
        ast::Expression::IfThenElse {
            boolean,
            true_expr,
            false_expr,
        } => {
            if let Value::Bool(boolean) = eval(env.clone(), boolean) {
                if boolean {
                    eval(env.clone(), true_expr)
                } else {
                    eval(env.clone(), false_expr)
                }
            } else {
                panic!()
            }
        }
        ast::Expression::FoldLoop {
            range,
            accumulator,
            body,
        } => {
            let start = match eval(env.clone(), &range.0) {
                Value::Int(x) => x,
                _ => unreachable!(),
            };
            let end = match eval(env.clone(), &range.1) {
                Value::Int(x) => x,
                _ => unreachable!(),
            };

            (start..end).fold(eval(env.clone(), &accumulator.1), |acc, _| {
                let new_vars = EnvVars::Rest {
                    first: (accumulator.0.clone(), acc),
                    rest: Box::new(env.vars.clone()),
                };
                let new_env = Env {
                    program: env.program,
                    vars: new_vars,
                };

                eval(new_env, body)
            })
        }
    }
}

fn eval_addition(env: Env, left: &ast::Expression, right: &ast::Expression) -> Value {
    let left_val = eval(env.clone(), left);
    let right_val = eval(env.clone(), right);

    match (left_val, right_val) {
        (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
        (Value::Float(a), Value::Int(b)) => Value::Float(a + (b as f64)),
        (Value::Int(a), Value::Float(b)) => Value::Float((a as f64) + b),
        (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
        _ => unreachable!(),
    }
}

fn eval_subtraction(env: Env, left: &ast::Expression, right: &ast::Expression) -> Value {
    let left_val = eval(env.clone(), left);
    let right_val = eval(env.clone(), right);

    match (left_val, right_val) {
        (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
        (Value::Float(a), Value::Int(b)) => Value::Float(a - (b as f64)),
        (Value::Int(a), Value::Float(b)) => Value::Float((a as f64) - b),
        (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
        _ => unreachable!(),
    }
}

fn eval_multiplication(env: Env, left: &ast::Expression, right: &ast::Expression) -> Value {
    let left_val = eval(env.clone(), left);
    let right_val = eval(env.clone(), right);

    match (left_val, right_val) {
        (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
        (Value::Float(a), Value::Int(b)) => Value::Float(a * (b as f64)),
        (Value::Int(a), Value::Float(b)) => Value::Float((a as f64) * b),
        (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
        _ => unreachable!(),
    }
}

fn eval_greater_than(env: Env, left: &ast::Expression, right: &ast::Expression) -> Value {
    let left_val = eval(env.clone(), left);
    let right_val = eval(env.clone(), right);

    match (left_val, right_val) {
        (Value::Int(a), Value::Int(b)) => Value::Bool(a > b),
        (Value::Float(a), Value::Int(b)) => Value::Bool(a > (b as f64)),
        (Value::Int(a), Value::Float(b)) => Value::Bool((a as f64) > b),
        (Value::Float(a), Value::Float(b)) => Value::Bool(a > b),
        _ => unreachable!(),
    }
}
/* 
fn eval_division(env: Env, left: &ast::Expression, right: &ast::Expression) -> Value {
    let left_val = eval(env.clone(), left);
    let right_val = eval(env, right);

    match (left_val, right_val) {
        (Value::Int(a), Value::Int(b)) => {
            if b == 0 {
                panic!("Division by zero");
            }
            Value::Float(a as f64 / b as f64) // Convert to Float
        }
        (Value::Float(a), Value::Int(b)) => {
            if b == 0 {
                panic!("Division by zero");
            }
            Value::Float(a / b as f64)
        }
        (Value::Int(a), Value::Float(b)) => {
            if b == 0.0 {
                panic!("Division by zero");
            }
            Value::Float(a as f64 / b)
        }
        (Value::Float(a), Value::Float(b)) => {
            if b == 0.0 {
                panic!("Division by zero");
            }
            Value::Float(a / b)
        }
        _ => panic!("Type error in division"),
    }
}
*/

fn eval_less_than(env: Env, left: &ast::Expression, right: &ast::Expression) -> Value {
    let left_val = eval(env.clone(), left);
    let right_val = eval(env.clone(), right);

    match (left_val, right_val) {
        (Value::Int(a), Value::Int(b)) => Value::Bool(a < b),
        (Value::Float(a), Value::Int(b)) => Value::Bool(a < (b as f64)),
        (Value::Int(a), Value::Float(b)) => Value::Bool((a as f64) < b),
        (Value::Float(a), Value::Float(b)) => Value::Bool(a < b),
        _ => unreachable!(),
    }
}

fn eval_not(env: Env, val: &ast::Expression) -> Value {
    let val = eval(env.clone(), val);

    match val {
        Value::Bool(a) => Value::Bool(!a),
        _ => unreachable!(),
    }
}
