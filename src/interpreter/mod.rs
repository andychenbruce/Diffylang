use crate::ast;

#[derive(Copy, Clone, Debug)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone)]
enum Env {
    End,
    Rest {
        first: (ast::Identifier, Value),
        rest: Box<Env>,
    },
}

impl Env {
    fn lookup_var(&self, var_name: &ast::Identifier) -> Value {
        match self {
            Env::End => unreachable!(),
            Env::Rest { first, rest } => {
                if first.0 == *var_name {
                    return first.1;
                } else {
                    return rest.lookup_var(var_name);
                }
            }
        }
    }
}

pub fn apply_function(program: &ast::Program, func_name: &str, arguments: Vec<Value>) -> Value {
    let _type_env: crate::type_checker::TypeEnv =
        crate::type_checker::type_check_program(&program).unwrap();

    let func = program.find_func(func_name);

    assert!(arguments.len() == func.arguments.len());

    let env_with_args = arguments
        .into_iter()
        .zip(func.arguments.iter())
        .map(|(arg_val, arg)| (arg.0.clone(), arg_val))
        .fold(Env::End, |acc, x| Env::Rest {
            first: x,
            rest: Box::new(acc),
        });

    return eval(env_with_args, &func.body);
}

fn eval(env: Env, expr: &ast::Expression) -> Value {
    match expr {
        ast::Expression::Variable { ident, span: _ } => env.lookup_var(ident),
        ast::Expression::Integer(x) => Value::Int(*x),
        ast::Expression::Str(_) => todo!(),
        ast::Expression::Float(x) => Value::Float(*x),
        ast::Expression::FuncApplication {
            func_name,
            args,
            span: _,
        } => match func_name.0.as_str() {
            "__add" => eval_addition(env, &args[0], &args[1]),
            "__sub" => eval_subtraction(env, &args[0], &args[1]),
            "__mul" => todo!(),
            "__div" => todo!(),
            "__eq" => todo!(),
            "__gt" => eval_greater_than(env, &args[0], &args[1]),
            "__lt" => todo!(),
            _ => todo!(),
        },
        ast::Expression::ExprWhere { bindings, inner } => {
            let new_env = bindings.iter().fold(env, |acc, x| Env::Rest {
                first: (x.ident.clone(), eval(acc.clone(), &x.value)),
                rest: Box::new(acc),
            });

            return eval(new_env, inner);
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
