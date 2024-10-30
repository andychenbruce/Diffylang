use crate::parser::Program;

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
        first: (parsel::ast::Ident, Value),
        rest: Box<Env>,
    },
}

impl Env {
    fn lookup_var(&self, var_name: parsel::ast::Ident) -> Value {
        match self {
            Env::End => unreachable!(),
            Env::Rest { first, rest } => {
                if first.0 == var_name {
                    return first.1;
                } else {
                    return rest.lookup_var(var_name);
                }
            }
        }
    }
}

pub fn apply_function(program: Program, func_name: &str, arguments: Vec<Value>) -> Value {
    let _type_env: crate::type_checker::TypeEnv =
        crate::type_checker::type_check_program(&program).unwrap();

    let func = program.find_func(func_name);
    let thing = func
        .inner
        .arguments
        .clone()
        .into_inner()
        .clone()
        .into_iter();

    assert!(arguments.len() == thing.len());

    let env_with_args = arguments
        .into_iter()
        .zip(thing)
        .map(|(arg_val, arg)| (arg.varname.clone(), arg_val))
        .fold(Env::End, |acc, x| Env::Rest {
            first: x,
            rest: Box::new(acc),
        });

    return eval(env_with_args, func.inner.func_body.clone().into_inner());
}

fn eval(env: Env, expr: crate::parser::Expression) -> Value {
    match expr {
        crate::parser::Expression::Variable(var_name) => env.lookup_var(var_name),
        crate::parser::Expression::IntegerLit(x) => Value::Int(x.into_inner()),
        crate::parser::Expression::StringLit(_) => todo!(),
        crate::parser::Expression::FloatLit(x) => Value::Float(x.into_inner().into()),
        crate::parser::Expression::Addition(ad) => {
            eval_addition(env, *ad.left_side.clone(), *ad.right_side.clone())
        }
        crate::parser::Expression::Subtraction(sb) => {
            eval_subtraction(env, *sb.left_side.clone(), *sb.right_side.clone())
        }
        crate::parser::Expression::Multiplication(_) => todo!(),
        crate::parser::Expression::Division(_) => todo!(),
        crate::parser::Expression::Equality(_) => todo!(),
        crate::parser::Expression::GreaterThan(gt) => {
            eval_greater_than(env, *gt.left_side.clone(), *gt.right_side.clone())
        }
        crate::parser::Expression::LessThan(_) => todo!(),
        crate::parser::Expression::ExprWhere {
            bindings,
            where_token: _,
            inner,
        } => {
            let new_env = bindings
                .into_inner()
                .into_inner()
                .into_iter()
                .map(|x| x.into_inner())
                .fold(env, |acc, x| Env::Rest {
                    first: (x.name, eval(acc.clone(), *x.value)),
                    rest: Box::new(acc),
                });

            return eval(new_env, *inner);
        }
    }
}

fn eval_addition(
    env: Env,
    left: crate::parser::Expression,
    right: crate::parser::Expression,
) -> Value {
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

fn eval_subtraction(
    env: Env,
    left: crate::parser::Expression,
    right: crate::parser::Expression,
) -> Value {
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

fn eval_greater_than(
    env: Env,
    left: crate::parser::Expression,
    right: crate::parser::Expression,
) -> Value {
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
