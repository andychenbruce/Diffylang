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
enum SoftEnv {
    End,
    Rest {
        first: (ast::Identifier, SoftValue),
        rest: Box<SoftEnv>,
    },
}

impl SoftEnv {
    fn lookup_var(&self, var_name: &ast::Identifier) -> SoftValue {
        match self {
            SoftEnv::End => unreachable!(),
            SoftEnv::Rest { first, rest } => {
                if first.0 == *var_name {
                    first.1
                } else {
                    rest.lookup_var(var_name)
                }
            }
        }
    }
}

pub fn soft_apply_function(
    program: &ast::Program,
    func_name: &str,
    arguments: Vec<SoftValue>,
) -> SoftValue {
    let _type_env: crate::type_checker::TypeEnv =
        crate::type_checker::type_check_program(program).unwrap();

    let func = program.find_func(func_name);

    assert!(arguments.len() == func.arguments.len());

    let env_with_args = arguments
        .into_iter()
        .zip(func.arguments.iter())
        .map(|(arg_val, arg)| (arg.0.clone(), arg_val))
        .fold(SoftEnv::End, |acc, x| SoftEnv::Rest {
            first: x,
            rest: Box::new(acc),
        });

    soft_eval(env_with_args, &func.body)
}

fn soft_eval(env: SoftEnv, expr: &ast::Expression) -> SoftValue {
    match expr {
        ast::Expression::Variable { ident, span: _ } => env.lookup_var(ident),
        ast::Expression::Integer(x) => SoftValue::Int(*x as f64),
        ast::Expression::Str(_) => todo!(),
        ast::Expression::Float(x) => SoftValue::Float(*x),
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
            _ => todo!(),
        },
        ast::Expression::ExprWhere { bindings, inner } => {
            let new_env = bindings.iter().fold(env, |acc, x| SoftEnv::Rest {
                first: (x.ident.clone(), soft_eval(acc.clone(), &x.value)),
                rest: Box::new(acc),
            });

            soft_eval(new_env, inner)
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
