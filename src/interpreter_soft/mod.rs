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
enum Env {
    End,
    Rest {
        first: (ast::Identifier, SoftValue),
        rest: Box<Env>,
    },
}

impl Env {
    fn lookup_var(&self, var_name: &ast::Identifier) -> SoftValue {
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

fn soft_eval(env: Env, expr: &ast::Expression) -> SoftValue {
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
            let new_env = bindings.iter().fold(env, |acc, x| Env::Rest {
                first: (x.ident.clone(), soft_eval(acc.clone(), &x.value)),
                rest: Box::new(acc),
            });

            return soft_eval(new_env, inner);
        }
    }
}

fn eval_soft_addition(env: Env, left: &ast::Expression, right: &ast::Expression) -> SoftValue {
    let left_val = soft_eval(env.clone(), left);
    let right_val = soft_eval(env.clone(), right);

    match (left_val, right_val) {
        (SoftValue::Int(a), SoftValue::Int(b)) => SoftValue::Int(a + b),
        (SoftValue::Float(a), SoftValue::Int(b)) => SoftValue::Float(a + (b as f64)),
        (SoftValue::Int(a), SoftValue::Float(b)) => SoftValue::Float((a as f64) + b),
        (SoftValue::Float(a), SoftValue::Float(b)) => SoftValue::Float(a + b),
        _ => unreachable!(),
    }
}

fn eval_soft_subtraction(env: Env, left: &ast::Expression, right: &ast::Expression) -> SoftValue {
    let left_val = soft_eval(env.clone(), left);
    let right_val = soft_eval(env.clone(), right);

    match (left_val, right_val) {
        (SoftValue::Int(a), SoftValue::Int(b)) => SoftValue::Int(a - b),
        (SoftValue::Float(a), SoftValue::Int(b)) => SoftValue::Float(a - (b as f64)),
        (SoftValue::Int(a), SoftValue::Float(b)) => SoftValue::Float((a as f64) - b),
        (SoftValue::Float(a), SoftValue::Float(b)) => SoftValue::Float(a - b),
        _ => unreachable!(),
    }
}


fn eval_soft_greater_than(env: Env, left: &ast::Expression, right: &ast::Expression) -> SoftValue {
    let left_val = soft_eval(env.clone(), left);
    let right_val = soft_eval(env.clone(), right);

    match (left_val, right_val) {
        (SoftValue::Float(a), SoftValue::Float(b)) => SoftValue::Float(softgt(a, b)),
        (SoftValue::Int(a), SoftValue::Float(b)) => SoftValue::Float(softgt(a as f64, b)),
        (SoftValue::Float(a), SoftValue::Int(b)) => SoftValue::Float(softgt(a, b as f64)),
        (SoftValue::Int(a), SoftValue::Int(b)) => SoftValue::Int(softgt(a, b as f64)),
        _ => unreachable!(),
    }
}

pub fn softgt(x: f64, c: f64) -> f64 {
    1.0 / (1.0 + (-1.0 * (x - c)).exp())
}
