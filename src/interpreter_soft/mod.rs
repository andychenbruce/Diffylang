use crate::ast;

#[derive(Clone, Debug)]
pub struct SoftValue {
    pub value: f64,
    pub gradient: f64,
    pub value_type: ValueType,
}

#[derive(Clone, Debug)]
pub enum ValueType {
    Int,
    Float,
    Bool,
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
                    first.1.clone()
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
        ast::Expression::Integer(x, _) => SoftValue {
            value: *x as f64,
            gradient: 0.0, // Initial gradient
            value_type: ValueType::Int,
        },
        ast::Expression::Str(_, _) => todo!(),
        ast::Expression::Float(x, _) => SoftValue {
            value: *x,
            gradient: 0.0, // Initial gradient
            value_type: ValueType::Float,
        },
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

    // Handle the addition based on the value types
    match (left_val.value_type, right_val.value_type) {
        (ValueType::Int, ValueType::Int) | (ValueType::Float, ValueType::Float) | (ValueType::Int, ValueType::Float) | (ValueType::Float, ValueType::Int) => {
            let result_value = left_val.value + right_val.value;
            let result_gradient = left_val.gradient + right_val.gradient; // Adjust based on your gradient computation logic
            SoftValue {
                value: result_value,
                gradient: result_gradient,
                value_type: ValueType::Float, // Promote to Float if necessary
            }
        },
        _ => panic!("Unsupported types for addition"),
    }
}

fn eval_soft_subtraction(env: SoftEnv, left: &ast::Expression, right: &ast::Expression) -> SoftValue {
    let left_val = soft_eval(env.clone(), left);
    let right_val = soft_eval(env.clone(), right);

    match (left_val.value_type, right_val.value_type) {
        (ValueType::Int, ValueType::Int) | (ValueType::Float, ValueType::Float) | (ValueType::Int, ValueType::Float) | (ValueType::Float, ValueType::Int) => {
            let result_value = left_val.value - right_val.value;
            let result_gradient = left_val.gradient - right_val.gradient; // Adjust based on your gradient computation logic
            SoftValue {
                value: result_value,
                gradient: result_gradient,
                value_type: ValueType::Float,
            }
        },
        _ => panic!("Unsupported types for subtraction"),
    }
}

fn eval_soft_greater_than(env: SoftEnv, left: &ast::Expression, right: &ast::Expression) -> SoftValue {
    let left_val = soft_eval(env.clone(), left);
    let right_val = soft_eval(env.clone(), right);

    let (result_value, result_gradient) = softgt(left_val.value, right_val.value);
    SoftValue {
        value: result_value,
        gradient: result_gradient,
        value_type: ValueType::Bool,
    }
}

fn eval_soft_multiplication(env: SoftEnv, left: &ast::Expression, right: &ast::Expression) -> SoftValue {
    let left_val = soft_eval(env.clone(), left);
    let right_val = soft_eval(env.clone(), right);

    let result_value = left_val.value * right_val.value;
    let result_gradient = left_val.value * right_val.gradient + right_val.value * left_val.gradient; // Product rule (Correct me if I'm wrong)
    SoftValue {
        value: result_value,
        gradient: result_gradient,
        value_type: ValueType::Float,
    }
}

pub fn softgt(x: f64, c: f64) -> (f64, f64) {
    let exp_neg = (-1.0 * (x - c)).exp();
    let value = 1.0 / (1.0 + exp_neg);
    let gradient = exp_neg / ((1.0 + exp_neg).powi(2));
    (value, gradient)
}
