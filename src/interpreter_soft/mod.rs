use crate::ast;

const SIGMOID_VARIANCE: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct SoftValue {
    pub value: ValueType,
    pub gradient: Gradient,
}

#[derive(Clone, Debug)]
pub struct Gradient {
    values: Vec<f64>,
}

impl core::ops::Add for Gradient {
    type Output = Gradient;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.values.len() == rhs.values.len());

        return Gradient {
            values: self
                .values
                .iter()
                .zip(rhs.values.iter())
                .map(|(l, r)| l + r)
                .collect(),
        };
    }
}

impl core::ops::Sub for Gradient {
    type Output = Gradient;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.values.len() == rhs.values.len());

        return Gradient {
            values: self
                .values
                .iter()
                .zip(rhs.values.iter())
                .map(|(l, r)| l - r)
                .collect(),
        };
    }
}

impl core::ops::Mul<f64> for Gradient {
    type Output = Gradient;

    fn mul(self, rhs: f64) -> Self::Output {
        return Gradient {
            values: self.values.iter().map(|x| x * rhs).collect(),
        };
    }
}

#[derive(Clone, Debug)]
pub enum ValueType {
    Int(f64),
    Float(f64),
    Bool(f64),
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
    arguments: Vec<ValueType>,
) -> SoftValue {
    let _type_env: crate::type_checker::TypeEnv =
        crate::type_checker::type_check_program(program).unwrap();

    soft_apply_function(
        SoftEnv {
            program,
            vars: SoftEnvVars::End,
        },
        func_name,
        arguments
            .into_iter()
            .map(|x| SoftValue {
                value: x,
                gradient: Gradient {
                    values: vec![0.0; program.num_ids],
                },
            })
            .collect(),
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
        ast::Expression::Integer(x, id) => SoftValue {
            value: ValueType::Int(*x as f64),
            gradient: make_oneshot(env.program.num_ids, id.0),
        },
        ast::Expression::Str(_, _) => todo!(),
        ast::Expression::Float(x, id) => SoftValue {
            value: ValueType::Float(*x),
            gradient: make_oneshot(env.program.num_ids, id.0),
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
    let left = soft_eval(env.clone(), left);
    let right = soft_eval(env.clone(), right);

    let left_val = get_number_vals(&left);
    let right_val = get_number_vals(&right);

    let new_val = match (left.value, right.value) {
        (ValueType::Int(_), ValueType::Int(_)) => ValueType::Int(left_val + right_val),
        _ => ValueType::Float(left_val + right_val),
    };

    let new_gradient = left.gradient + right.gradient;

    SoftValue {
        value: new_val,
        gradient: new_gradient,
    }
}

fn eval_soft_subtraction(
    env: SoftEnv,
    left: &ast::Expression,
    right: &ast::Expression,
) -> SoftValue {
    let left = soft_eval(env.clone(), left);
    let right = soft_eval(env.clone(), right);

    let left_val = get_number_vals(&left);
    let right_val = get_number_vals(&right);

    let new_val = match (left.value, right.value) {
        (ValueType::Int(_), ValueType::Int(_)) => ValueType::Int(left_val - right_val),
        _ => ValueType::Float(left_val - right_val),
    };

    let new_gradient = left.gradient - right.gradient;

    SoftValue {
        value: new_val,
        gradient: new_gradient,
    }
}

fn eval_soft_greater_than(
    env: SoftEnv,
    left: &ast::Expression,
    right: &ast::Expression,
) -> SoftValue {
    let left = soft_eval(env.clone(), left);
    let right = soft_eval(env.clone(), right);

    let left_val = get_number_vals(&left);
    let right_val = get_number_vals(&right);

    softgt(left_val, right_val, left.gradient, right.gradient)
}

fn eval_soft_multiplication(
    env: SoftEnv,
    left: &ast::Expression,
    right: &ast::Expression,
) -> SoftValue {
    let left = soft_eval(env.clone(), left);
    let right = soft_eval(env.clone(), right);

    let left_val = get_number_vals(&left);
    let right_val = get_number_vals(&right);

    let result_value = match (left.value, right.value) {
        (ValueType::Int(_), ValueType::Int(_)) => ValueType::Int(left_val * right_val),
        _ => ValueType::Float(left_val * right_val),
    };
    let result_gradient = left.gradient * right_val + right.gradient * left_val; //product rule

    SoftValue {
        value: result_value,
        gradient: result_gradient,
    }
}

pub fn sigmoid(u: f64) -> f64 {
    1.0 / (1.0 + (-1.0 * u / SIGMOID_VARIANCE).exp())
}

pub fn sigmoid_gradient(u: f64) -> f64 {
    sigmoid(u) * (1.0 - sigmoid(u))
}

pub fn softgt(x: f64, c: f64, x_grad: Gradient, c_grad: Gradient) -> SoftValue {
    let value = sigmoid(x - c);

    SoftValue {
        gradient: Gradient {
            values: (x_grad - c_grad)
                .values
                .iter()
                .map(|g| g * sigmoid_gradient(x - c))
                .collect(),
        },
        value: ValueType::Bool(value),
    }
}

fn make_oneshot(size: usize, pos: usize) -> Gradient {
    let mut output = vec![0.0; size];
    output[pos] = 1.0;

    Gradient { values: output }
}

fn get_number_vals(val: &SoftValue) -> f64 {
    match val.value {
        ValueType::Int(x) => x,
        ValueType::Float(x) => x,
        _ => unreachable!(),
    }
}
