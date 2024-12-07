use crate::ast;

const SIGMOID_VARIANCE: f64 = 1.0;

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
pub enum SoftValue {
    Int(SoftInt),
    Float(SoftFloat),
    Bool(SoftBool),
    List(SoftList),
}

#[derive(Clone, Debug)]
pub struct SoftInt {
    pub val: f64,
    pub gradient: Gradient,
}

#[derive(Clone, Debug)]
pub struct SoftFloat {
    pub val: f64,
    pub gradient: Gradient,
}

#[derive(Clone, Debug)]
pub struct SoftBool {
    pub val: f64,
    pub gradient: Gradient,
}

#[derive(Clone, Debug)]
pub struct SoftList {
    pub vals: Vec<SoftValue>,
}

#[derive(Clone)]
struct SoftEnv<'a> {
    program: &'a ast::Program,
    vars: SoftEnvVars,
}

impl ast::Program {
    pub fn make_int(&self, x: f64) -> SoftValue {
        SoftValue::Int(SoftInt {
            val: x,
            gradient: Gradient {
                values: vec![0.0; self.num_ids],
            },
        })
    }
    pub fn make_float(&self, x: f64) -> SoftValue {
        SoftValue::Float(SoftFloat {
            val: x,
            gradient: Gradient {
                values: vec![0.0; self.num_ids],
            },
        })
    }
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

pub fn soft_eval_test_cases(program: &ast::Program) -> Vec<(f64, Gradient)> {
    program
        .test_cases
        .iter()
        .map(|test_case| {
            soft_eval(
                SoftEnv {
                    program,
                    vars: SoftEnvVars::End,
                },
                test_case,
            )
        })
        .map(|x| {
            if let SoftValue::Bool(v) = x {
                (v.val, v.gradient)
            } else {
                panic!()
            }
        })
        .collect()
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
        ast::Expression::Integer(x, id) => SoftValue::Int(SoftInt {
            val: *x as f64,
            gradient: make_oneshot(env.program.num_ids, *id),
        }),
        ast::Expression::Str(_, _) => todo!(),
        ast::Expression::Float(x, id) => SoftValue::Float(SoftFloat {
            val: *x,
            gradient: make_oneshot(env.program.num_ids, *id),
        }),
        ast::Expression::Bool(_, _) => todo!(),
        ast::Expression::FuncApplication {
            func_name,
            args,
            span: _,
        } => match func_name.0.as_str() {
            "__add" => soft_addition(
                soft_eval(env.clone(), &args[0]),
                soft_eval(env.clone(), &args[1]),
            ),
            "__sub" => soft_subtraction(
                soft_eval(env.clone(), &args[0]),
                soft_eval(env.clone(), &args[1]),
            ),
            "__mul" => soft_multiplication(
                soft_eval(env.clone(), &args[0]),
                soft_eval(env.clone(), &args[1]),
            ),
            "__div" => todo!(),
            "__eq" => todo!(),
            "__gt" => soft_greater_than(
                soft_eval(env.clone(), &args[0]),
                soft_eval(env.clone(), &args[1]),
            ),
            "__lt" => todo!(),
            "__and" => todo!(),
            "__or" => todo!(),
            "__not" => soft_not(soft_eval(env.clone(), &args[0])),
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
        ast::Expression::IfThenElse {
            boolean,
            true_expr,
            false_expr,
        } => {
            let boolean_eval = soft_eval(env.clone(), boolean);
            if let SoftValue::Bool(boolean_val) = boolean_eval {
                let true_eval = soft_eval(env.clone(), true_expr);
                let false_eval = soft_eval(env.clone(), false_expr);

                soft_addition(
                    soft_multiplication(
                        SoftValue::Float(SoftFloat {
                            gradient: boolean_val.gradient.clone(),
                            val: boolean_val.val,
                        }),
                        true_eval,
                    ),
                    soft_multiplication(
                        SoftValue::Float(SoftFloat {
                            gradient: boolean_val.gradient.clone() * -1.0,
                            val: 1.0 - boolean_val.val,
                        }),
                        false_eval,
                    ),
                )
            } else {
                panic!()
            }
        }
        ast::Expression::FoldLoop {
            accumulator,
            body,
            fold_iter,
        } => {
            let iter = match **fold_iter {
                ast::FoldIter::Range(ref start, ref end) => {
                    let start = match soft_eval(env.clone(), &start) {
                        SoftValue::Int(x) => x.val,
                        _ => unreachable!(),
                    }
                    .floor() as usize;

                    let end = match soft_eval(env.clone(), &end) {
                        SoftValue::Int(x) => x.val,
                        _ => unreachable!(),
                    }
                    .floor() as usize;

                    (start..end)
                        .map(|x| env.program.make_int(x as f64))
                        .collect()
                }
                ast::FoldIter::ExprList(ref list) => match soft_eval(env.clone(), &list) {
                    SoftValue::List(l) => l.vals,
                    _ => unreachable!(),
                },
            };

            iter.into_iter()
                .fold(soft_eval(env.clone(), &accumulator.1), |acc, _| {
                    let new_vars = SoftEnvVars::Rest {
                        first: (accumulator.0.clone(), acc),
                        rest: Box::new(env.vars.clone()),
                    };
                    let new_env = SoftEnv {
                        program: env.program,
                        vars: new_vars,
                    };

                    soft_eval(new_env, body)
                })
        }
        ast::Expression::List {
            type_name: _,
            values,
        } => SoftValue::List(SoftList {
            vals: values
                .into_iter()
                .map(|x| soft_eval(env.clone(), x))
                .collect(),
        }),
    }
}

fn soft_addition(left: SoftValue, right: SoftValue) -> SoftValue {
    let (left_val, left_grad) = get_number_vals(&left);
    let (right_val, right_grad) = get_number_vals(&right);

    let new_val = left_val + right_val;
    let new_gradient = left_grad + right_grad;

    match (left, right) {
        (SoftValue::Int(_), SoftValue::Int(_)) => SoftValue::Int(SoftInt {
            val: new_val,
            gradient: new_gradient,
        }),
        _ => SoftValue::Float(SoftFloat {
            val: new_val,
            gradient: new_gradient,
        }),
    }
}

fn soft_subtraction(left: SoftValue, right: SoftValue) -> SoftValue {
    let (left_val, left_grad) = get_number_vals(&left);
    let (right_val, right_grad) = get_number_vals(&right);

    let new_val = left_val + right_val;
    let new_gradient = left_grad + right_grad;

    match (left, right) {
        (SoftValue::Int(_), SoftValue::Int(_)) => SoftValue::Int(SoftInt {
            val: new_val,
            gradient: new_gradient,
        }),
        _ => SoftValue::Float(SoftFloat {
            val: new_val,
            gradient: new_gradient,
        }),
    }
}

fn soft_greater_than(left: SoftValue, right: SoftValue) -> SoftValue {
    let (left_val, left_grad) = get_number_vals(&left);
    let (right_val, right_grad) = get_number_vals(&right);

    softgt(left_val, right_val, left_grad, right_grad)
}

fn soft_multiplication(left: SoftValue, right: SoftValue) -> SoftValue {
    let (left_val, left_grad) = get_number_vals(&left);
    let (right_val, right_grad) = get_number_vals(&right);

    let new_val = left_val * right_val;
    let new_gradient = left_grad * right_val + right_grad * left_val; //product rule

    match (left, right) {
        (SoftValue::Int(_), SoftValue::Int(_)) => SoftValue::Int(SoftInt {
            val: new_val,
            gradient: new_gradient,
        }),
        _ => SoftValue::Float(SoftFloat {
            val: new_val,
            gradient: new_gradient,
        }),
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

    SoftValue::Bool(SoftBool {
        gradient: Gradient {
            values: (x_grad - c_grad)
                .values
                .iter()
                .map(|g| g * sigmoid_gradient(x - c))
                .collect(),
        },
        val: value,
    })
}

pub fn make_oneshot(size: usize, pos: crate::ast::LitId) -> Gradient {
    let mut output = vec![0.0; size];
    if let Some(pos) = pos.0 {
        output[pos] = 1.0;
    }

    Gradient { values: output }
}

fn get_number_vals(val: &SoftValue) -> (f64, Gradient) {
    match val {
        SoftValue::Int(x) => (x.val, x.gradient.clone()),
        SoftValue::Float(x) => (x.val, x.gradient.clone()),
        _ => unreachable!(),
    }
}

pub fn apply_gradient_program(program: &mut ast::Program, grad: &Gradient) {
    for function in program.functions.iter_mut() {
        apply_gradient_expr(&mut function.body, grad);
    }
}

fn apply_gradient_expr(expr: &mut ast::Expression, grad: &Gradient) {
    match expr {
        ast::Expression::Integer(val, id) => {
            *expr = ast::Expression::Float((*val as f64) + grad.values[id.0.unwrap()], *id)
        }
        ast::Expression::Float(val, id) => *val += grad.values[id.0.unwrap()],
        ast::Expression::Str(_, _) => todo!(),
        ast::Expression::Bool(_, _) => todo!(),
        ast::Expression::FuncApplication {
            func_name: _,
            args,
            span: _,
        } => {
            for arg in args {
                apply_gradient_expr(arg, grad);
            }
        }
        ast::Expression::ExprWhere { bindings, inner } => {
            for binding in bindings {
                apply_gradient_expr(&mut binding.value, grad);
            }

            apply_gradient_expr(inner, grad);
        }
        _ => {}
    }
}

fn soft_not(val: SoftValue) -> SoftValue {
    match val {
        SoftValue::Bool(a) => {
            let new_grad = a.gradient * -1.0;

            SoftValue::Bool(SoftBool {
                val: 1.0 - a.val,
                gradient: new_grad,
            })
        }
        _ => unreachable!(),
    }
}
