use crate::ast;

const SIGMOID_VARIANCE: f64 = 1.0;

pub type SoftValue = ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64>;

pub const SOFT_AST_INIT: ast::ProgramInitFunctions<SoftInt, SoftFloat, SoftBool, i64> =
    ast::ProgramInitFunctions {
        make_int,
        make_float,
        make_bool,
        make_hard,
    };

pub fn make_int(x: i64, lit: ast::LitId, num_ids: usize) -> SoftInt {
    SoftInt {
        val: x as f64,
        gradient: make_oneshot(num_ids, lit),
    }
}

pub fn make_float(x: f64, lit: ast::LitId, num_ids: usize) -> SoftFloat {
    SoftFloat {
        val: x,
        gradient: make_oneshot(num_ids, lit),
    }
}

pub fn make_bool(x: bool, lit: ast::LitId, num_ids: usize) -> SoftBool {
    SoftBool {
        val: if x { 1.0 } else { 0.0 },
        gradient: make_oneshot(num_ids, lit),
    }
}

pub fn make_hard(x: i64) -> i64 {
    x
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

pub fn sigmoid(u: f64) -> f64 {
    1.0 / (1.0 + (-1.0 * u / SIGMOID_VARIANCE).exp())
}

pub fn sigmoid_gradient(u: f64) -> f64 {
    sigmoid(u) * (1.0 - sigmoid(u))
}

pub fn make_oneshot(size: usize, pos: crate::ast::LitId) -> Gradient {
    let mut output = vec![0.0; size];
    if let Some(pos) = pos.0 {
        output[pos] = 1.0;
    }

    Gradient { values: output }
}

pub fn apply_gradient_program(
    program: &mut ast::Program<SoftInt, SoftFloat, SoftBool, i64>,
    grad: &Gradient,
) {
    for function in program.functions.iter_mut() {
        apply_gradient_expr(&mut function.body, grad);
    }
}

fn apply_gradient_expr(
    expr: &mut ast::Expression<SoftInt, SoftFloat, SoftBool, i64>,
    grad: &Gradient,
) {
    match expr {
        ast::Expression::Integer(val, id) => val.val += grad.values[id.0.unwrap()],
        ast::Expression::Float(val, id) => val.val += grad.values[id.0.unwrap()],
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

pub struct SoftEvaluator {}

impl crate::ast::eval::Evaluator<SoftInt, SoftFloat, SoftBool, i64> for SoftEvaluator {
    fn eval_addition_ints(a: SoftInt, b: SoftInt) -> SoftInt {
        let new_val = a.val + b.val;
        let new_gradient = a.gradient + b.gradient;

        SoftInt {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_addition_floats(a: SoftFloat, b: SoftFloat) -> SoftFloat {
        let new_val = a.val + b.val;
        let new_gradient = a.gradient + b.gradient;

        SoftFloat {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_multiplication_int(a: SoftInt, b: SoftInt) -> SoftInt {
        let new_val = a.val * b.val;
        let new_gradient = (a.gradient * b.val) + (b.gradient * a.val); //product rule

        SoftInt {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_multiplication_floats(a: SoftFloat, b: SoftFloat) -> SoftFloat {
        let new_val = a.val * b.val;
        let new_gradient = (a.gradient * b.val) + (b.gradient * a.val); //product rule

        SoftFloat {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_negation_int(a: SoftInt) -> SoftInt {
        SoftInt {
            val: -1.0 * a.val,
            gradient: a.gradient * -1.0,
        }
    }

    fn eval_negation_float(a: SoftFloat) -> SoftFloat {
        SoftFloat {
            val: -1.0 * a.val,
            gradient: a.gradient * -1.0,
        }
    }

    fn eval_equality_ints(a: SoftInt, b: SoftInt) -> SoftBool {
        let diff = a.val - b.val;
        let diff_grad = a.gradient - b.gradient;

        let val = (-(diff * diff) / 4.0).exp();
        let val_grad = diff_grad * (((1.0 / 4.0) * ((diff * diff) / 4.0).exp()) * (2.0 * diff)); //chain rule

        SoftBool {
            val,
            gradient: val_grad,
        }
    }

    fn eval_equality_floats(a: SoftFloat, b: SoftFloat) -> SoftBool {
        let diff = a.val - b.val;
        let diff_grad = a.gradient - b.gradient;

        let val = (-(diff * diff) / 4.0).exp();
        let val_grad = diff_grad * (((1.0 / 4.0) * ((diff * diff) / 4.0).exp()) * (2.0 * diff)); //chain rule

        SoftBool {
            val,
            gradient: val_grad,
        }
    }

    fn eval_less_than_ints(a: SoftInt, b: SoftInt) -> SoftBool {
        let value = sigmoid(b.val - a.val);

        SoftBool {
            gradient: Gradient {
                values: (b.gradient - a.gradient)
                    .values
                    .iter()
                    .map(|g| g * sigmoid_gradient(b.val - a.val))
                    .collect(),
            },
            val: value,
        }
    }

    fn eval_less_than_floats(a: SoftFloat, b: SoftFloat) -> SoftBool {
        let value = sigmoid(b.val - a.val);

        SoftBool {
            gradient: Gradient {
                values: (b.gradient - a.gradient)
                    .values
                    .iter()
                    .map(|g| g * sigmoid_gradient(b.val - a.val))
                    .collect(),
            },
            val: value,
        }
    }

    fn eval_not(a: SoftBool) -> SoftBool {
        let new_grad = a.gradient * -1.0;
        SoftBool {
            val: 1.0 - a.val,
            gradient: new_grad,
        }
    }

    fn eval_and(a: SoftBool, b: SoftBool) -> SoftBool {
        let new_val = a.val * b.val;
        let new_gradient = (a.gradient * b.val) + (b.gradient * a.val); //product rule

        SoftBool {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_or(a: SoftBool, b: SoftBool) -> SoftBool {
        let anded = Self::eval_and(a.clone(), b.clone());

        let new_val = a.val + b.val - anded.val;
        let new_gradient = a.gradient + b.gradient - anded.gradient;

        SoftBool {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_if(
        cond: SoftBool,
        true_branch: ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64>,
        false_branch: ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64>,
    ) -> ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64> {
        match (true_branch, false_branch) {
            (ast::eval::EvalVal::Int(a), ast::eval::EvalVal::Int(b)) => {
                ast::eval::EvalVal::Int(Self::eval_addition_ints(
                    Self::eval_multiplication_int(
                        a,
                        SoftInt {
                            val: cond.val,
                            gradient: cond.gradient.clone(),
                        },
                    ),
                    Self::eval_multiplication_int(
                        b,
                        SoftInt {
                            val: (1.0 - cond.val),
                            gradient: cond.gradient * -1.0,
                        },
                    ),
                ))
            }
            (ast::eval::EvalVal::Float(a), ast::eval::EvalVal::Float(b)) => {
                ast::eval::EvalVal::Float(Self::eval_addition_floats(
                    Self::eval_multiplication_floats(
                        a,
                        SoftFloat {
                            val: cond.val,
                            gradient: cond.gradient.clone(),
                        },
                    ),
                    Self::eval_multiplication_floats(
                        b,
                        SoftFloat {
                            val: (1.0 - cond.val),
                            gradient: cond.gradient * -1.0,
                        },
                    ),
                ))
            }
            (ast::eval::EvalVal::Bool(a), ast::eval::EvalVal::Bool(b)) => {
                let true_weighted = Self::eval_and(a, cond.clone());
                let false_weighted = Self::eval_and(b, Self::eval_not(cond));

                ast::eval::EvalVal::Bool(SoftBool {
                    val: true_weighted.val + false_weighted.val,
                    gradient: true_weighted.gradient + false_weighted.gradient,
                })
            }
            _ => todo!(),
        }
    }

    fn make_range(_start: SoftInt, _end: SoftInt) -> Vec<SoftInt> {
        todo!()
    }
}
