use crate::ast;
// use statrs::distribution::Continuous;
// use statrs::distribution::ContinuousCDF;

pub type SoftValue = ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool>;

pub const SOFT_AST_INIT: ast::ProgramInitFunctions<SoftInt, SoftFloat, SoftBool> =
    ast::ProgramInitFunctions {
        make_int,
        make_float,
        make_bool,
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

#[derive(Clone, Debug, serde::Serialize)]
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

#[derive(Clone, Debug, serde::Serialize)]
pub struct SoftInt {
    pub val: f64,
    pub gradient: Gradient,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct SoftFloat {
    pub val: f64,
    pub gradient: Gradient,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct SoftBool {
    pub val: f64,
    pub gradient: Gradient,
}

pub fn sigmoid(u: f64, variance: f64) -> f64 {
    1.0 / (1.0 + (-1.0 * u / variance).exp())
}

pub fn sigmoid_gradient(u: f64, variance: f64) -> f64 {
    sigmoid(u, variance) * (1.0 - sigmoid(u, variance))
}

pub fn make_oneshot(size: usize, pos: crate::ast::LitId) -> Gradient {
    let mut output = vec![0.0; size];
    if let Some(pos) = pos.0 {
        output[pos] = 1.0;
    }

    Gradient { values: output }
}

pub fn apply_gradient_program(
    program: &mut ast::Program<SoftInt, SoftFloat, SoftBool>,
    grad: &Gradient,
) {
    for function in program.functions.iter_mut() {
        apply_gradient_expr(&mut function.body, grad);
    }
}

fn apply_gradient_expr(expr: &mut ast::Expression<SoftInt, SoftFloat, SoftBool>, grad: &Gradient) {
    match expr {
        ast::Expression::Integer(val, id) => val.val += grad.values[id.0.unwrap()],
        ast::Expression::Float(val, id) => val.val += grad.values[id.0.unwrap()],
        ast::Expression::Bool(val, id) => val.val += grad.values[id.0.unwrap()],
        ast::Expression::FuncApplication { func_name: _, args } => {
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
        ast::Expression::Product(vals) => {
            for val in vals {
                apply_gradient_expr(val, grad);
            }
        }
        ast::Expression::IfThenElse {
            boolean,
            true_expr,
            false_expr,
        } => {
            apply_gradient_expr(boolean, grad);
            apply_gradient_expr(true_expr, grad);
            apply_gradient_expr(false_expr, grad);
        }
        ast::Expression::Variable { ident: _ } => {}
        ast::Expression::Universe(_) => {}
    }
}

pub struct SoftEvaluator {
    pub sigmoid_variance: f64,
    pub equality_variance: f64,
    pub sigma_list: f64,
}

impl crate::ast::eval::Evaluator<SoftInt, SoftFloat, SoftBool> for SoftEvaluator {
    fn eval_addition_ints(&self, a: SoftInt, b: SoftInt) -> SoftInt {
        let new_val = a.val + b.val;
        let new_gradient = a.gradient + b.gradient;

        SoftInt {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_addition_floats(&self, a: SoftFloat, b: SoftFloat) -> SoftFloat {
        let new_val = a.val + b.val;
        let new_gradient = a.gradient + b.gradient;

        SoftFloat {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_multiplication_int(&self, a: SoftInt, b: SoftInt) -> SoftInt {
        let new_val = a.val * b.val;
        let new_gradient = (a.gradient * b.val) + (b.gradient * a.val); //product rule

        SoftInt {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_multiplication_floats(&self, a: SoftFloat, b: SoftFloat) -> SoftFloat {
        let new_val = a.val * b.val;
        let new_gradient = (a.gradient * b.val) + (b.gradient * a.val); //product rule

        SoftFloat {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_negation_int(&self, a: SoftInt) -> SoftInt {
        SoftInt {
            val: -1.0 * a.val,
            gradient: a.gradient * -1.0,
        }
    }

    fn eval_negation_float(&self, a: SoftFloat) -> SoftFloat {
        SoftFloat {
            val: -1.0 * a.val,
            gradient: a.gradient * -1.0,
        }
    }

    fn eval_equality_ints(&self, a: SoftInt, b: SoftInt) -> SoftBool {
        let diff = a.val - b.val;
        let diff_grad = a.gradient - b.gradient;
        let val = (-(diff * diff) / self.equality_variance).exp();
        let val_grad = diff_grad
            * (((-1.0 / self.equality_variance) * (-(diff * diff) / self.equality_variance).exp())
                * (2.0 * diff)); //chain rule

        SoftBool {
            val,
            gradient: val_grad,
        }
    }

    fn eval_equality_floats(&self, a: SoftFloat, b: SoftFloat) -> SoftBool {
        let diff = a.val - b.val;
        let diff_grad = a.gradient - b.gradient;

        let val = (-(diff * diff) / self.equality_variance).exp();
        let val_grad = diff_grad
            * (((1.0 / self.equality_variance) * ((diff * diff) / self.equality_variance).exp())
                * (2.0 * diff)); //chain rule

        SoftBool {
            val,
            gradient: val_grad,
        }
    }

    fn eval_less_than_ints(&self, a: SoftInt, b: SoftInt) -> SoftBool {
        let value = sigmoid(b.val - a.val, self.sigmoid_variance);

        SoftBool {
            gradient: Gradient {
                values: (b.gradient - a.gradient)
                    .values
                    .iter()
                    .map(|g| g * sigmoid_gradient(b.val - a.val, self.sigmoid_variance))
                    .collect(),
            },
            val: value,
        }
    }

    fn eval_less_than_floats(&self, a: SoftFloat, b: SoftFloat) -> SoftBool {
        let value = sigmoid(b.val - a.val, self.sigmoid_variance);

        SoftBool {
            gradient: Gradient {
                values: (b.gradient - a.gradient)
                    .values
                    .iter()
                    .map(|g| g * sigmoid_gradient(b.val - a.val, self.sigmoid_variance))
                    .collect(),
            },
            val: value,
        }
    }

    fn eval_not(&self, a: SoftBool) -> SoftBool {
        let new_grad = a.gradient * -1.0;
        SoftBool {
            val: 1.0 - a.val,
            gradient: new_grad,
        }
    }

    fn eval_and(&self, a: SoftBool, b: SoftBool) -> SoftBool {
        let new_val = a.val * b.val;
        let new_gradient = (a.gradient * b.val) + (b.gradient * a.val); //product rule

        SoftBool {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_or(&self, a: SoftBool, b: SoftBool) -> SoftBool {
        let anded = self.eval_and(a.clone(), b.clone());

        let new_val = a.val + b.val - anded.val;
        let new_gradient = a.gradient + b.gradient - anded.gradient;

        SoftBool {
            val: new_val,
            gradient: new_gradient,
        }
    }

    fn eval_if(
        &self,
        cond: SoftBool,
        true_branch: ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool>,
        false_branch: ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool>,
    ) -> ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool> {
        match (true_branch, false_branch) {
            (ast::eval::EvalVal::Int(a), ast::eval::EvalVal::Int(b)) => {
                ast::eval::EvalVal::Int(self.eval_addition_ints(
                    self.eval_multiplication_int(
                        a,
                        SoftInt {
                            val: cond.val,
                            gradient: cond.gradient.clone(),
                        },
                    ),
                    self.eval_multiplication_int(
                        b,
                        SoftInt {
                            val: (1.0 - cond.val),
                            gradient: cond.gradient * -1.0,
                        },
                    ),
                ))
            }
            (ast::eval::EvalVal::Float(a), ast::eval::EvalVal::Float(b)) => {
                ast::eval::EvalVal::Float(self.eval_addition_floats(
                    self.eval_multiplication_floats(
                        a,
                        SoftFloat {
                            val: cond.val,
                            gradient: cond.gradient.clone(),
                        },
                    ),
                    self.eval_multiplication_floats(
                        b,
                        SoftFloat {
                            val: (1.0 - cond.val),
                            gradient: cond.gradient * -1.0,
                        },
                    ),
                ))
            }
            (ast::eval::EvalVal::Bool(a), ast::eval::EvalVal::Bool(b)) => {
                let true_weighted = self.eval_and(a, cond.clone());
                let false_weighted = self.eval_and(b, self.eval_not(cond));

                ast::eval::EvalVal::Bool(SoftBool {
                    val: true_weighted.val + false_weighted.val,
                    gradient: true_weighted.gradient + false_weighted.gradient,
                })
            }
            (ast::eval::EvalVal::Product(a), ast::eval::EvalVal::Product(b)) => {
                assert!(a.len() == b.len());

                ast::eval::EvalVal::Product(
                    a.into_iter()
                        .zip(b)
                        .map(|(a_val, b_val)| self.eval_if(cond.clone(), a_val, b_val))
                        .collect(),
                )
            }
            _ => todo!(),
        }
    }
}
