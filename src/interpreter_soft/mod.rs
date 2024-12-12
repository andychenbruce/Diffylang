use crate::ast;

const SIGMOID_VARIANCE: f64 = 1.0;
const EQUALITY_VARIANCE: f64 = 20.0;
const SIGMA_LIST: f64 = 0.5;
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
        ast::Expression::Product(vals) => {
            for val in vals {
                apply_gradient_expr(val, grad);
            }
        }
        ast::Expression::ProductProject { value, index: _ } => {
            apply_gradient_expr(value, grad);
        }
        ast::Expression::List {
            type_name: _,
            values,
        } => {
            for val in values {
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
        ast::Expression::FoldLoop {
            fold_iter,
            accumulator,
            body,
        } => {
            match fold_iter.as_mut() {
                ast::FoldIter::ExprList(ref mut l) => {
                    apply_gradient_expr(l, grad);
                }
                ast::FoldIter::Range(ref mut start, ref mut end) => {
                    apply_gradient_expr(start, grad);
                    apply_gradient_expr(end, grad);
                }
            }
            apply_gradient_expr(accumulator.1.as_mut(), grad);
            apply_gradient_expr(body, grad);
        }
        ast::Expression::WhileLoop {
            accumulator,
            cond,
            body,
            exit_body,
        } => {
            apply_gradient_expr(accumulator.1.as_mut(), grad);
            apply_gradient_expr(cond.as_mut(), grad);
            apply_gradient_expr(body.as_mut(), grad);
            apply_gradient_expr(exit_body.as_mut(), grad);
        }
        ast::Expression::Variable { ident: _, span: _ } => {}
        ast::Expression::HardInt(_) => {}
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
        println!("comparing {:?} and {:?}", a, b);
        let diff = a.val - b.val;
        let diff_grad = a.gradient - b.gradient;
        let val = (-(diff * diff) / EQUALITY_VARIANCE).exp();
        let val_grad = diff_grad
            * (((-1.0 / EQUALITY_VARIANCE) * (-(diff * diff) / EQUALITY_VARIANCE).exp())
                * (2.0 * diff)); //chain rule

        SoftBool {
            val,
            gradient: val_grad,
        }
    }

    fn eval_equality_floats(a: SoftFloat, b: SoftFloat) -> SoftBool {
        let diff = a.val - b.val;
        let diff_grad = a.gradient - b.gradient;

        let val = (-(diff * diff) / EQUALITY_VARIANCE).exp();
        let val_grad = diff_grad
            * (((1.0 / EQUALITY_VARIANCE) * ((diff * diff) / EQUALITY_VARIANCE).exp())
                * (2.0 * diff)); //chain rule

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

    fn make_range(start: i64, end: i64, num_ids: usize) -> Vec<SoftInt> {
        (start..end)
            .map(|x| make_int(x, ast::LitId(None), num_ids))
            .collect()
    }

    fn eval_index(
        l: Vec<ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64>>,
        i: SoftInt,
    ) -> ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64> {
        let stuff = l
            .iter()
            .enumerate()
            .map(|(p, v)| {
                let bott_val = (p as f64) - i.val - 0.5;
                let top_val = (p as f64) - i.val + 0.5;
                let bottom_cdf = if p == 0 {
                    0.0
                } else {
                    rgsl::randist::gaussian::gaussian_P(bott_val, SIGMA_LIST)
                };
                let top_cdf = if p == (l.len() - 1) {
                    1.0
                } else {
                    rgsl::randist::gaussian::gaussian_P(top_val, SIGMA_LIST)
                };

                let cdf = top_cdf - bottom_cdf;
                let bottom_pdf = if p == 0 {
                    0.0
                } else {
                    rgsl::randist::gaussian::gaussian_pdf(bott_val, SIGMA_LIST)
                };
                let top_pdf = if p == (l.len() - 1) {
                    0.0
                } else {
                    rgsl::randist::gaussian::gaussian_pdf(top_val, SIGMA_LIST)
                };

                let index_grad = i.gradient.clone() * -1.0 * (top_pdf - bottom_pdf);
                match v {
                    ast::eval::EvalVal::Int(x) => {
                        ast::eval::EvalVal::<SoftInt, SoftFloat, SoftBool, i64>::Int(
                            Self::eval_multiplication_int(
                                x.clone(),
                                SoftInt {
                                    val: cdf,
                                    gradient: index_grad,
                                },
                            ),
                        )
                    }
                    ast::eval::EvalVal::Float(_) => todo!(),
                    ast::eval::EvalVal::Bool(_) => todo!(),
                    _ => todo!(),
                }
            })
            .collect::<Vec<_>>();

        match stuff[0].clone() {
            ast::eval::EvalVal::Float(x) => ast::eval::EvalVal::Float(stuff.into_iter().fold(
                SoftFloat {
                    val: 0.0,
                    gradient: make_oneshot(x.gradient.values.len(), crate::ast::LitId(None)),
                },
                |acc, val| {
                    if let ast::eval::EvalVal::Float(other) = val {
                        Self::eval_addition_floats(acc, other)
                    } else {
                        panic!()
                    }
                },
            )),
            ast::eval::EvalVal::Int(x) => ast::eval::EvalVal::Int(stuff.into_iter().fold(
                SoftInt {
                    val: 0.0,
                    gradient: make_oneshot(x.gradient.values.len(), crate::ast::LitId(None)),
                },
                |acc, val| {
                    if let ast::eval::EvalVal::Int(other) = val {
                        Self::eval_addition_ints(acc, other)
                    } else {
                        panic!()
                    }
                },
            )),
            _ => todo!(),
        }
    }

    fn eval_set_index(
        l: Vec<ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64>>,
        i: SoftInt,
        v: ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64>,
    ) -> Vec<ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64>> {
        let stuff = (0..l.len())
            .map(|p| {
                let bott_val = (p as f64) - i.val - 0.5;
                let top_val = (p as f64) - i.val + 0.5;
                let bottom_cdf = if p == 0 {
                    0.0
                } else {
                    rgsl::randist::gaussian::gaussian_P(bott_val, SIGMA_LIST)
                };
                let top_cdf = if p == (l.len() - 1) {
                    1.0
                } else {
                    rgsl::randist::gaussian::gaussian_P(top_val, SIGMA_LIST)
                };

                let cdf = top_cdf - bottom_cdf;
                let bottom_pdf = if p == 0 {
                    0.0
                } else {
                    rgsl::randist::gaussian::gaussian_pdf(bott_val, SIGMA_LIST)
                };
                let top_pdf = if p == (l.len() - 1) {
                    0.0
                } else {
                    rgsl::randist::gaussian::gaussian_pdf(top_val, SIGMA_LIST)
                };

                let index_grad = i.gradient.clone() * -1.0 * (top_pdf - bottom_pdf);

                (cdf, index_grad)
            })
            .collect::<Vec<_>>();

        stuff
            .into_iter()
            .zip(l)
            .map(|((percent, grad), l_v)| {
                Self::eval_if(
                    SoftBool {
                        val: percent,
                        gradient: grad,
                    },
                    v.clone(),
                    l_v,
                )
            })
            .collect()
    }

    fn eval_len(l: Vec<ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64>>) -> i64 {
        l.len() as i64
    }

    fn stop_while_eval(cond: SoftBool) -> bool {
        cond.val < 0.5
    }

    fn eval_product_index(
        p: Vec<ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64>>,
        i: i64,
    ) -> ast::eval::EvalVal<SoftInt, SoftFloat, SoftBool, i64> {
        p[i as usize].clone()
    }
}
