use crate::ast;

pub const HARD_AST_INIT: ast::ProgramInitFunctions<i64, f64, bool> =
    ast::ProgramInitFunctions {
        make_int,
        make_float,
        make_bool,
    };

fn make_int(x: i64, _: ast::LitId, _: usize) -> i64 {
    x
}

fn make_float(x: f64, _: ast::LitId, _: usize) -> f64 {
    x
}
fn make_bool(x: bool, _: ast::LitId, _: usize) -> bool {
    x
}
fn make_hard(x: i64) -> i64 {
    x
}

pub struct HardEvaluator {}

impl crate::ast::eval::Evaluator<i64, f64, bool> for HardEvaluator {
    fn eval_addition_ints(&self, a: i64, b: i64) -> i64 {
        a + b
    }
    fn eval_addition_floats(&self, a: f64, b: f64) -> f64 {
        a + b
    }

    fn eval_multiplication_int(&self, a: i64, b: i64) -> i64 {
        a * b
    }

    fn eval_multiplication_floats(&self, a: f64, b: f64) -> f64 {
        a * b
    }

    fn eval_negation_int(&self, a: i64) -> i64 {
        -a
    }

    fn eval_negation_float(&self, a: f64) -> f64 {
        -a
    }

    fn eval_equality_ints(&self, a: i64, b: i64) -> bool {
        a == b
    }

    fn eval_equality_floats(&self, a: f64, b: f64) -> bool {
        a == b
    }

    fn eval_not(&self, a: bool) -> bool {
        !a
    }
    fn eval_and(&self, a: bool, b: bool) -> bool {
        a && b
    }

    fn eval_or(&self, a: bool, b: bool) -> bool {
        a || b
    }
    fn eval_less_than_ints(&self, a: i64, b: i64) -> bool {
        a < b
    }

    fn eval_less_than_floats(&self, a: f64, b: f64) -> bool {
        a < b
    }
    fn eval_if(
        &self,
        cond: bool,
        true_branch: ast::eval::EvalVal<i64, f64, bool>,
        false_branch: ast::eval::EvalVal<i64, f64, bool>,
    ) -> ast::eval::EvalVal<i64, f64, bool> {
        if cond {
            true_branch
        } else {
            false_branch
        }
    }

}
