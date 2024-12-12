use crate::ast;

pub const HARD_AST_INIT: ast::ProgramInitFunctions<i64, f64, bool, i64> =
    ast::ProgramInitFunctions {
        make_int,
        make_float,
        make_bool,
        make_hard,
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

impl crate::ast::eval::Evaluator<i64, f64, bool, i64> for HardEvaluator {
    fn eval_addition_ints(&self, a: i64, b: i64) -> i64 {
        a + b
    }
    fn eval_addition_hards(&self, a: i64, b: i64) -> i64 {
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
    fn eval_negation_hard(&self, a: i64) -> i64 {
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
        true_branch: ast::eval::EvalVal<i64, f64, bool, i64>,
        false_branch: ast::eval::EvalVal<i64, f64, bool, i64>,
    ) -> ast::eval::EvalVal<i64, f64, bool, i64> {
        if cond {
            true_branch
        } else {
            false_branch
        }
    }

    fn make_range(&self, start: i64, end: i64, _num_ids: usize) -> Vec<i64> {
        (start..end).collect()
    }

    fn eval_index(
        &self, 
        l: Vec<ast::eval::EvalVal<i64, f64, bool, i64>>,
        i: i64,
    ) -> ast::eval::EvalVal<i64, f64, bool, i64> {
        if i < 0 {
            l[0].clone()
        } else if i as usize >= l.len() - 1 {
            l.last().unwrap().clone()
        } else {
            l[i as usize].clone()
        }
    }

    fn eval_set_index(
        &self, 
        mut l: Vec<ast::eval::EvalVal<i64, f64, bool, i64>>,
        i: i64,
        v: ast::eval::EvalVal<i64, f64, bool, i64>,
    ) -> Vec<ast::eval::EvalVal<i64, f64, bool, i64>> {
        if i < 0 {
            l[0] = v;
        } else if i as usize >= l.len() - 1 {
            *l.last_mut().unwrap() = v;
        } else {
            l[i as usize] = v
        }
        l
    }

    fn eval_len(&self, l: Vec<ast::eval::EvalVal<i64, f64, bool, i64>>) -> i64 {
        l.len() as i64
    }

    fn stop_while_eval(&self, cond: bool) -> bool {
        !cond
    }

    fn eval_product_index(
        &self, 
        p: Vec<ast::eval::EvalVal<i64, f64, bool, i64>>,
        i: i64,
    ) -> ast::eval::EvalVal<i64, f64, bool, i64> {
        p[i as usize].clone()
    }
}
