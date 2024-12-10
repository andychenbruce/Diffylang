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
    fn eval_addition_ints(a: i64, b: i64) -> i64 {
        a + b
    }

    fn eval_addition_floats(a: f64, b: f64) -> f64 {
        a + b
    }

    fn eval_multiplication_int(a: i64, b: i64) -> i64 {
        a * b
    }

    fn eval_multiplication_floats(a: f64, b: f64) -> f64 {
        a * b
    }

    fn eval_negation_int(a: i64) -> i64 {
        -a
    }

    fn eval_negation_float(a: f64) -> f64 {
        -a
    }

    fn eval_equality_ints(a: i64, b: i64) -> bool {
        a == b
    }

    fn eval_equality_floats(a: f64, b: f64) -> bool {
        a == b
    }

    fn eval_not(a: bool) -> bool {
        !a
    }
    fn eval_and(a: bool, b: bool) -> bool {
        a && b
    }

    fn eval_or(a: bool, b: bool) -> bool {
        a || b
    }
    fn eval_less_than_ints(a: i64, b: i64) -> bool {
        a < b
    }

    fn eval_less_than_floats(a: f64, b: f64) -> bool {
        a < b
    }
    fn eval_if(
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

    fn make_range(start: i64, end: i64, _num_ids: usize) -> Vec<i64> {
        (start..end).collect()
    }

    fn eval_index(
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

    fn eval_len(l: Vec<ast::eval::EvalVal<i64, f64, bool, i64>>) -> i64 {
        l.len() as i64
    }

    fn stop_while_eval(cond: bool) -> bool {
        !cond
    }

    fn eval_product_index(
        p: Vec<ast::eval::EvalVal<i64, f64, bool, i64>>,
        i: i64,
    ) -> ast::eval::EvalVal<i64, f64, bool, i64> {
        p[i as usize].clone()
    }
}
