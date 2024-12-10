// use crate::ast;

use crate::interpreter_soft::{SoftBool, SoftFloat, SoftInt};

pub fn harden_ast(
    soft_tree: crate::ast::Program<SoftInt, SoftFloat, SoftBool, i64>,
) -> crate::ast::Program<i64, f64, bool, i64> {
    crate::ast::Program {
        functions: harden_functions(soft_tree.functions),
        test_cases: soft_tree.test_cases.into_iter().map(harden_expr).collect(),
        num_ids: soft_tree.num_ids,
    }
}

fn harden_functions(
    functions: Vec<crate::ast::FunctionDefinition<SoftInt, SoftFloat, SoftBool, i64>>,
) -> Vec<crate::ast::FunctionDefinition<i64, f64, bool, i64>> {
    functions.into_iter().map(harden_function).collect()
}

fn harden_function(
    function: crate::ast::FunctionDefinition<SoftInt, SoftFloat, SoftBool, i64>,
) -> crate::ast::FunctionDefinition<i64, f64, bool, i64> {
    crate::ast::FunctionDefinition {
        name: function.name,
        arguments: function.arguments,
        to_type: function.to_type,
        body: harden_expr(function.body),
    }
}

fn harden_expr(
    expr: crate::ast::Expression<SoftInt, SoftFloat, SoftBool, i64>,
) -> crate::ast::Expression<i64, f64, bool, i64> {
    match expr {
        crate::ast::Expression::Variable { ident, span } => {
            crate::ast::Expression::Variable { ident, span }
        }
        crate::ast::Expression::HardInt(x) => crate::ast::Expression::HardInt(x),
        crate::ast::Expression::Integer(x, n) => {
            crate::ast::Expression::Integer(x.val.round() as i64, n)
        }
        crate::ast::Expression::Float(x, n) => crate::ast::Expression::Float(x.val, n),
        crate::ast::Expression::Str(x, n) => crate::ast::Expression::Str(x, n),
        crate::ast::Expression::Bool(x, n) => crate::ast::Expression::Bool(x.val > 0.5, n),
        crate::ast::Expression::List { type_name, values } => crate::ast::Expression::List {
            type_name,
            values: values.into_iter().map(harden_expr).collect(),
        },
        crate::ast::Expression::FuncApplication {
            func_name,
            args,
            span,
        } => crate::ast::Expression::FuncApplication {
            func_name,
            args: args.into_iter().map(harden_expr).collect(),
            span,
        },
        crate::ast::Expression::ExprWhere { bindings, inner } => {
            crate::ast::Expression::ExprWhere {
                bindings: bindings
                    .into_iter()
                    .map(|binding| crate::ast::LetBind {
                        ident: binding.ident,
                        value: harden_expr(binding.value),
                    })
                    .collect(),
                inner: Box::new(harden_expr(*inner)),
            }
        }
        crate::ast::Expression::IfThenElse {
            boolean,
            true_expr,
            false_expr,
        } => crate::ast::Expression::IfThenElse {
            boolean: Box::new(harden_expr(*boolean)),
            true_expr: Box::new(harden_expr(*true_expr)),
            false_expr: Box::new(harden_expr(*false_expr)),
        },
        crate::ast::Expression::FoldLoop {
            fold_iter,
            accumulator,
            body,
        } => crate::ast::Expression::FoldLoop {
            fold_iter: {
                match *fold_iter {
                    crate::ast::FoldIter::ExprList(l) => {
                        Box::new(crate::ast::FoldIter::ExprList(harden_expr(l)))
                    }
                    crate::ast::FoldIter::Range(start, end) => Box::new(
                        crate::ast::FoldIter::Range(harden_expr(start), harden_expr(end)),
                    ),
                }
            },
            accumulator: (accumulator.0, Box::new(harden_expr(*accumulator.1))),
            body: Box::new(harden_expr(*body)),
        },
        crate::ast::Expression::WhileLoop {
            accumulator,
            cond,
            body,
            exit_body,
        } => crate::ast::Expression::WhileLoop {
            accumulator: (accumulator.0, Box::new(harden_expr(*accumulator.1))),
            cond: Box::new(harden_expr(*cond)),
            body: Box::new(harden_expr(*body)),
            exit_body: Box::new(harden_expr(*exit_body)),
        },
        crate::ast::Expression::Product(values) => {
            crate::ast::Expression::Product(values.into_iter().map(harden_expr).collect())
        }
        crate::ast::Expression::ProductProject { value, index } => {
            crate::ast::Expression::ProductProject {
                value: Box::new(harden_expr(*value)),
                index,
            }
        }
    }
}
