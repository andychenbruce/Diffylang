// use crate::ast;

use crate::interpreter_soft::{SoftBool, SoftFloat, SoftInt};

pub fn harden_ast(
    soft_tree: crate::ast::Program<SoftInt, SoftFloat, SoftBool>,
) -> crate::ast::Program<i64, f64, bool> {
    crate::ast::Program {
        functions: harden_functions(soft_tree.functions),
        test_cases: soft_tree.test_cases.into_iter().map(harden_expr).collect(),
        gadts: harden_gadts(soft_tree.gadts),
        num_ids: soft_tree.num_ids,
    }
}

fn harden_functions(
    functions: Vec<crate::ast::FunctionDefinition<SoftInt, SoftFloat, SoftBool>>,
) -> Vec<crate::ast::FunctionDefinition<i64, f64, bool>> {
    functions.into_iter().map(harden_function).collect()
}

fn harden_gadts(
    functions: Vec<crate::ast::GadtDefinition<SoftInt, SoftFloat, SoftBool>>,
) -> Vec<crate::ast::GadtDefinition<i64, f64, bool>> {
    functions.into_iter().map(harden_gadt).collect()
}

fn harden_gadt(
    gadt: crate::ast::GadtDefinition<SoftInt, SoftFloat, SoftBool>,
) -> crate::ast::GadtDefinition<i64, f64, bool> {
    crate::ast::GadtDefinition {
        name: gadt.name,
        universe: gadt.universe,
        arguments: gadt
            .arguments
            .into_iter()
            .map(|(ident, expr)| (ident, harden_expr(expr)))
            .collect(),
        constructors: gadt
            .constructors
            .into_iter()
            .map(|(ident, expr)| (ident, harden_expr(expr)))
            .collect(),
    }
}

fn harden_function(
    function: crate::ast::FunctionDefinition<SoftInt, SoftFloat, SoftBool>,
) -> crate::ast::FunctionDefinition<i64, f64, bool> {
    crate::ast::FunctionDefinition {
        name: function.name,
        universe: function.universe,
        arguments: function
            .arguments
            .into_iter()
            .map(|(ident, expr)| (ident, harden_expr(expr)))
            .collect(),
        to_type: harden_expr(function.to_type),
        body: harden_expr(function.body),
    }
}

fn harden_expr(
    expr: crate::ast::Expression<SoftInt, SoftFloat, SoftBool>,
) -> crate::ast::Expression<i64, f64, bool> {
    match expr {
        crate::ast::Expression::Variable { ident } => crate::ast::Expression::Variable { ident },
        crate::ast::Expression::Integer(x, n) => {
            crate::ast::Expression::Integer(x.val.round() as i64, n)
        }
        crate::ast::Expression::Float(x, n) => crate::ast::Expression::Float(x.val, n),
        crate::ast::Expression::Bool(x, n) => crate::ast::Expression::Bool(x.val > 0.5, n),
        crate::ast::Expression::FuncApplication { func_name, args } => {
            crate::ast::Expression::FuncApplication {
                func_name,
                args: args.into_iter().map(harden_expr).collect(),
            }
        }
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
        crate::ast::Expression::Product(values) => {
            crate::ast::Expression::Product(values.into_iter().map(harden_expr).collect())
        }
        crate::ast::Expression::Universe(x) => crate::ast::Expression::Universe(x),
    }
}
