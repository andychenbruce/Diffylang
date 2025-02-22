// use crate::ast;

use crate::interpreter_soft::{SoftBool, SoftFloat, SoftInt};

pub fn harden_ast(
    soft_tree: crate::ast::Program<SoftInt, SoftFloat, SoftBool>,
) -> crate::ast::Program<i64, f64, bool> {
    crate::ast::Program {
        global_bindings: harden_globals(soft_tree.global_bindings),
        test_cases: soft_tree.test_cases.into_iter().map(harden_expr).collect(),
        gadts: harden_gadts(soft_tree.gadts),
        num_ids: soft_tree.num_ids,
    }
}

fn harden_globals(
    functions: Vec<crate::ast::Binding<SoftInt, SoftFloat, SoftBool>>,
) -> Vec<crate::ast::Binding<i64, f64, bool>> {
    functions.into_iter().map(harden_global).collect()
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
            .map(|arg| crate::ast::Argument {
                name: arg.name,
                arg_type: harden_expr(arg.arg_type),
            })
            .collect(),
        constructors: gadt
            .constructors
            .into_iter()
            .map(|arg| crate::ast::Argument {
                name: arg.name,
                arg_type: harden_expr(arg.arg_type),
            })
            .collect(),
    }
}

fn harden_global(
    binding: crate::ast::Binding<SoftInt, SoftFloat, SoftBool>,
) -> crate::ast::Binding<i64, f64, bool> {
    crate::ast::Binding {
        name: binding.name,
        elem_type: harden_expr(binding.elem_type),
        value: match binding.value {
            crate::ast::Definition::Instrinsic => crate::ast::Definition::Instrinsic,
            crate::ast::Definition::Evaluatable(expression) => {
                crate::ast::Definition::Evaluatable(harden_expr(expression))
            }
        },
    }
}

fn harden_argument(
    arg: crate::ast::Argument<SoftInt, SoftFloat, SoftBool>,
) -> crate::ast::Argument<i64, f64, bool> {
    crate::ast::Argument {
        name: arg.name,
        arg_type: harden_expr(arg.arg_type),
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
        crate::ast::Expression::FuncApplicationMultipleArgs { func, args } => {
            crate::ast::Expression::FuncApplicationMultipleArgs {
                func: Box::new(harden_expr(*func)),
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
        crate::ast::Expression::DependentProductType {
            type_first,
            type_second,
        } => crate::ast::Expression::DependentProductType {
            type_first: Box::new(harden_argument(*type_first)),
            type_second: Box::new(harden_expr(*type_second)),
        },
        crate::ast::Expression::Universe(x) => crate::ast::Expression::Universe(x),
        crate::ast::Expression::DependentFunctionType { type_from, type_to } => {
            crate::ast::Expression::DependentFunctionType {
                type_from: Box::new(harden_argument(*type_from)),
                type_to: Box::new(harden_expr(*type_to)),
            }
        }
        crate::ast::Expression::Lambda { input, body } => crate::ast::Expression::Lambda {
            input,
            body: Box::new(harden_expr(*body)),
        },
    }
}
