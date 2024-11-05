mod ast;
mod interpreter;
mod interpreter_soft;
mod parser;
mod type_checker;

fn main() {
    let program_parse_tree_result =
        parsel::parse_str(&std::fs::read_to_string("test.prog").unwrap());

    let program_parse_tree: parser::ProgramParseTree = match program_parse_tree_result {
        Ok(x) => x,
        Err(err) => panic!("bruh at {:?}: {:?}", err.span().start(), err),
    };

    let program_ast = ast::make_program(program_parse_tree);

    let val = interpreter::run_function(&program_ast, "test", vec![interpreter::Value::Int(3)]);
    assert!(val == interpreter::Value::Bool(true));

    println!("val = {:?}", val);

    for i in -5..5 {
        let soft_val = interpreter_soft::soft_run_function(
            &program_ast,
            "test",
            vec![interpreter_soft::ValueType::Int(i as f64)],
        );
        println!("i = {}, soft val = {:?}", i, soft_val);
        if let interpreter_soft::ValueType::Bool(x) = soft_val.value {
            if i > 2 {
                assert!(x > 0.5);
            } else if i < 2 {
                assert!(x < 0.5);
            } else {
                assert!((x - 0.5).abs() < 0.01);
            }
        } else {
            panic!();
        }
    }
}
