mod ast;
mod parser;
mod type_checker;
mod interpreter;
mod interpreter_soft;

fn main() {
    let program_parse_tree: parser::ProgramParseTree =
        parsel::parse_str(&std::fs::read_to_string("test.prog").unwrap()).unwrap();

    let program_ast = program_parse_tree.into();

    let val = interpreter::apply_function(
        &program_ast,
        "is_greater_than_2",
        vec![interpreter::Value::Int(3)],
    );

    println!("val = {:?}", val);

    println!("{}", serde_json::to_string_pretty(&program_ast).unwrap());

    // Calling softgt using the module path
    let result1 = interpreter_soft::softgt(3.0, 2.0); // Should be close to 1
    let result2 = interpreter_soft::softgt(1.0, 2.0); // Should be close to 0
    println!("softgt(3.0, 2.0) = {}", result1);
    println!("softgt(1.0, 2.0) = {}", result2);
}
