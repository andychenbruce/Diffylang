mod json_generator;
mod parser;
mod type_checker;
mod interpreter;

fn main() {
    let program_ast: parser::Program =
        parsel::parse_str(&std::fs::read_to_string("test.prog").unwrap()).unwrap();

    let val = interpreter::apply_function(program_ast.clone(), "is_greater_than_2", vec![
        interpreter::Value::Int(2),
        
    ]);

    println!("val = {:?}", val);
    
    let json: json_generator::ProgramJson = program_ast.into();

    println!("{}", serde_json::to_string_pretty(&json).unwrap());
}
