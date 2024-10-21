use crate::type_checker::TypeEnv;

mod json_generator;
mod parser;
mod type_checker;

fn main() {
    let thing: parser::FunctionDefinition =
        parsel::parse_str(&std::fs::read_to_string("test.prog").unwrap()).unwrap();


    type_checker::type_check_func(TypeEnv::empty(), &thing).unwrap();
    let json: json_generator::FunctionDefinitionJson = thing.into();

    println!("{}", serde_json::to_string_pretty(&json).unwrap());
}