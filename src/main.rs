mod json_generator;
mod parser;

fn main() {
    let thing: parser::FunctionDefinition =
        parsel::parse_str(&std::fs::read_to_string("test.prog").unwrap()).unwrap();

    let json: json_generator::FunctionDefinitionJson = thing.into();
    
    println!("{}", serde_json::to_string_pretty(&json).unwrap());
}
