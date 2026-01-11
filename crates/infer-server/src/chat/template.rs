use anyhow::Result;

pub trait ChatTemplate: Send + Sync {
    fn apply(&self, messages: &[crate::api::openai::ChatMessage]) -> Result<String>;
}

pub struct Llama3Template;

impl ChatTemplate for Llama3Template {
    fn apply(&self, messages: &[crate::api::openai::ChatMessage]) -> Result<String> {
        let mut prompt = String::from("<|begin_of_text|>");

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|eot_id|>");
                }
                "user" => {
                    prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|eot_id|>");
                }
                "assistant" => {
                    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|eot_id|>");
                }
                _ => {}
            }
        }

        // Add assistant header for generation
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

        Ok(prompt)
    }
}

pub fn get_template(_model_name: &str) -> Box<dyn ChatTemplate + Send + Sync> {
    // Future: detect from config.json
    Box::new(Llama3Template)
}
