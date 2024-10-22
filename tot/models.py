import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class Model():
    def __init__(self,
                 model_id, 
                 temperature=1.0,  
                 ) -> None:
        
        custom_cache_dir = ""
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            cache_dir=custom_cache_dir, 
            quantization_config=quantization_config, 
            device_map="auto",
            torch_dtype=torch.bfloat16,)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=custom_cache_dir)
        if 'meta-llama' in model_id:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        self.temperature = temperature
        self.model_id = model_id
        
        
    def inference(self, prompt, max_tokens=3000, num_return_sequences=1, stop=None) -> list:
        if 'meta-llama' in self.model_id:
            messages = [{"role": "user", "content": prompt}]
            model_inputs = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
                ).to(self.model.device)
            while num_return_sequences > 0:
                cnt = min(num_return_sequences, 20)
                num_return_sequences -= cnt
                generated_ids = self.model.generate(
                    model_inputs, 
                    temperature=self.temperature, 
                    max_new_tokens=max_tokens, 
                    num_return_sequences=cnt, 
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.terminators
                    )
        else:
            model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            while num_return_sequences > 0:
                cnt = min(num_return_sequences, 20)
                num_return_sequences -= cnt
                generated_ids = self.model.generate(
                    **model_inputs, 
                    temperature=self.temperature, 
                    max_new_tokens=max_tokens, 
                    num_return_sequences=cnt, 
                    do_sample=True,
                    )
        decoded = self.tokenizer.batch_decode(
            generated_ids[:, len(model_inputs[0]):], 
            skip_special_tokens=True)
        return decoded
