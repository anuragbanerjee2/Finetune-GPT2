from transformers import AutoModelForCausalLM, AutoTokenizer


class Inference:
    def __init__(self,model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def get_answer(self,query, temperature=0.1, max_length=100):
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids

        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=temperature,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text
