import torch
import time
import json
import random
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from .schemas import ModelOutput, ALLOWED_TOOLS

class ModelRunner:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        
    def generate(self, prompt: str, strategy: str) -> ModelOutput:
        tools_json = json.dumps(ALLOWED_TOOLS, indent=2)
        system_prompt = f"""You are an AI assistant processing user requests. 
Analyze the provided history/context and determine the final required action.
You MUST output EXACTLY one JSON object and nothing else. Do not wrap in markdown blocks.

--- ALLOWED TOOLS ---
{tools_json}
---------------------

Format:
{
  "tool": "tool_name",
  "arguments": {
    "...": "..."
  },
  "clarification": null
}

If you need to clarify something or an action is cancelled without a new one, output:
{
  "tool": null,
  "arguments": {},
  "clarification": "Your clarification question or acknowledgement"
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        prompt_tokens = model_inputs.input_ids.shape[1]
        
        start_time = time.time()
        
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None
            )
            
        latency = time.time() - start_time
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        completion_tokens = len(generated_ids[0])
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        parsed_output = {"tool": None, "arguments": {}, "clarification": None}
        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
                
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
                
            parsed = json.loads(clean_text)
            parsed_output.update(parsed)
        except Exception:
            pass

        return ModelOutput(
            tool=parsed_output.get("tool"),
            arguments=parsed_output.get("arguments", {}),
            clarification=parsed_output.get("clarification"),
            raw_output=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            strategy=strategy,
            model_revision="main",
            transformers_version=transformers.__version__,
            random_seed=self.seed
        )
