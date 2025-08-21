from vllm import LLM, SamplingParams
import torch.distributed as dist

def generate(prompts, model_path, sampling_params, prompt_template="{prompt}"):
    """Generate synthetic text data using vLLM"""
    llm = LLM(model=model_path,seed=42)
    
    prompts = [prompt_template.format(prompt=prompt) for prompt in prompts]
    outputs = llm.generate(prompts, sampling_params)
    
    generated_texts = [output.outputs[0].text for output in outputs]
    return generated_texts

def main():
    prompts = [""] * 100
    model_path = "../../test-stuff-scale-2/base_model/epoch-0"
    sampling_params = {"temperature": 1.0}

    generated_texts = generate(
            prompts=prompts,
            model_path=model_path,
            sampling_params=SamplingParams(**sampling_params)
        )
    
    print("GENERATED TEXTS:")
    print(generated_texts)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()