from vllm import LLM, SamplingParams


if __name__ == "__main__":
    model_path = "meta-llama/Llama-3.2-3B-Instruct"

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)


    llm = LLM(model=model_path)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")