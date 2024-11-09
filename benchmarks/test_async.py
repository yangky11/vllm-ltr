import ray
import argparse
import asyncio
from loguru import logger

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams

from typing import List, Tuple

async def iterate_over_output_for_one_prompt(output_iterator: AsyncStream) -> str:
    last_text = ""
    prompt = "???"

    async for output in output_iterator:
        ids = output.outputs[0].token_ids
        pred_score = output.outputs[0].pred_score
    return len(ids), pred_score


async def generate(engine: AsyncLLMEngine, prompts: list[str], max_tokens: list[int], scores: list[float] = None) -> list[str]:
    #note that I set ignore_eos to True, so the model will not stop at the end of the sentence
    sampling_params = [SamplingParams(n=1, max_tokens=max_tokens[i], ignore_eos=True) for i in range(len(prompts))]
    output_iterators = [await engine.add_request(f"req_{i}", prompt, sampling_params[i], score=scores[i])
                        for i, prompt in enumerate(prompts)]
    outputs = await asyncio.gather(*[iterate_over_output_for_one_prompt(output_iterator)
                                     for output_iterator in output_iterators])
    return list(outputs)


@ray.remote(num_gpus=8)
class VllmActor:

    def __init__(self) -> None:
        pass

    def initialize(self) -> None:
        logger.info("Initializing vLLM")
        args = AsyncEngineArgs(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            # model="facebook/opt-125m",
            tensor_parallel_size=8,
            max_num_batched_tokens=8192,
            schedule_type = "predscore",
            enable_chunked_prefill=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        logger.info("vLLM initialized")

    async def generate(
        self, req_id: str, prompt: str, num_samples: int
    ) -> List[Tuple[str, float]]:
        sampling_params = SamplingParams(
            n=num_samples,
            temperature=1.0,
            logprobs=0,
        )
        async for oup in self.engine.generate(prompt, sampling_params, req_id, score=512):
            final_output = oup
        return [(x.text, x.cumulative_logprob) for x in final_output.outputs]


async def main():
    actor = VllmActor.remote()
    ray.get(actor.initialize.remote())
    outputs = ray.get(actor.generate.remote("xxxjgoierjioajf", "Hi", 10))
    print(outputs)
    print("FINISHED")

#the script will not exit because of a bug https://github.com/vllm-project/vllm/issues/4789
if __name__ == '__main__':
    asyncio.run(main())
