import argparse
import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


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


async def main():
    args = parse_args()
    args.schedule_type = 'predscore'
    args.enable_chunked_prefill = True
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args)) #lower score first serve
    prompts = ["Hi" * 10 for _ in range(1024)]
    max_tokens = [100 for _ in range(1024)]
    scores = [1024 - i for i in range(1024)]
    outputs = await generate(engine, prompts, max_tokens, scores)

    print(outputs)
    print("FINISHED")

#the script will not exit because of a bug https://github.com/vllm-project/vllm/issues/4789
if __name__ == '__main__':
    asyncio.run(main())
