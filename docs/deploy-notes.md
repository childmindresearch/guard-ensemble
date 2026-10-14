# Guard Ensemble Deployment Strategy

## Introduction

This risk-detection pipeline scores text with an ensemble of four safety
classifiers (ShieldGemma 9B, Llama Guard 3 8B, Granite Guardian 3.3 8B,
Qwen3Guard-Gen 8B), as well as  a general-purpose model (Qwen3-4B) that reads
their scores and produces a final verdict given a deliberation prompt.

It's worthwhile to consider two forms of input we may desire for this pipeline
to fulfill, which will ultimately determine the shape of the hosted
infrastructure:

1. Single entries of text that require an immediate answer
1. Batches of entries that require a task to complete eventually

This document presents findings from exploring hosting options on AWS and
outlines the implications different approaches have for the resulting code
implementation.

> **Note:** The prices displayed in this document are LLM-assisted estimates
and may not reflect true operating costs.

## Initial findings

The first attempt at running the risk-detection pipeline employed a
`g4dn.xlarge` EC2 instance, which offers an NVIDIA T4 GPU with 16GB of VRAM.
This is the same instance type the [Redactor][redactor] anonymization pipeline
utilizes to process AWS Batch jobs.

[redactor]: https://github.com/childmindresearch/bucket-drop/tree/main/redactor

GPU memory limits quick revealed themselves as the primary consideration for
hosting multiple models at once. One 8B model requires ~16GB of memory at
16-bit precision, which is reduced to ~9GB of when quantized to FP8, while GGUF
models require \~8GB of memory.

Context size also affects memory usage. Some of the guard models default to
large windows which quickly exceed memory limits; setting the size to a lower
token count is required to mitigate this cost.

Given these limitations, it's only possible to load one model at a time on a
T4. When running the pipeline via `docker model`, each model must be unloaded
after processing completes:

```python
CONTEXT_SIZE = 8192

def _prepare(model_id: str) -> None:
    """Clear the GPU, then set this model's context size (unloading resets it)."""
    subprocess.run(["docker", "model", "unload", "--all"], check=False, capture_output=True)
    subprocess.run(
        ["docker", "model", "configure", "--context-size", str(CONTEXT_SIZE), model_id],
        check=True, capture_output=True,
    )
    print(f"Unloaded models and configured {model_id}")
```

Model loading comes with a hefty performance cost. Each load takes up to a
minute, sometimes more. While this may work for running against batches of text,
it's not possible to send a single text entry (or a small set of them) and be
met a quick response.

### Batch jobs

It's possible to run Guard Ensemble similarly to Redactor: listen to S3 events
and let Batch provision `g4dn.xlarge` instances to handle them. This may not
be advisable, however, as this requires:

- Building and hosting a Docker image with models embedded inside of it. This
  comes with several issues:
  - Building the image via GitHub actions requires a large amount of disk
      space
  - Deploy times may take tens of minutes
  - Downloading the image on each invocation adds minutes to cold-start time
- Running each model sequentially, causing job times to elongate

The first issue may be resolvable with clever use of S3 for syncing a model
cache; however, the second may be cause for concern since, depending on the
frequency of jobs, we may wind up with an always-on instance that is managing
an ever-growing backlog of jobs. If this becomes untenable, scaling up instances
is trivial, but at that point the cost benefit of running Batch begins to
disappear.

## Running as a service

Given the critical nature of this pipeline, Guard Ensemble may do best as a
hosted service. This would eliminate cold-starts and open the door for larger
EC2 instance types which can hold multiple models in memory at once. One option
is a `g6e.xlarge` instance which contain NVIDIA L40S GPUs with 48GB of memory.

A simple services that accepts single requests and batches was run successfully
on one of these instances, with the caveat that only *four* models could fit
onto the card at once. ShieldGemma was omitted for the sake of testing; it may
be worth considering whether we should omit it entirely as it takes up the most
memory and requires multiple calls per text, making it the least efficient model
of the bunch.

This test was done using [vLLM](https://vllm.ai) to run each model instance,
which has benefits over `docker model`, mainly for its inference which batches
multiple requests across the GPU, resulting in higher performance all around.

Below is a memory budget breakdown for different model hosting methods:

| Models | VRAM needed | Fits on |
| --- | --- | --- |
| All five, Q8 GGUF | ~43 GB | 48 GB (L40S) |
| All five, FP8 in vLLM | ~50 GB | does not fit in 48 GB |
| Four (without ShieldGemma), FP8 in vLLM | ~39 GB | 48 GB |
| Small variants (1B-4B guards, 4B deliberator) | ~15 GB | one 24 GB card (L4/A10G) |
| One model at a time, Q8 | ~10 GB | 16 GB (T4) |

### Jobs

Some work would have to be done around processing jobs via a queue (using SQS,
for example) and implementing retries on failure. Jobs also must not block
incoming single requests.

## Other considerations

### Smaller models

Each guard family publishes small variants: Llama Guard 3 1B, Granite Guardian
3.3 2B and Qwen3Guard-Gen 4B. With these, the whole ensemble fits on one 24 GB
card, which reduces the cost associated with hosting a service by a fair amount.

### Bedrock

AWS Bedrock offers per-token pricing with no instances. All data stays in the
account and can be routed over a VPC endpoint.

Custom Model Import hosts uploaded weights and scales to zero after five idle
minutes, billed in five-minute windows while active. This would work well for
batch jobs but quickly become expensive with steady traffic.

Bedrock only accepts certain architectures: Llama is supported, but Gemma,
Granite, and Qwen3 are not. Given this route, one could upload a custom model
that runs the deliberator prompt directly, removing the need for an ensemble
altogether.

## Instance prices

Below are costs associated with EC2 instance types.

| Instance | GPU | VRAM | $/hr | 24/7 month |
| --- | --- | --- | --- | --- |
| g4dn.xlarge | T4 | 16 GB | 0.53 | ~384 |
| g6e.xlarge | L40S | 48 GB | 1.86 | ~1,359 |
