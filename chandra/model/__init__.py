from typing import List

from chandra.model.hf import load_model, generate_hf
from chandra.model.schema import BatchInputItem, BatchOutputItem
from chandra.model.vllm import generate_vllm
from chandra.output import parse_markdown, parse_html, parse_chunks, extract_images


class InferenceManager:
    def __init__(self, method: str = "vllm"):
        assert method in ("vllm", "hf"), "method must be 'vllm' or 'hf'"
        self.method = method

        if method == "hf":
            self.model = load_model()
        else:
            self.model = None

    def generate(
        self, batch: List[BatchInputItem], max_output_tokens=None, **kwargs
    ) -> List[BatchOutputItem]:
        output_kwargs = {}
        if "include_images" in kwargs:
            output_kwargs["include_images"] = kwargs.pop("include_images")
        if "include_headers_footers" in kwargs:
            output_kwargs["include_headers_footers"] = kwargs.pop(
                "include_headers_footers"
            )

        if self.method == "vllm":
            results = generate_vllm(
                batch, max_output_tokens=max_output_tokens, **kwargs
            )
        else:
            results = generate_hf(
                batch, self.model, max_output_tokens=max_output_tokens, **kwargs
            )

        output = []
        for result, input_item in zip(results, batch):
            chunks = parse_chunks(result.raw, input_item.image)
            output.append(
                BatchOutputItem(
                    markdown=parse_markdown(result.raw, **output_kwargs),
                    html=parse_html(result.raw, **output_kwargs),
                    chunks=chunks,
                    raw=result.raw,
                    page_box=[0, 0, input_item.image.width, input_item.image.height],
                    token_count=result.token_count,
                    images=extract_images(result.raw, chunks, input_item.image),
                    error=result.error,
                )
            )
        return output
