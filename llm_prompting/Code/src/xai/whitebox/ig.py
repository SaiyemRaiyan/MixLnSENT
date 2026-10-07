"""Optional CUDA 4-bit Llama Integrated Gradients implementation."""

from importlib.util import find_spec

from src.prompting.prompts import build_sentiment_prompt

from ..config import PRIMARY_PROMPT
from ..tokenize import tokenize

LABELS = ("Positive", "Negative", "Neutral", "Mixed")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def availability() -> dict:
    missing = [
        package for package in ("torch", "transformers", "captum", "bitsandbytes")
        if find_spec(package) is None
    ]
    if missing:
        return {"status": "skipped", "reason": f"missing optional packages: {', '.join(missing)}"}
    import torch

    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA GPU is unavailable"}
    return {"status": "ready", "reason": None}


class LlamaIntegratedGradients:
    def __init__(self, model_id: str = MODEL_ID, steps: int = 50):
        status = availability()
        if status["status"] != "ready":
            raise RuntimeError(f"white-box arm skipped: {status['reason']}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from captum.attr import IntegratedGradients

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            raise ValueError("the tokenizer has no pad token for the preregistered IG baseline")
        quantization = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization,
        )
        self.model.eval()
        self.integrated_gradients = IntegratedGradients
        self.steps = steps
        self.input_embedding = self.model.get_input_embeddings()

    def explain(self, sentence: str) -> dict:
        torch = self.torch
        prompt = build_sentiment_prompt(sentence, PRIMARY_PROMPT)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        label_token_ids = []
        for label in LABELS:
            ids = self.tokenizer.encode(" " + label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{label!r} is not a single next-token label for this tokenizer")
            label_token_ids.append(ids[0])
        embeddings = self.input_embedding(input_ids)
        pad_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        baseline = self.input_embedding(pad_ids)
        output = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        final_logits = output.logits[0, -1, label_token_ids]
        probabilities = torch.softmax(final_logits, dim=0)
        predicted_class = int(torch.argmax(probabilities).item())

        def selected_probability(input_embeds, mask, token_ids, target):
            result = self.model(inputs_embeds=input_embeds, attention_mask=mask)
            logits = result.logits[:, -1, :][:, token_ids]
            return torch.softmax(logits, dim=1)[:, target]

        ig = self.integrated_gradients(selected_probability)
        attribution = ig.attribute(
            embeddings,
            baselines=baseline,
            additional_forward_args=(
                attention_mask,
                label_token_ids,
                predicted_class,
            ),
            n_steps=self.steps,
        )
        subword_scores = attribution.sum(dim=-1)[0].detach().cpu().numpy()
        prompt_start = len(prompt) - len(sentence)
        token_scores = []
        for token in tokenize(sentence):
            start, end = prompt_start + token.start, prompt_start + token.end
            score = sum(
                float(subword_scores[position])
                for position, (offset_start, offset_end) in enumerate(offsets)
                if offset_end > start and offset_start < end
            )
            token_scores.append(score)
        return {
            "prediction": predicted_class,
            "probabilities": probabilities.detach().cpu().tolist(),
            "token_texts": [token.text for token in tokenize(sentence)],
            "scores": token_scores,
        }


def api_agreement(api_predictions: list[int], local_predictions: list[int]) -> dict:
    if len(api_predictions) != len(local_predictions):
        raise ValueError("API and local prediction lists must have equal length")
    agreements = [
        first == second for first, second in zip(api_predictions, local_predictions)
    ]
    return {
        "n": len(agreements),
        "agreement": sum(agreements) / len(agreements) if agreements else None,
        "comparable_to_api": bool(agreements) and sum(agreements) / len(agreements) >= 0.90,
    }
