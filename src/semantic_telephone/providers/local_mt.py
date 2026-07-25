from __future__ import annotations

import asyncio
import re
from typing import Any, ClassVar

from ..models import TranslationResult

NLLB_LANGUAGE_CODES = {
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ka": "kat_Geor",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
    "no": "nob_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}

M2M100_LANGUAGE_CODES = {
    code: code
    for code in [
        "af",
        "ar",
        "ast",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "br",
        "bs",
        "ca",
        "ceb",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fa",
        "ff",
        "fi",
        "fr",
        "fy",
        "ga",
        "gd",
        "gl",
        "gu",
        "ha",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "ig",
        "ilo",
        "is",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "lb",
        "lg",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "my",
        "ne",
        "nl",
        "no",
        "ns",
        "oc",
        "or",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sd",
        "si",
        "sk",
        "sl",
        "so",
        "sq",
        "sr",
        "ss",
        "su",
        "sv",
        "sw",
        "ta",
        "th",
        "tl",
        "tn",
        "tr",
        "uk",
        "ur",
        "uz",
        "vi",
        "wo",
        "xh",
        "yi",
        "yo",
        "zh",
        "zu",
    ]
}


class TransformersUnavailableError(RuntimeError):
    pass


class _TransformerSeq2SeqProvider:
    name = "transformers"
    category = "local_nmt"
    default_model = ""
    language_codes: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        *,
        model: str | None = None,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        max_input_tokens: int = 450,
        decoding: dict[str, Any] | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model or self.default_model
        self.requested_revision = revision
        self.device_setting = device
        self.dtype_setting = dtype
        self.max_input_tokens = max_input_tokens
        self.decoding = {
            "mode": "greedy",
            "num_beams": 1,
            "max_new_tokens": 512,
            "no_repeat_ngram_size": 3,
            **(decoding or {}),
        }
        self.local_files_only = local_files_only
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._resolved_revision: str | None = revision
        # Loading, tokenizer source mutation, RNG seeding, and generation all
        # touch shared model state. Keep one inference active per model instance.
        self._inference_lock = asyncio.Lock()

    def supports_pair(self, source_language: str, target_language: str) -> bool:
        return (
            source_language in self.language_codes
            and target_language in self.language_codes
            and source_language != target_language
        )

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None = None,
    ) -> TranslationResult:
        if not self.supports_pair(source_language, target_language):
            raise ValueError(f"{self.name} does not support {source_language}->{target_language}")
        async with self._inference_lock:
            return await asyncio.to_thread(
                self._translate_sync, text, source_language, target_language, seed
            )

    def _load(self, source_language: str) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise TransformersUnavailableError(
                "local MT requires the optional 'local-mt' dependencies"
            ) from error
        use_cuda = torch.cuda.is_available() and self.device_setting in {"auto", "cuda"}
        if self.device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self._device = "cuda" if use_cuda else "cpu"
        torch_dtype = None
        if self._device == "cuda" and self.dtype_setting in {"auto", "float16", "fp16"}:
            torch_dtype = torch.float16
        tokenizer_kwargs: dict[str, Any] = {}
        if self.requested_revision:
            tokenizer_kwargs["revision"] = self.requested_revision
        if self.local_files_only:
            tokenizer_kwargs["local_files_only"] = True
        tokenizer_kwargs.update(self._tokenizer_load_kwargs(source_language))
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **tokenizer_kwargs
        )
        model_kwargs: dict[str, Any] = {}
        if self.requested_revision:
            model_kwargs["revision"] = self.requested_revision
        if self.local_files_only:
            model_kwargs["local_files_only"] = True
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_kwargs).to(
            self._device
        )
        self._model.eval()
        self._torch = torch
        self._resolved_revision = (
            getattr(self._model.config, "_commit_hash", None) or self.requested_revision
        )

    def _tokenizer_load_kwargs(self, source_language: str) -> dict[str, Any]:
        return {}

    def _prepare_source(self, source_language: str) -> None:
        del source_language

    def _forced_bos_token_id(self, target_language: str) -> int | None:
        raise NotImplementedError

    def _translate_sync(
        self,
        text: str,
        source_language: str,
        target_language: str,
        seed: int | None,
    ) -> TranslationResult:
        self._load(source_language)
        self._prepare_source(source_language)
        segments = self._segments(text)
        translated: list[str] = []
        segment_log: list[dict[str, Any]] = []
        do_sample = self.decoding.get("mode", "greedy") in {"sample", "sampling"}
        generation_kwargs = {
            key: value for key, value in self.decoding.items() if key not in {"mode", "sampling"}
        }
        generation_kwargs.setdefault("num_beams", 1)
        generation_kwargs["do_sample"] = do_sample
        if generation_kwargs["num_beams"] == 1:
            # Some MT checkpoints carry a beam-search-only value in their
            # generation config. Override it for greedy decoding so current
            # Transformers versions do not warn and ignore it.
            generation_kwargs.setdefault("early_stopping", False)
        forced_bos = self._forced_bos_token_id(target_language)
        if forced_bos is not None:
            generation_kwargs["forced_bos_token_id"] = forced_bos
        if seed is not None:
            self._torch.manual_seed(seed)
            if self._device == "cuda":
                self._torch.cuda.manual_seed_all(seed)
        for index, (segment, separator) in enumerate(segments):
            encoded = self._tokenizer(
                segment,
                return_tensors="pt",
                truncation=False,
            )
            token_count = int(encoded["input_ids"].shape[-1])
            if token_count > self.max_input_tokens:
                raise RuntimeError(
                    f"internal segmentation exceeded safe limit: "
                    f"{token_count}>{self.max_input_tokens}"
                )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                output_ids = self._model.generate(**encoded, **generation_kwargs)
            output = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            translated.append(output + separator)
            segment_log.append(
                {
                    "index": index,
                    "input": segment,
                    "output": output,
                    "input_tokens": token_count,
                    "separator": separator,
                }
            )
        result = "".join(translated).rstrip() if text.rstrip() == text else "".join(translated)
        return TranslationResult(
            text=result,
            provider=self.name,
            model=self.model_name,
            usage={
                "input_characters": len(text),
                "output_characters": len(result),
                "segments": len(segment_log),
                "input_tokens": sum(item["input_tokens"] for item in segment_log),
            },
            deterministic=not do_sample,
            metadata={
                "category": self.category,
                "revision": self._resolved_revision,
                "device": self._device,
                "dtype": str(next(self._model.parameters()).dtype),
                "decoding": {**generation_kwargs, "mode": self.decoding.get("mode", "greedy")},
                "segments": segment_log,
            },
        )

    def _segments(self, text: str) -> list[tuple[str, str]]:
        sentence_parts = re.split(r"(?<=[.!?…])(\s+)", text)
        sentences: list[tuple[str, str]] = []
        for index in range(0, len(sentence_parts), 2):
            sentence = sentence_parts[index]
            separator = sentence_parts[index + 1] if index + 1 < len(sentence_parts) else ""
            if sentence:
                sentences.extend(self._split_oversized(sentence, separator))
        return sentences or [(text, "")]

    def _split_oversized(self, text: str, final_separator: str) -> list[tuple[str, str]]:
        token_ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        special_tokens = int(self._tokenizer.num_special_tokens_to_add(pair=False))
        chunk_size = max(1, self.max_input_tokens - special_tokens)
        if len(token_ids) <= chunk_size:
            return [(text, final_separator)]
        chunks = [
            self._tokenizer.decode(
                token_ids[index : index + chunk_size], skip_special_tokens=True
            ).strip()
            for index in range(0, len(token_ids), chunk_size)
        ]
        return [
            (chunk, final_separator if index == len(chunks) - 1 else " ")
            for index, chunk in enumerate(chunks)
            if chunk
        ]


class NllbTranslationProvider(_TransformerSeq2SeqProvider):
    name = "nllb"
    default_model = "facebook/nllb-200-distilled-600M"
    language_codes = NLLB_LANGUAGE_CODES

    def _tokenizer_load_kwargs(self, source_language: str) -> dict[str, Any]:
        return {"src_lang": self.language_codes[source_language]}

    def _prepare_source(self, source_language: str) -> None:
        self._tokenizer.src_lang = self.language_codes[source_language]

    def _forced_bos_token_id(self, target_language: str) -> int:
        token_id = self._tokenizer.convert_tokens_to_ids(self.language_codes[target_language])
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"unknown NLLB target language: {target_language}")
        return token_id


class M2M100TranslationProvider(_TransformerSeq2SeqProvider):
    name = "m2m100"
    default_model = "facebook/m2m100_418M"
    language_codes = M2M100_LANGUAGE_CODES

    def _prepare_source(self, source_language: str) -> None:
        self._tokenizer.src_lang = self.language_codes[source_language]

    def _forced_bos_token_id(self, target_language: str) -> int:
        return int(self._tokenizer.get_lang_id(self.language_codes[target_language]))
