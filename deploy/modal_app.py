"""Modal app: hosts detect / paraphrase / embed / backtranslate / health.

Deploy: .venv/bin/modal deploy deploy/modal_app.py
App name: humanizer-models (workspace: set by your Modal account)

Each class gets its OWN stable HTTPS URL per web-endpoint method (Modal
fastapi_endpoint behavior — not a shared base path). Record the URLs in .env
as MODAL_DETECT_URL, MODAL_PARAPHRASE_URL, MODAL_EMBED_URL,
MODAL_BACKTRANSLATE_URL, MODAL_HEALTH_URL. NOTE: switching from plain
@app.function endpoints to @app.cls endpoints changes the URL shape — re-copy
all five URLs after your first deploy of this version.

Bug fixes baked in here (see docs/decision-log.md for the full story — do not
regress these):
  - Iteration 6: image needs sentencepiece + protobuf + tiktoken, or T5's fast
    tokenizer silently falls through to a broken tiktoken conversion path.
  - Iteration 6 [OBSOLETE — kept for history]: desklib's forward() had to
    accept & drop **kwargs (token_type_ids), and transformers was pinned
    <4.49 because desklib's custom PreTrainedModel subclass broke on
    >=4.49's all_tied_weights_keys. Both only mattered for desklib, which
    this file no longer uses (see the detector-loading section below for
    why) — the transformers<4.49 pin itself is left in place since nothing
    currently requires unpinning it, not because it's still needed for this
    specific reason.
  - Iteration 12/13: translate/paraphrase ONE SENTENCE AT A TIME — meaning
    each individual model input must be a single sentence, never several
    sentences concatenated into one ~400-token block (both MarianMT and
    T5_Paraphrase_Paws are sentence-level models and go out-of-distribution on
    multi-sentence chunks, producing repetition loops / filler-token
    gibberish). Running several *separate* one-sentence inputs together as a
    padded batch is fine — each sequence is still exactly one sentence, we're
    just not looping over them one-by-one in Python anymore. See the
    Perf iteration note below.
  - Iteration 13: do not force min_length / length_penalty / no_repeat_ngram_size
    on MarianMT or T5 decode — those combinations are the documented trigger for
    degenerate output. Use plain beam decoding for MarianMT, and the model's
    official sampling recipe (do_sample, top_k, top_p, temperature scaled by
    `strength`) for T5_Paraphrase_Paws.
  - Iteration 13 (security hardening): `strength` is validated + clamped to
    int 1..5; both paraphrase and backtranslate cap the per-request sentence
    loop at 200 sentences (DoS guard).
  - Perf iteration (this file): the previous version loaded every model from
    scratch inside the request handler on every single call, and never moved
    the model or its inputs onto the GPU despite requesting one — so `detect`
    and `paraphrase` were paying full model-instantiation cost AND running
    inference on CPU every time, and `backtranslate` had no GPU at all. That
    was the actual source of the 15-40s executions, not the models
    themselves. Fixed by: (1) moving model loading into `@modal.enter()` on a
    Modal class so it happens once per container and is reused across warm
    requests, (2) explicitly moving models + inputs to CUDA and running in
    fp16, (3) batching the per-sentence loop into a single padded
    forward/generate call, (4) giving backtranslate a GPU, (5) keeping
    containers warm for a short window after their last request via
    scaledown_window so back-to-back calls don't repeatedly eat cold-start
    cost, without paying for an always-on GPU 24/7.
  - Adversarial endpoint (this file): previously the caller (Flask app) ran
    the rewrite -> detect -> repeat loop itself, making a separate HTTP call
    into this Modal app for every step. That's now a single /adversarial
    endpoint (see the Adversarial class below) that runs the whole loop
    inside one warm container — one network hop total instead of one per
    iteration.
  - Cost-tiered adversarial strategies: the loop now escalates
    free_pass (regex, $0) -> paraphrase (T5, GPU) -> backtranslate (MarianMT,
    GPU) -> llm_rewrite (Gemini 3.1 Flash Lite API, per-token $) in that
    order, and only reaches the Gemini call at all if the cheaper GPU-only
    strategies didn't already hit target_ai_probability. llm_rewrite calls
    are additionally capped by MAX_LLM_CALLS_PER_REQUEST independent of
    max_iterations, so a hard-to-humanize input can't turn into an unbounded
    number of paid API calls. Requires a Modal Secret named
    "gemini-api-key" with a GEMINI_API_KEY entry — deploy will fail without
    it once Adversarial tries to start.

Outstanding (see decision-log "Outstanding / blockers"): endpoints are public/
unauthenticated with no concurrency_limit — acceptable for a research deploy
on a local/VPN network, add Modal auth + concurrency_limit before wider exposure.
"""
import random
import re

import modal

app = modal.App("humanizer-models")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers<4.49",
        "sentence-transformers",
        "sentencepiece",
        "protobuf",
        "tiktoken",
        "fastapi[standard]",
    )
)

volume = modal.Volume.from_name("humanizer-models-vol", create_if_missing=True)
MODEL_CACHE = "/cache"

MAX_SENTENCES_PER_REQUEST = 200

# Keep a container warm for a few minutes after its last request, so
# back-to-back calls (like the adversarial loop) hit a warm container
# without paying idle GPU cost around the clock like a permanent
# min_containers=1 would. First request after a longer idle gap still pays
# the full cold-start cost.
SCALEDOWN_WINDOW = 300  # seconds — tune between 120-180 as you like


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()][:MAX_SENTENCES_PER_REQUEST]


def _clamp_strength(strength) -> int:
    try:
        s = int(strength)
    except (TypeError, ValueError, OverflowError):
        s = 3
    return max(1, min(5, s))


def _clamp_int(value, lo, hi, default) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError, OverflowError):
        v = default
    return max(lo, min(hi, v))


def _clamp_float(value, lo, hi, default) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        v = default
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------
# Free pass — banned-word/phrase + synonym substitution, zero GPU/API cost.
# Pure regex, no model involved. This is deliberately conservative: it swaps
# known AI-register vocabulary and a couple of structural filler phrases for
# plainer alternatives, and does NOT reorder or restructure sentences (no
# shuffling, no inserted noise — dropped per the "forgo noise and shuffling"
# call). Because it's this cheap, it's applied unconditionally as its own
# zero-cost candidate before any paid rewrite runs, and again after every
# paid candidate as a cleanup pass — in both cases the result still goes
# through the same similarity/score acceptance gate as everything else, so a
# substitution that happens to read worse in context just gets rejected
# rather than silently corrupting the candidate.
#
# The list itself is a known moving target — multiple independent
# "anti-AI-slop" projects converge on almost the same words (delve,
# leverage, tapestry, robust, seamless, pivotal...), which is a good sign
# they're real signals, but also means whichever specific words are public
# knowledge today are also the ones most likely to already be priced into
# detector training. Treat this dict as something to revisit periodically,
# not a fixed asset.
_TELL_REPLACEMENTS: dict[str, list[str]] = {
    r"\bdelves?\b": ["looks at", "goes into"],
    r"\bdelving\b": ["looking at", "going into"],
    r"\bleverages?\b": ["uses", "makes use of"],
    r"\bleveraging\b": ["using"],
    r"\butiliz(?:es?|ing)\b": ["uses", "using"],
    r"\btapestry\b": ["mix", "picture"],
    r"\brealm\b": ["area", "space"],
    r"\brobust\b": ["solid", "reliable"],
    r"\bseamlessly\b": ["smoothly", "cleanly"],
    r"\bseamless\b": ["smooth", "simple"],
    r"\bpivotal\b": ["key", "important"],
    r"\btestament to\b": ["a sign of", "proof of"],
    r"\bgroundbreaking\b": ["new", "notable"],
    r"\btransformative\b": ["big", "significant"],
    r"\bunlock\b": ["open up", "enable"],
    r"\bembark(?:s|ed|ing)? on\b": ["start", "begin"],
    r"\bnavigate the complexities of\b": ["deal with", "work through"],
    r"\bin today'?s fast-paced world\b": ["these days", "right now"],
    r"\bin today'?s ever-evolving landscape\b": ["these days", "currently"],
    r"\bat the end of the day\b": ["ultimately", "in the end"],
    r"\bit'?s worth noting that\b": [""],
    r"\bit is important to note that\b": [""],
    r"\bmoreover\b": ["also", "and"],
    r"\bfurthermore\b": ["also", "and"],
    r"\bhence\b": ["so"],
    r"\bthus\b": ["so"],
}
_TELL_PATTERNS = [(re.compile(p, re.IGNORECASE), repls) for p, repls in _TELL_REPLACEMENTS.items()]


def _free_pass(text: str) -> str:
    if not text:
        return text
    out = text
    for pattern, replacements in _TELL_PATTERNS:
        if pattern.search(out):
            out = pattern.sub(lambda m: random.choice(replacements), out)
    # Collapse any double spaces left behind by phrase-to-empty-string swaps
    # (e.g. "It's worth noting that the results..." -> "the results...").
    out = re.sub(r"\s{2,}", " ", out).strip()
    if out and out[0].islower() and text[0].isupper():
        out = out[0].upper() + out[1:]
    return out


# Adversarial-endpoint tuning + DoS guards.
MAX_ADVERSARIAL_ITERATIONS = 8
DEFAULT_ADVERSARIAL_ITERATIONS = 4
DEFAULT_TARGET_AI_PROBABILITY = 0.35
DEFAULT_MIN_SIMILARITY = 0.72
# Iterations multiply cost, so this endpoint gets a tighter per-request
# sentence cap than the plain paraphrase/backtranslate endpoints.
MAX_SENTENCES_PER_ADVERSARIAL_REQUEST = 60
# The LLM strategy is the expensive one (Gemini API call) — capped
# separately from max_iterations so a long iteration budget can't turn into
# an unbounded number of paid API calls on genuinely hard inputs.
MAX_LLM_CALLS_PER_REQUEST = 2
DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite"


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------
@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------
# Shared AI-text-detector loader — used by both Detector and Adversarial's
# @modal.enter(). Defined once at module level so a fix or a future swap
# only needs to happen in one place.
#
# HISTORY: this used to load desklib/ai-text-detector-v1.01 via a custom
# PreTrainedModel wrapper class (self.model + self.classifier, loaded
# through a nested AutoModel.from_config()). That approach went through
# four rounds of real, distinct bugs in production — a silently
# randomly-initialized backbone (checkpoint key names didn't match a bare
# AutoModel), then two different fp16/dtype mismatches once that was fixed,
# then a DeBERTa-specific fp16 numerical-overflow bug (every input
# saturating to ~100%) once THOSE were fixed. Each one only became visible
# once the previous one stopped hiding it.
#
# Rather than keep patching that integration, swapped to
# Oxidane/tmr-ai-text-detector: RoBERTa-base (125M, smaller and faster than
# desklib's DeBERTa-v3-large), trained on the same RAID benchmark, using
# Focal Loss + Self-Hard-Negative iterative mining specifically to reduce
# false positives on human text — directly targeting the "100% on text
# other detectors call <20% AI" symptom that prompted this switch. Loads
# through plain AutoModelForSequenceClassification, no custom wrapper class
# at all, which is what actually eliminates the whole bug category above
# rather than fixing it a fifth time. RoBERTa also doesn't share DeBERTa's
# fp16 disentangled-attention instability, so fp16 is safe here.
#
# Standard 2-class classification head: index 0 = human, index 1 = AI (per
# this model's documented usage). Label order isn't universally
# standardized across checkpoints — re-verify this if swapping detectors
# again rather than assuming.
#
# Imports live inside the function body, not at true module top level, on
# purpose: this file gets parsed locally by `modal deploy` on a machine that
# may not have torch/transformers installed — only the remote container
# (built from `image`) does. A function body's imports don't execute until
# the function is called, so this stays safe to import/parse locally while
# still only running inside the warm container at request time.
# --------------------------------------------------------------------------
DETECTOR_MODEL_NAME = "Oxidane/tmr-ai-text-detector"
DETECTOR_AI_LABEL_INDEX = 1


def _load_detector_model(name: str, cache_dir: str, device: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            name, cache_dir=cache_dir, torch_dtype=torch.float16
        )
        .to(device)
        .eval()
    )
    return tok, model


def _score_with_detector(tok, model, device: str, text: str) -> float:
    import torch

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1)
        return probs[0][DETECTOR_AI_LABEL_INDEX].item()


# --------------------------------------------------------------------------
# /detect  — Oxidane/tmr-ai-text-detector (default) or RADAR, selected via
# {"model": "..."}. Loaded once per container in @modal.enter(), kept on
# GPU in fp16.
# --------------------------------------------------------------------------
@app.cls(
    image=image,
    volumes={MODEL_CACHE: volume},
    timeout=120,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Detector:
    @modal.enter()
    def load(self):
        self.device = "cuda"
        self.detector_tok, self.detector_model = _load_detector_model(
            DETECTOR_MODEL_NAME, MODEL_CACHE, self.device
        )

        # RADAR is much bigger (Vicuna-7B backbone) — load it lazily, only if
        # a request actually asks for it, but still cache it on the instance
        # so a second request in the same warm container doesn't reload it.
        self._radar = None

    def _load_radar(self):
        if self._radar is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            name = "TrustSafeAI/RADAR-Vicuna-7B"
            tok = AutoTokenizer.from_pretrained(name, cache_dir=MODEL_CACHE)
            model = (
                AutoModelForSequenceClassification.from_pretrained(
                    name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
                )
                .to(self.device)
                .eval()
            )
            self._radar = (tok, model)
        return self._radar

    @modal.fastapi_endpoint(method="POST")
    def detect(self, item: dict):
        import torch

        text = item.get("text", "")
        model_name = item.get("model", DETECTOR_MODEL_NAME)

        if model_name == DETECTOR_MODEL_NAME:
            prob = _score_with_detector(self.detector_tok, self.detector_model, self.device, text)
            return {"ai_probability": prob}

        elif model_name == "TrustSafeAI/RADAR-Vicuna-7B":
            tok, model = self._load_radar()
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
                probs = torch.softmax(logits.float(), dim=-1)
                # class 0 = AI, per docs/architecture.md label-direction note.
                prob = probs[0][0].item()
            return {"ai_probability": prob}

        return {"error": f"unknown model {model_name}"}, 400


# --------------------------------------------------------------------------
# /paraphrase — Vamsi/T5_Paraphrase_Paws
# Loaded once per container; sentences are batched into one padded
# generate() call instead of looped one at a time.
# --------------------------------------------------------------------------
@app.cls(
    image=image,
    volumes={MODEL_CACHE: volume},
    timeout=180,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Paraphraser:
    @modal.enter()
    def load(self):
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.device = "cuda"
        name = "Vamsi/T5_Paraphrase_Paws"
        self.tok = T5Tokenizer.from_pretrained(name, cache_dir=MODEL_CACHE)
        self.model = (
            T5ForConditionalGeneration.from_pretrained(
                name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

    @modal.fastapi_endpoint(method="POST")
    def paraphrase(self, item: dict):
        import torch

        text = item.get("text", "")
        strength = _clamp_strength(item.get("strength", 3))

        # strength 1-5 scales sampling temperature/top_k/top_p — official
        # sampling recipe, no min_length / length_penalty / beam search
        # (Iteration 13 fix).
        temperature = 0.6 + 0.15 * strength    # ~0.75 .. 1.35
        top_k = 40 + 10 * strength             # 50 .. 90
        top_p = min(0.90 + 0.02 * strength, 1.0)   # 0.92 .. 1.00

        sentences = _split_sentences(text)
        if not sentences:
            return {"paraphrased": ""}

        # Each list item is still exactly one sentence going into the model
        # (Iteration 12/13 constraint unchanged) — padding lets the GPU run
        # them all in one forward pass instead of one Python loop iteration
        # per sentence.
        prompts = [f"paraphrase: {s} </s>" for s in sentences]
        enc = self.tok(
            prompts, return_tensors="pt", truncation=True, max_length=256, padding=True
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.inference_mode():
            out_ids = self.model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_length=256,
                early_stopping=True,
            )

        out_sentences = self.tok.batch_decode(out_ids, skip_special_tokens=True)
        return {"paraphrased": " ".join(out_sentences)}


# --------------------------------------------------------------------------
# /backtranslate — MarianMT en->fr->en
# Previously had NO gpu at all (biggest single cause of its slowness).
# Now on GPU, loaded once per container, sentences batched per hop.
# --------------------------------------------------------------------------
@app.cls(
    image=image,
    volumes={MODEL_CACHE: volume},
    timeout=180,
    scaledown_window=SCALEDOWN_WINDOW,
)
class BackTranslator:
    @modal.enter()
    def load(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self.device = "cuda"

        en_fr_name = "Helsinki-NLP/opus-mt-en-fr"
        fr_en_name = "Helsinki-NLP/opus-mt-fr-en"

        self.tok_en_fr = MarianTokenizer.from_pretrained(en_fr_name, cache_dir=MODEL_CACHE)
        self.model_en_fr = (
            MarianMTModel.from_pretrained(
                en_fr_name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

        self.tok_fr_en = MarianTokenizer.from_pretrained(fr_en_name, cache_dir=MODEL_CACHE)
        self.model_fr_en = (
            MarianMTModel.from_pretrained(
                fr_en_name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

    def _translate_batch(self, sentences, tok, model):
        import torch

        enc = tok(
            sentences, return_tensors="pt", truncation=True, max_length=256, padding=True
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            # Plain beam decoding, no no_repeat_ngram_size / repetition_penalty
            # (Iteration 12 fix — those triggered runaway repetition loops).
            out_ids = model.generate(
                **enc, num_beams=4, do_sample=False, early_stopping=True, max_length=256
            )
        return tok.batch_decode(out_ids, skip_special_tokens=True)

    @modal.fastapi_endpoint(method="POST")
    def backtranslate(self, item: dict):
        text = item.get("text", "")
        sentences = _split_sentences(text)
        if not sentences:
            return {"backtranslated": ""}

        fr_sentences = self._translate_batch(sentences, self.tok_en_fr, self.model_en_fr)
        back_sentences = self._translate_batch(fr_sentences, self.tok_fr_en, self.model_fr_en)
        return {"backtranslated": " ".join(back_sentences)}


# --------------------------------------------------------------------------
# /embed — sentence-transformers/all-MiniLM-L6-v2
# Small enough that CPU inference is fine; the win here is purely avoiding
# a fresh model load on every request.
# --------------------------------------------------------------------------
@app.cls(
    image=image,
    volumes={MODEL_CACHE: volume},
    timeout=60,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Embedder:
    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODEL_CACHE
        )

    @modal.fastapi_endpoint(method="POST")
    def embed(self, item: dict):
        text = item.get("text", "")
        vec = self.model.encode(text).tolist()
        return {"embedding": vec}


# --------------------------------------------------------------------------
# /adversarial — runs the rewrite -> detect -> repeat loop entirely inside
# one warm container instead of the caller making N separate HTTP round
# trips to /paraphrase, /backtranslate and /detect. This is a deliberately
# self-contained class (it loads its own copies of the detector, paraphrase,
# and backtranslate models rather than sharing code with the classes above)
# — duplicating ~40 lines of model-loading code is a much smaller risk than
# refactoring the already-deployed, already-working Detector/Paraphraser/
# BackTranslator classes to share it.
#
# Strategy tiers, cheapest to most expensive:
#   0. free_pass  — regex word/phrase substitution, zero GPU/API cost.
#   1. paraphrase — T5, GPU, batched.
#   2. backtranslate — MarianMT x2, GPU, batched. Deterministic (beam
#      search, no sampling) — only meaningfully re-run when best_text has
#      actually changed since its last use, never twice on the same input.
#   3. llm_rewrite — Gemini 3.1 Flash Lite API call, paragraph-level
#      rewrite. The only strategy that can restructure across sentence
#      boundaries rather than one sentence at a time — but it's an API call
#      with real per-token cost, so it's last in the rotation (only reached
#      if the cheaper strategies didn't already hit target) AND separately
#      capped by MAX_LLM_CALLS_PER_REQUEST regardless of max_iterations.
#
# Algorithm, per request:
#   1. Score the original text. If it's already under `target_ai_probability`,
#      return immediately — no rewriting needed.
#   2. Try a zero-cost free_pass on the original text first — some inputs
#      clear target from word-level cleanup alone, with zero paid iterations
#      spent.
#   3. Otherwise rotate through [paraphrase, backtranslate, llm_rewrite]
#      (llm_rewrite skipped once MAX_LLM_CALLS_PER_REQUEST is hit), mutating
#      the current BEST candidate — a rejected candidate is discarded, not
#      built on. Every paid candidate gets a free_pass cleanup applied
#      before scoring. Paraphrase/llm_rewrite strength ramps up each time
#      that specific strategy is reused, capped at 5.
#   4. Every candidate is checked two ways: does its detector score improve
#      on the current best, AND is it still semantically close enough to the
#      ORIGINAL text (via MiniLM cosine similarity) — comparing against the
#      original on every iteration, not the previous candidate, is what
#      stops semantic drift from silently compounding over several rounds.
#      A candidate that drifts too far is discarded even if its AI score is
#      lower.
#   5. Stops early once target_ai_probability is reached, or after
#      max_iterations either way. Returns the best candidate found, plus a
#      per-iteration trace for debugging/benchmarking.
# --------------------------------------------------------------------------
adversarial_image = image.pip_install("google-genai")


@app.cls(
    image=adversarial_image,
    volumes={MODEL_CACHE: volume},
    timeout=300,
    scaledown_window=SCALEDOWN_WINDOW,
    secrets=[modal.Secret.from_name("gemini-api-key")],
)
class Adversarial:
    @modal.enter()
    def load(self):
        import os

        import torch
        from google import genai
        from transformers import (
            T5Tokenizer,
            T5ForConditionalGeneration,
            MarianMTModel,
            MarianTokenizer,
        )
        from sentence_transformers import SentenceTransformer

        self.device = "cuda"

        # ---- detector ----
        self.detect_tok, self.detect_model = _load_detector_model(
            DETECTOR_MODEL_NAME, MODEL_CACHE, self.device
        )

        # ---- paraphraser ----
        para_name = "Vamsi/T5_Paraphrase_Paws"
        self.para_tok = T5Tokenizer.from_pretrained(para_name, cache_dir=MODEL_CACHE)
        self.para_model = (
            T5ForConditionalGeneration.from_pretrained(
                para_name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

        # ---- backtranslator ----
        en_fr_name = "Helsinki-NLP/opus-mt-en-fr"
        fr_en_name = "Helsinki-NLP/opus-mt-fr-en"
        self.bt_tok_en_fr = MarianTokenizer.from_pretrained(en_fr_name, cache_dir=MODEL_CACHE)
        self.bt_model_en_fr = (
            MarianMTModel.from_pretrained(
                en_fr_name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )
        self.bt_tok_fr_en = MarianTokenizer.from_pretrained(fr_en_name, cache_dir=MODEL_CACHE)
        self.bt_model_fr_en = (
            MarianMTModel.from_pretrained(
                fr_en_name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

        # ---- similarity guard ----
        self.embed_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=MODEL_CACHE,
            device=self.device,
        )

        # ---- Gemini client (LLM rewrite strategy) ----
        # Requires a Modal Secret named "gemini-api-key" with a GEMINI_API_KEY
        # entry: `modal secret create gemini-api-key GEMINI_API_KEY=<your key>`
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _score(self, text: str) -> float:
        return _score_with_detector(self.detect_tok, self.detect_model, self.device, text)

    def _similarity(self, a: str, b: str) -> float:
        from sentence_transformers import util

        emb = self.embed_model.encode([a, b], convert_to_tensor=True)
        return util.cos_sim(emb[0], emb[1]).item()

    def _paraphrase(self, text: str, strength: int) -> str:
        import torch

        sentences = _split_sentences(text)
        if not sentences:
            return text

        temperature = 0.6 + 0.15 * strength
        top_k = 40 + 10 * strength
        top_p = min(0.90 + 0.02 * strength, 1.0)

        prompts = [f"paraphrase: {s} </s>" for s in sentences]
        enc = self.para_tok(
            prompts, return_tensors="pt", truncation=True, max_length=256, padding=True
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out_ids = self.para_model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_length=256,
                early_stopping=True,
            )
        return " ".join(self.para_tok.batch_decode(out_ids, skip_special_tokens=True))

    def _backtranslate_hop(self, sentences, tok, model):
        import torch

        enc = tok(sentences, return_tensors="pt", truncation=True, max_length=256, padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out_ids = model.generate(
                **enc, num_beams=4, do_sample=False, early_stopping=True, max_length=256
            )
        return tok.batch_decode(out_ids, skip_special_tokens=True)

    def _backtranslate(self, text: str) -> str:
        sentences = _split_sentences(text)
        if not sentences:
            return text
        fr = self._backtranslate_hop(sentences, self.bt_tok_en_fr, self.bt_model_en_fr)
        back = self._backtranslate_hop(fr, self.bt_tok_fr_en, self.bt_model_fr_en)
        return " ".join(back)

    def _llm_rewrite(self, text: str, strength: int) -> str:
        # Structure first, vocabulary second — fixing word choice alone
        # leaves the uniform "AI rhythm" in place; sentence-length variance
        # is the instruction that actually changes the shape of the prose.
        # Vocabulary tells are also handled deterministically by _free_pass
        # afterwards, so this prompt only needs to nudge register broadly,
        # not enumerate an exhaustive banned-word list.
        intensity = {
            1: "Make only light touch-ups to word choice and a couple of transitions.",
            2: "Lightly vary word choice and a few sentence openings.",
            3: "Moderately restructure: mix short and long sentences, vary openings, keep paragraph order.",
            4: "Substantially restructure: noticeably vary sentence length, reorder some clauses, use plain transitions.",
            5: "Heavily restructure while preserving every fact and claim exactly: mix very short and longer sentences, avoid uniform paragraph rhythm, use natural, slightly informal phrasing.",
        }[strength]

        system_instruction = (
            "You rewrite text so it reads the way a person writes it, not with "
            "the even, uniformly-hedged rhythm typical of AI-generated prose. "
            "Priority order: (1) vary sentence length noticeably within each "
            "paragraph — mix short sentences with longer ones, don't let every "
            "sentence run the same length; (2) avoid formulaic structures like "
            "rule-of-three lists, 'not just X but Y' contrasts, manufactured "
            "on-one-hand/on-the-other balance, and dramatic concluding "
            "sentences; (3) avoid inflated or overly formal vocabulary. "
            "Preserve every fact, number, and claim in the text exactly — do "
            "not add, remove, or soften any of them. Output ONLY the "
            "rewritten text, no commentary, no headers, no quotation marks "
            "around it."
        )

        # Cap output roughly to input size instead of a fixed budget — short
        # inputs shouldn't pay for a large token allowance.
        approx_input_tokens = max(len(text.split()), 20)
        max_output_tokens = min(int(approx_input_tokens * 2.5), 800)

        response = self.gemini_client.models.generate_content(
          model=DEFAULT_LLM_MODEL,
          config={
              "system_instruction": system_instruction,
              "temperature": min(0.5 + 0.15 * strength, 1.2),
              "max_output_tokens": max_output_tokens,
              "thinking_config": {
                   "thinking_level": "LOW",
              },
          },
          contents=f"{intensity}\n\nText:\n{text}",
        )

        rewritten = (response.text or "").strip()
        return rewritten if rewritten else text

    @modal.fastapi_endpoint(method="POST")
    def adversarial(self, item: dict):
        text = item.get("text", "")
        if not text.strip():
            return {"error": "text is required"}, 400

        # Tighter sentence cap than the plain endpoints — this loop repeats
        # the rewrite work up to max_iterations times per request.
        sentences = _split_sentences(text)
        if len(sentences) > MAX_SENTENCES_PER_ADVERSARIAL_REQUEST:
            text = " ".join(sentences[:MAX_SENTENCES_PER_ADVERSARIAL_REQUEST])

        max_iterations = _clamp_int(
            item.get("max_iterations", DEFAULT_ADVERSARIAL_ITERATIONS),
            1, MAX_ADVERSARIAL_ITERATIONS, DEFAULT_ADVERSARIAL_ITERATIONS,
        )
        target = _clamp_float(
            item.get("target_ai_probability", DEFAULT_TARGET_AI_PROBABILITY),
            0.0, 1.0, DEFAULT_TARGET_AI_PROBABILITY,
        )
        min_similarity = _clamp_float(
            item.get("min_similarity", DEFAULT_MIN_SIMILARITY),
            0.0, 1.0, DEFAULT_MIN_SIMILARITY,
        )
        max_llm_calls = _clamp_int(
            item.get("max_llm_calls", MAX_LLM_CALLS_PER_REQUEST),
            0, MAX_LLM_CALLS_PER_REQUEST, MAX_LLM_CALLS_PER_REQUEST,
        )

        original_score = self._score(text)
        best_text = text
        best_score = original_score
        history = [{
            "iteration": 0, "strategy": "original", "strength": None,
            "ai_probability": original_score, "similarity": 1.0,
            "accepted": True,
        }]

        def _try_candidate(iteration: int, strategy: str, strength, candidate: str):
            nonlocal best_text, best_score
            candidate = _free_pass(candidate)
            sim = self._similarity(text, candidate)
            score = self._score(candidate)
            accepted = sim >= min_similarity and score < best_score
            history.append({
                "iteration": iteration, "strategy": strategy, "strength": strength,
                "ai_probability": score, "similarity": sim, "accepted": accepted,
            })
            if accepted:
                best_text, best_score = candidate, score
            return accepted, sim

        # Step 0 — free pass alone on the original text. Zero paid cost;
        # some inputs clear target from word-level cleanup with no rewrite
        # model or API call at all.
        _try_candidate(0, "free_pass", None, text)

        if best_score <= target:
            return {
                "humanized": best_text,
                "ai_probability": best_score,
                "original_ai_probability": original_score,
                "similarity": self._similarity(text, best_text),
                "iterations_used": 0,
                "target_reached": True,
                "history": history,
            }

        # Cheapest to most expensive. llm_rewrite is last, and separately
        # capped — only reached, and only reached a bounded number of times,
        # when the cheaper GPU-only strategies weren't enough on their own.
        strategies = ["paraphrase", "backtranslate", "llm_rewrite"]
        strategy_use_count = {"paraphrase": 0, "backtranslate": 0, "llm_rewrite": 0}
        llm_calls_used = 0
        last_backtranslate_input = None

        for i in range(1, max_iterations + 1):
            strategy = strategies[(i - 1) % len(strategies)]

            if strategy == "llm_rewrite" and llm_calls_used >= max_llm_calls:
                continue  # skip this turn of the rotation, don't spend a paid call

            if strategy == "backtranslate" and best_text == last_backtranslate_input:
                continue  # deterministic — nothing changed since last run, skip

            strategy_use_count[strategy] += 1
            strength = min(2 + strategy_use_count[strategy], 5)

            if strategy == "paraphrase":
                candidate = self._paraphrase(best_text, strength)
            elif strategy == "backtranslate":
                last_backtranslate_input = best_text
                candidate = self._backtranslate(best_text)
            else:
                candidate = self._llm_rewrite(best_text, strength)
                llm_calls_used += 1

            _, sim = _try_candidate(
                i, strategy, strength if strategy != "backtranslate" else None, candidate
            )

            if best_score <= target and sim >= min_similarity:
                break

        return {
            "humanized": best_text,
            "ai_probability": best_score,
            "original_ai_probability": original_score,
            "similarity": self._similarity(text, best_text),
            "iterations_used": len(history) - 1,
            "target_reached": best_score <= target,
            "llm_calls_used": llm_calls_used,
            "history": history,
        }
