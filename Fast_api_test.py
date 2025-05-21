import torch
import uvicorn
import time
import uuid
import os
import contextlib
import functools # For partial function application in hooks
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F # Added for normalize

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# Make sure this path is accessible where you run the FastAPI app
BANNED_TOKEN_FILE = "/workspace/logits-hidden/english_tokens_with_text.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Specify which layers to apply the intervention on (inclusive)
INTERVENTION_START_LAYER = 1 # Example: start from layer 1
INTERVENTION_END_LAYER = 32 # Example: end at layer 32 (adjust based on model)

# --- Global Variables (Loaded at Startup) ---
model = None
tokenizer = None
banned_token_ids_set = None

# --- Helper Functions (from your original code) ---
def load_banned_token_ids_from_file(file_path):
    banned_token_ids = []
    if not os.path.exists(file_path):
        print(f"Warning: Banned token file not found at {file_path}. No tokens will be banned.")
        return set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        token_id = int(parts[0])
                        banned_token_ids.append(token_id)
                    except ValueError:
                        print(f"Skipping invalid line in banned tokens file: {line.strip()}")
                        continue
    except Exception as e:
        print(f"Error loading banned token file {file_path}: {e}")
        return set()
    print(f"Loaded {len(banned_token_ids)} banned token IDs.")
    return set(banned_token_ids)

# --- Your Intervention Logic (Adapted slightly for Hooks) ---
# Note: Reduced printing to avoid excessive logs during generation
def filter_english_in_hidden(
    model_ref, # Pass model explicitly
    tokenizer_ref, # Pass tokenizer explicitly
    banned_ids_set_ref, # Pass banned ids set explicitly
    hidden_states: torch.Tensor,
    layer_idx: int,
    topk: int = 5,
    verbose: bool = False # Control printing
) -> torch.Tensor:
    """
    Applies the hidden state intervention.
    Modifies hidden_states based on likely banned tokens.
    Args:
        model_ref: The loaded AutoModelForCausalLM.
        tokenizer_ref: The loaded AutoTokenizer.
        banned_ids_set_ref: The set of banned token IDs.
        hidden_states: The input hidden states to the layer (output from previous).
        layer_idx: The index of the current layer.
        topk: For debug printing comparison.
        verbose: If True, print debug information.
    Returns:
        The modified hidden_states tensor.
    """
    if not banned_ids_set_ref: # Skip if no tokens are banned
        return hidden_states

    modified_hidden = hidden_states.clone()
    # Assuming batch size is 1 for typical generation endpoint
    batch_size, seq_len, hidden_dim = hidden_states.shape
    if batch_size != 1:
         # This logic currently assumes batch size 1. Modification needed for > 1.
         print("Warning: Intervention logic currently assumes batch size 1.")
         return hidden_states # Skip intervention for simplicity if batch > 1

    λ = 0  # Base suppression coefficient

    # Precompute normed hidden states and logits (for the whole sequence)
    # Need model's norm and lm_head
    try:
        # Access norm and lm_head from the passed model reference
        norm_layer = model_ref.model.norm
        lm_head = model_ref.lm_head
    except AttributeError as e:
        print(f"Error accessing model norm/lm_head: {e}. Skipping intervention.")
        return hidden_states

    # Important: Apply norm *before* calculating logits, like in the original model forward pass
    normed_hidden = norm_layer(hidden_states)
    logits = lm_head(normed_hidden)  # [batch=1, seq, vocab]

    # Process the *last* token's hidden state, as that's what determines the *next* token
    # In generate, seq_len grows, but we only care about the prediction for the next step.
    # The hook runs *after* the layer processes all current tokens.
    pos = seq_len - 1 # Index of the last token's state

    current_logits = logits[0, pos, :] # Logits for the last position
    # Increase k slightly for better chance of catching relevant banned tokens
    topk_values, topk_indices = torch.topk(current_logits, 1000, sorted=True)

    # Collect banned tokens present in the top-k predictions for the *next* token
    current_banned_ids = []
    current_banned_logits = []
    for tid, val in zip(topk_indices, topk_values):
        tid_item = tid.item()
        if tid_item in banned_ids_set_ref:
            current_banned_ids.append(tid_item)
            current_banned_logits.append(val)
            # Limit the number of banned tokens to consider for stability/performance
            # if len(current_banned_ids) >= 20:
            #      break

    if not current_banned_ids: # No relevant banned tokens predicted
        # No modification needed for this position/layer
        return hidden_states # Return original hidden states

    # --- Projection Logic ---
    logits_tensor = torch.stack(current_banned_logits)
    weights = torch.softmax(logits_tensor.float(), dim=0) # Use float32 for softmax stability
    λ_weights = (λ * weights).to(hidden_states.dtype)

    V = lm_head.weight[current_banned_ids, :] # Get embeddings [num_banned, hidden_dim]
    V = F.normalize(V, dim=-1).to(hidden_states.dtype) # Normalize embedding vectors

    h = hidden_states[0, pos, :] # Hidden state for the last token [hidden_dim]

    dots = torch.matmul(V, h)  # Inner products [num_banned]
    # Weighted sum of banned vectors, scaled by inner product and lambda_weights
    projection = torch.sum((dots * λ_weights).unsqueeze(1) * V, dim=0) # [hidden_dim]

    modified_h = h - projection # Subtract the projection

    # Update the hidden state for the last position only
    modified_hidden[0, pos, :] = modified_h

    # --- Debug Printing (Optional) ---
    if verbose and pos == seq_len -1: # Only print for the very last token state being modified
        with torch.no_grad():
            # Calculate logits from the modified hidden state *for the last position*
            mod_h_normed = norm_layer(modified_h.unsqueeze(0).unsqueeze(0)) # Add batch/seq dims
            mod_logits = lm_head(mod_h_normed).squeeze(0).squeeze(0) # Remove batch/seq dims

            orig_top = torch.topk(current_logits, topk)
            mod_top = torch.topk(mod_logits, topk)

            if not torch.equal(orig_top.indices, mod_top.indices):
                print(f"\nIntervention Debug: Layer {layer_idx} Pos {pos}:")
                orig_tokens = [tokenizer_ref.decode([t]) for t in orig_top.indices.tolist()]
                mod_tokens = [tokenizer_ref.decode([t]) for t in mod_top.indices.tolist()]
                orig_probs = F.softmax(orig_top.values.float(), dim=-1).tolist()
                mod_probs = F.softmax(mod_top.values.float(), dim=-1).tolist()
                print(f"  Original Top-{topk}: {list(zip(orig_tokens, orig_probs))}")
                print(f"  Modified Top-{topk}: {list(zip(mod_tokens, mod_probs))}")

    return modified_hidden # Return the potentially modified hidden states

# --- Hook Function Definition ---
# This function will be called after each specified layer's forward pass
# ... (Keep imports, config, globals, load_banned_token_ids_from_file, filter_english_in_hidden) ...

# --- Hook Function Definition (Modified) ---
# REMOVE layer_idx from the arguments here
def _forward_hook(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
    outputs: Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
    layer_idx: int # Pass layer_idx as an EXTRA argument via the closure factory
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Forward hook function to apply intervention.
    `outputs` is typically (hidden_states, present_key_value_states)
    NOTE: layer_idx is now passed separately via closure, not by PyTorch hook call signature.
    """
    global model, tokenizer, banned_token_ids_set # Access globals

    # The primary hidden states are the first element of the output tuple
    hidden_states = outputs[0]

    # Apply the intervention logic, using the layer_idx passed via closure
    modified_hidden_states = filter_english_in_hidden(
        model_ref=model,
        tokenizer_ref=tokenizer,
        banned_ids_set_ref=banned_token_ids_set,
        hidden_states=hidden_states,
        layer_idx=layer_idx, # Use the layer_idx captured by the closure
        verbose=False
    )

    # Return the modified hidden states, keeping the rest of the output tuple intact
    if isinstance(outputs, tuple):
        return (modified_hidden_states,) + outputs[1:]
    else:
        return modified_hidden_states

# --- Context Manager for Hook Management (Modified) ---
@contextlib.contextmanager
def intervention_context(model_to_hook: torch.nn.Module, start_layer: int, end_layer: int):
    """Context manager to register and remove hooks for specified layers."""
    handles = []
    if model_to_hook is None:
         print("Warning: Model not loaded, cannot apply hooks.")
         yield
         return

    # --- Closure Factory for Hook ---
    def create_hook_closure(layer_idx_to_capture: int):
        """Creates the actual hook function with layer_idx captured."""
        def hook_fn(module, inputs, outputs):
            # This inner function is the hook called by PyTorch.
            # It calls our logic function, passing the captured layer_idx.
            return _forward_hook(module, inputs, outputs, layer_idx=layer_idx_to_capture)
        return hook_fn
    # --- End Closure Factory ---

    try:
        num_layers = len(model_to_hook.model.layers)
        actual_start = max(0, start_layer)
        actual_end = min(num_layers - 1, end_layer)

        print(f"Registering hooks for layers {actual_start} to {actual_end}")
        for i in range(actual_start, actual_end + 1):
            # Create a new closure for each layer index i
            hook_closure = create_hook_closure(i)
            handle = model_to_hook.model.layers[i].register_forward_hook(hook_closure)
            handles.append(handle)
        yield
    finally:
        print(f"Removing {len(handles)} hooks...")
        for handle in handles:
            handle.remove()
        print("Hooks removed.")

# ... (Keep Pydantic models, FastAPI app setup, startup_event) ...


# --- Pydantic Models (Same as before) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256
    top_p: Optional[float] = 1.0

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

# --- FastAPI Application ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer when the server starts."""
    global model, tokenizer, banned_token_ids_set
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Force torch_dtype to match expected hidden state dtype if needed
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, # Or torch.bfloat16 if supported/preferred
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("Loading banned token IDs...")
    banned_token_ids_set = load_banned_token_ids_from_file(BANNED_TOKEN_FILE)
    # Determine layer range dynamically if needed, or use config
    global INTERVENTION_END_LAYER
    INTERVENTION_END_LAYER = model.config.num_hidden_layers -1 # Example: Go up to last layer
    print(f"Intervention will be applied from layer {INTERVENTION_START_LAYER} to {INTERVENTION_END_LAYER}")

    print("Model, tokenizer, and banned tokens loaded.")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    # This function remains largely the same, just calls the modified context manager
    global model, tokenizer, banned_token_ids_set

    if model is None or tokenizer is None: # Removed banned_token_ids_set check as it's handled inside filter func
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded yet.")

    try:
        # 1. Format input
        formatted_prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        prompt_token_ids = inputs["input_ids"]
        prompt_length = prompt_token_ids.shape[1]

        # 3. Generate response *within the hook context*
        print(f"Generating response with hooks enabled...")
        outputs = None
        start_time = time.time()
        # Use the modified context manager
        with intervention_context(model, INTERVENTION_START_LAYER, INTERVENTION_END_LAYER):
             with torch.no_grad():
                 outputs = model.generate(
                     **inputs,
                     max_new_tokens=request.max_tokens,
                     temperature=request.temperature if request.temperature > 0 else 1.0,
                     top_p=request.top_p,
                     do_sample=request.temperature > 0,
                     pad_token_id=tokenizer.pad_token_id,
                     eos_token_id=tokenizer.eos_token_id,
                 )
        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        if outputs is None:
             raise HTTPException(status_code=500, detail="Generation failed within hook context.")

        # 4. Decode
        generated_ids = outputs[0][prompt_length:]
        completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion_length = len(generated_ids)

        print(f"Generated text: {completion_text}")

        # 5. Format response
        response_message = ChatMessage(role="assistant", content=completion_text)
        choice = ChatCompletionResponseChoice(index=0, message=response_message, finish_reason="stop")
        usage = UsageInfo(
            prompt_tokens=prompt_length,
            completion_tokens=completion_length,
            total_tokens=prompt_length + completion_length
        )
        response = ChatCompletionResponse(
            model=request.model,
            choices=[choice],
            usage=usage
        )
        return response

    except Exception as e:
        print(f"Error during generation with hooks: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Run the Server ---
if __name__ == "__main__":
    print(f"Attempting to load banned tokens from: {os.path.abspath(BANNED_TOKEN_FILE)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)