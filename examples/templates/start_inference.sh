# MODEL_ID=unsloth/gemma-4-31B-it-GGUF
# MODEL_QUANT_FORMAT=UD-Q8_K_XL
MODEL_PATH=/home/direct/Projects/models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
PORT=8000
API_KEY=key
MAX_NEW_TOKENS=2000
MAX_CONTEXT_LENGTH=13200

# If MODEL_PATH is set, use that instead of MODEL_ID
if [ -n "$MODEL_PATH" ]; then
    MODEL_SPECIFIER="-m ${MODEL_PATH}"
    echo "Model: ${MODEL_PATH}"
else
    MODEL_SPECIFIER="-hf $MODEL_ID:$MODEL_QUANT_FORMAT"
    echo "Model: ${MODEL_ID}:${MODEL_QUANT_FORMAT}"
fi

llama-server \
    $MODEL_SPECIFIER \
    -c ${MAX_CONTEXT_LENGTH:-132000} \
    --host 0.0.0.0 \
    --port $PORT \
    --api-key "${API_KEY}"\
    -n ${MAX_NEW_TOKENS:-2000}
