MODEL=CodeLlama-34b-Instruct-hf
QUANT=q4f16_1

SHARDS=2

echo "Clear existing build folder..."
rm -rf ./dist/$MODEL-$QUANT/

echo "Build model..."
python -m mlc_llm.build --model $MODEL --quantization $QUANT --num-shards $SHARDS --max-seq-len 4096 --target cuda-multiarch --use-cuda-graph --use-flash-attn-mqa --build-model-only

echo "Convert weights..."
python -m mlc_llm.build --model $MODEL --quantization $QUANT --num-shards $SHARDS --max-seq-len 4096 --target cuda-multiarch --use-cuda-graph --use-flash-attn-mqa --convert-weight-only

echo "Benchmarking..."
python tests/benchmark.py --model $MODEL --quantization $QUANT --num-shards $SHARDS --num-warm-up=2 --num-measurements=5 --benchmark-mode=tvm --num-input-tokens=2000 --num-output-tokens=128

echo "Storing config..."
mv ./dist/$MODEL-$QUANT ./dist/$MODEL-s$SHARDS-$QUANT

