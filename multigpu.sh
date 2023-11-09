SHARDS=2
QUANT=q0f16

while getopts q:s: flag
do
    case "${flag}" in
        q) QUANT=${OPTARG};;
        s) SHARDS=${OPTARG};;
    esac
done
MODEL=CodeLlama-34b-Instruct-hf

echo "Model: $MODEL";
echo "Quant: $QUANT";
echo "Shards: $SHARDS";

echo "Clear existing build folder..."
rm -rf ./dist/$MODEL-$QUANT/

echo "Build model..."
python -m mlc_llm.build --model $MODEL --quantization $QUANT --num-shards $SHARDS --max-seq-len 4096 --target cuda-multiarch --use-cuda-graph --use-flash-attn-mqa --build-model-only

echo "Convert weights..."
python -m mlc_llm.build --model $MODEL --quantization $QUANT --num-shards $SHARDS --max-seq-len 4096 --target cuda-multiarch --use-cuda-graph --use-flash-attn-mqa --convert-weight-only

echo "Storing config..."
mv ./dist/$MODEL-$QUANT ./dist/$MODEL-s$SHARDS-$QUANT

echo "Benchmarking..."
echo "Model: $MODEL";
echo "Quant: $QUANT";
echo "Shards: $SHARDS";
python tests/benchmark.py --model $MODEL --quantization $QUANT --num-shards $SHARDS --num-warm-up=2 --num-measurements=5 --benchmark-mode=tvm --num-input-tokens=2000 --num-output-tokens=128


