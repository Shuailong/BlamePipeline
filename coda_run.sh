CMD="python train.py \
     --model-dir . \
     --data-dir . \
     --embed-dir embed-dir \
     --train-file train.json \
     --dev-file dev.json \
     --test-file test.json \
     --pretrain-file elmo \
     --fix-embeddings True \
     --unk-entity False \
     --xavier-init True \
     --early-stopping 5 \
     --batch-size 1 \
     --test-batch-size 1 \
     --display-iter 1250 \
     --stats-file True"
echo $CMD

cl run \
	train.py:script/blameextract/train.py \
	train.json:directed-dataset/samples-directed-train.json \
	dev.json:directed-dataset/samples-directed-dev.json \
	test.json:directed-dataset/samples-directed-test.json \
	embed-dir:elmo-weights \
	"$CMD" \
	--request-docker-image shuailongliang/blamepipeline:0.3.1 \
  	--request-gpus 1 \
  	--request-time 2d \
  	--request-disk 2g \
  	--request-memory 32g \
  	-n run-elmo