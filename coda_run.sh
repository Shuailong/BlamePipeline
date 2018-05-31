cl run \
	train.py:script/blameextract/train.py \
	samples-undirected-train.json:dataset/samples-undirected-train.json \
	samples-undirected-dev.json:dataset/samples-undirected-dev.json \
	samples-undirected-test.json:dataset/samples-undirected-test.json \
	embed-dir:elmo-weights \
	data-dir:dataset\
	'python train.py --pretrain-file elmo --batch-size 1 --embed-dir embed-dir --data-dir data-dir --test-batch-size 1 --display-iter 1250 --train-file samples-undirected-train.json --dev-file samples-undirected-dev.json --test-file samples-undirected-test.json --fix-embeddings True --unk-entity True --early-stopping 5 --stats-file True' \
	--request-docker-image shuailongliang/blamepipeline:0.3.1 \
  	--request-gpus 1 \
  	--request-time 2d \
  	--request-disk 2g \
  	--request-memory 32g \
  	-n run-elmo