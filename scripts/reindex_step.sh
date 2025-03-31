AGGREGATOR=rnn # Options: rnn, sum, weighted, trm, we find rnn works best for the reindex step.

for lr in 1e-3 1e-2 1e-4; do
    for decay in 0 1e-6 1e-4 1e-2; do
        CUDA_VISIBLE_DEVICES=0 python reindex_step/train.py \
            --decay ${decay} \
            --lr ${lr} \
            --aggegator ${AGGREGATOR}
    done
done

for i in {0..11}; do
    CUDA_VISIBLE_DEVICES=0 python reindex_step/test.py \
    --load_pretrained_weights reindex_step/logs/inspired_redial_redditv1.5_wikipedia/rnn/version_${i}/checkpoints/best.ckpt \
    --aggregator ${AGGREGATOR} 
done