for lr in 1e-4; do
	for decay in 0 1e-4 1e-2 1; do
		for agg_freeze in True False; do
			CUDA_VISIBLE_DEVICES=0 python adapt_step/bias_term_adjustment/train_for_reddit.py \
				--epochs 1000 \
				--data_names [redditv1.5] \
				--freeze_aggregator $agg_freeze \
				--additive_bias True \
				--lr $lr \
				--decay $decay \
				--multiplicative_bias True \
				--print_every 2 \
				--ckpt_dir adapt_step/bias_term_adjustment/logs/reddit_gw_b/agg_freeze_$agg_freeze \
				--max_minutes 180
		done
	done
done

for lr in 1e-3; do
	for decay in 0 1e-6 1e-4 1e-2 1; do
		for agg_freeze in True False; do
			CUDA_VISIBLE_DEVICES=0 python adapt_step/recsys_model_gating/train_for_reddit.py \
				--epochs 1000 \
				--data_names [redditv1.5] \
				--freeze_aggregator $agg_freeze \
				--lr $lr \
				--recsys fism \
				--recsys_with_bias True \
				--decay $decay \
				--print_every 2 \
				--ckpt_dir adapt_step/recsys_model_gating/logs/fism/agg_freeze_$agg_freeze \
				--max_minutes 180
		done
	done
done

for lr in 1e-3; do
	for decay in 0 1e-6 1e-4 1e-2 1; do
		for agg_freeze in True False; do
			CUDA_VISIBLE_DEVICES=0 python adapt_step/recsys_model_gating/train_for_reddit.py \
				--epochs 1000 \
				--data_names [redditv1.5] \
				--freeze_aggregator $agg_freeze \
				--lr $lr \
				--recsys sasrec \
				--recsys_with_bias True \
				--decay $decay \
				--print_every 2 \
				--ckpt_dir adapt_step/recsys_model_gating/logs/sasrec/agg_freeze_$agg_freeze \
				--max_minutes 180
		done
	done
done