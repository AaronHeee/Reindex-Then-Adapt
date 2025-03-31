for lr in 1e-4 1e-6; do
	for decay in 0 1e-4 1e-2 1; do
		for agg_freeze in True False; do
			CUDA_VISIBLE_DEVICES=0 python adapt_step/bias_term_adjustment/train.py \
				--decay $decay \
				--epochs 200 \
				--lr $lr \
				--data_names [redial] \
				--freeze_aggregator $agg_freeze \
				--additive_bias True \
				--multiplicative_bias True \
				--ckpt_dir adapt_step/bias_term_adjustment/logs/agg_freeze_$agg_freeze \
				--max_minutes 30
		done
	done
done

for decay in 0 1e-4 1e-2 1; do
	for agg_freeze in True False; do
		CUDA_VISIBLE_DEVICES=0 python adapt_step/recsys_model_gating/train.py \
			--decay $decay \
			--epochs 200 \
			--lr 1e-3 \
			--data_names [redial] \
			--freeze_aggregator $agg_freeze \
			--recsys fism \
			--recsys_with_bias True \
			--ckpt_dir adapt_step/recsys_model_gating/logs/agg_freeze_$agg_freeze \
			--max_minutes 30
	done
done

for decay in 0 1e-4 1e-2 1; do
	for agg_freeze in True False; do
		CUDA_VISIBLE_DEVICES=0 python adapt_step/recsys_model_gating/train.py \
			--decay $decay \
			--epochs 200 \
			--lr 1e-3 \
			--data_names [redial] \
			--freeze_aggregator $agg_freeze \
			--recsys sasrec \
			--recsys_with_bias True \
			--ckpt_dir adapt_step/recsys_model_gating/logs/agg_freeze_$agg_freeze \
			--max_minutes 30
	done
done