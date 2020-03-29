/home/e813/anaconda3/envs/FCOS/bin/python -m torch.distributed.launch \
	--nproc_per_node=2 \
	tools/train_net.py \
	--config-file ./fcos_imprv_R_50_FPN_1x.yaml \
	DATALOADER.NUM_WORKERS 2 \
	OUTPUT_DIR /media/e813/E/output/FCOS/test1
