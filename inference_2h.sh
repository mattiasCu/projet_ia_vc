python3 inference_2h.py \
    --folder '/home/ensta/ensta-sui/workplace/data/' \
    --ckpt_model './ckpt/G_520000.pth' \
    --ckpt_voc './vocoder/voc_ckpt.pth' \
    --ckpt_f0_vqvae './f0_vqvae/G_720000.pth' \
    --output_dir './converted' \
    -t 6