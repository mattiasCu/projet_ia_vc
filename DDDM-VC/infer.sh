python3 inference.py \
    --src_path './sample/src_p227_013.wav' \
    --trg_path './sample/tar_p229_005.wav' \
    --ckpt_model './ckpt/model_base.pth' \
    --ckpt_voc './vocoder/voc_ckpt.pth' \
    --ckpt_f0_vqvae './f0_vqvae/f0_vqvae.pth' \
    --output_dir './converted' \
    -t 6