python3 inference.py \
    --src_path './DDDM-VC/sample/src_p227_013.wav' \
    --trg_path './DDDM-VC/sample/tar_p229_005.wav' \
    --ckpt_model './DDDM-VC/ckpt/model_base.pth' \
    --ckpt_voc './DDDM-VC/vocoder/voc_ckpt.pth' \
    --ckpt_f0_vqvae './DDDM-VC/f0_vqvae/f0_vqvae.pth' \
    --output_dir './DDDM-VC/converted' \
    -t 6