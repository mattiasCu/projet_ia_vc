python3 inference.py \
    --src_path '/home/ensta/ensta-sui/workplace/DDDM-VC/sample/src_p227_013.wav' \
    --trg_path '/home/ensta/ensta-sui/workplace/DDDM-VC/sample/tar_p229_005.wav' \
    --ckpt_model '/home/ensta/ensta-sui/workplace/DDDM-VC/ckpt/model_base.pth' \
    --ckpt_voc '/home/ensta/ensta-sui/workplace/DDDM-VC/vocoder/voc_ckpt.pth' \
    --ckpt_f0_vqvae '/home/ensta/ensta-sui/workplace/DDDM-VC/f0_vqvae/f0_vqvae.pth' \
    --output_dir '/home/ensta/ensta-sui/workplace/DDDM-VC/converted' \
    -t 6
