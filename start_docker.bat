echo "Starting Docker Container..."
echo ""

docker run --rm -it --gpus all --shm-size=8g ^
  -e HF_TOKEN=your_hf_token_here ^
  -p 7444:7444 ^
  -v F:\sam-3d-body-fbxify:/workspace ^
  -v F:\sam-3d-body-fbxify\cache\videt_checkpoint:/root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692 ^
  -v F:\sam-3d-body-fbxify\cache\hf_cache:/root/.cache/huggingface ^
  -v F:\sam-3d-body-fbxify\cache\mhr_assets:/opt/venv/lib/python3.12/site-packages/assets ^
  sam-3d-3.12 bash
  