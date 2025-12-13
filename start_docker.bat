
docker run --rm -it --gpus all --shm-size=8g ^
  -e HF_TOKEN=your_hf_token_here ^
  -p 7444:7444 ^
  -v F:\sam-3d-body-fbxify-main:/workspace ^
  -v F:\sam-3d-body-fbxify-main\videt_checkpoint:/root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692 ^
  sam3d bash
  