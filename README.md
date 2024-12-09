# Scaling Down Text Encoder of Text-To-Image Diffusion Models
This repository provides the implementation for our paper "Scaling Down Text Encoder of Text-To-Image Diffusion Models". We replace the large T5-XXL in Flux with T5-Base, achieving 50 times reduction in model size.

## Install environment
```shell
conda create -n scaling python=3.12
conda activate scaling
pip install -r requirements.txt
```

Diffusers may occasionally raise a 'no attribute _execution_device' error when using custom pipelines. For more details, refer to this [issue](https://github.com/huggingface/diffusers/issues/9180). To resolve this, we recommend replacing all instances of _execution_device in your pipeline with self.transformer.device, or installing our modified version of Diffusers.
```shell
pip install ./diffusers
```

 ## Download models
Download the checkpoint from drive and put it in ckpt folder.

 ## Inference
```python
pyth
on inference_flux.py --ckpt_name $ckpt_name
``` 