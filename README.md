# TemporalUV: Capturing Loose Clothing with Temporally Coherent UV Coordinates
Here you can find the source code for the CVPR 2022 paper "TemporalUV: Capturing Loose Clothing with Temporally Coherent UV Coordinates". Authors: You Xie(TUM), Huiqi Mao(NUS), Angela Yao(NUS), Nils Thuerey(TUM).

Our [paper](https://arxiv.org/pdf/2204.03671.pdf)
You can also check out [here](https://ge.in.tum.de/publications/temporaluv-capturing-loose-clothing-with-temporally-coherent-uv-coordinates/) for further details:

We used [Fashion Video Dataset](https://vision.cs.ubc.ca/datasets/fashion/) as our training dataset.

1. Setup DensePose according to [here](https://github.com/facebookresearch/Densepose). Then we can generate DensePose IUV from one single RGB image.
2. UV extension: results in IUV with full silhouette coverage
    	python extrapolation_IUV_mix_final.py
3. UV optimization
    3.1 optimization: remove artifacts in IUV
        python run_opt.py
    3.2 use texture at T_0 as fixed texture for the whole sequence
     I. random sampling for T_0 with
	    python 0.texture_random_sampling.py
     II.texture extrapolation for T_0 with
	  	python 1.texture_extrapolation.py
4. Temporal Relocation
    I.  random sampling for all frames with
		python 0.texture_random_sampling_all.py
    II. temporal relocation with
		python 1.temporal_relocation.py
5. Model training
    I.  Train the model with IUV only
		python 0.training_with_UV.py
    II. add a new image discriminator into the model, and train the image discriminator only with fixing other model components
		python 1.training_img_discriminator.py
    III.Train the model with both IUVs and images
		python 2.trianing_with_UV_img.py

# Virtual try-on results
  ![](virtual_try_on_results/output.gif)

# Acknowledgements
  This research / project was supported by the Ministry of Education, Singapore, under its MOE Academic Re- search Fund Tier 2 (STEM RIE2025 MOE-T2EP20220- 0015) and the ERC Consolidator Grant SpaTe (ERC-2019- COG-863850).
