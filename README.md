# DeepSlides



In pathology, Immunohistochemical staining (IHC) of tissue sections is regularly used to diagnose and grade malignant tumors. Typically, IHC stain interpretation is rendered by a trained pathologist using a manual method, which consists of counting each positively- and negatively-stained cell under a microscope. The manual enumeration suffers from poor reproducibility even in the hands of expert pathologists. To facilitate this process, we propose a novel method to create artificial datasets with the known ground truth which allows us to analyze the recall, precision, accuracy, and intra- and inter-observer variability in a systematic manner,  enabling us to compare different computer analysis approaches. Our method employs a conditional Generative Adversarial Network that uses a database of Ki67 stained tissues of breast cancer patients to generate synthetic digital slides.

This study is based on [Tensorflow implementation](https://github.com/affinelayer/pix2pix-tensorflow) of [pix2pix](https://phillipi.github.io/pix2pix/). 

if you want to download the image dataset, please click [here](https://doi.org/10.5281/zenodo.1184621)
### Tech
DeepSlides uses a number of open source projects to work properly:

* [Python] 
* [Tensorflow] 
* [Numpy] 

### How to use:
- Download and unzip trained [network](https://doi.org/10.5281/zenodo.1184644)  for ki67 stained breast cancer. (to "code\artificialKi67")
- Please create a png file. (\Testdata\xxx.png)
- The size of the square image should be 256\*n X 256\*n where n=1,2,3,...
- Draw the possitive and negative cells with green and red colors respectively.
- Also you can use yellow color to represent possitive cells with staining problem.
- Run createTissue.py --scale_size  **image size (256*n)** 
**note: Please be sure that images sizes are same in that folder.**
**note: Please be sure that the annotated cells' area are realistic for 40x magnification**
##### example image:

![input mask](images/input.png)

##### output image:

![input mask](images/output.png)
### Todos

 - Web based client GUI

