---
language:
- en

tags:
- computer-vision
- semantic-segmentation
- deep-learning
- u-net
- precision-agriculture

datasets:
- uav_sugar_beets_2015_16
- remote_sensing_2018_weed_map

metrics:
- pixel_accuracy
- intersection_over_union

model-index:
- name: u-net_for_semantic_crop_segmentation
  results:
  - task:
      type: semantic-segmentation
      name: Semantic Segmentation
    dataset:
      type: uav_sugar_beets_2015_16
      name: UAV Sugar Beets 2015-16
    metrics:
      - type: pixel_accuracy
        value: 76.22
        name: Mean Accuracy (no black)
      - type: intersection_over_union
        value: 63.00
        name: Mean IoU (no black)
  - task:
      type: semantic-segmentation
      name: Semantic Segmentation
    dataset:
      type: remote_sensing_2018_weed_map
      name: Remote Sensing 2018 Weed Map
    metrics:
      - type: pixel_accuracy
        value: 89.57
        name: Mean Accuracy
      - type: intersection_over_union
        value: 83.76
        name: Mean IoU

co2_eq_emissions:
	emissions: 2.04
	source: "CodeCarbon"
	training_type: "pre-training"
	geographical_location: "Apulia, Italy"
	hardware_used: "Apple M2 Max"
---

U-NET for Semantic Crop Segmentation - Model Card
==============================

## Model details

- **Person or organization developing model**: Alberto G. Valerio, Computer Vision case study, a.y. 2022-2023
- **Model date**: 2023
- **Model version**: 1.0
- **Model type**: The model is based on a U-NET neural network, which represents the current state-of-the-art for semantic segmentation;
- **Training algorithms, parameters, fairness constraints or other applied approaches, and features**: The Adam function was defined as the optimizer which takes as input the parameters of our model and the learning rate, and as loss function the BCE with Logits Loss was used;
- **Paper or other resource for more information**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) by Ronneberger et al.;
- **Citation details**: 
    ```
    @article{DBLP:journals/corr/RonnebergerFB15,
    author       = {Olaf Ronneberger and
                    Philipp Fischer and
                    Thomas Brox},
    title        = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
    journal      = {CoRR},
    volume       = {abs/1505.04597},
    year         = {2015},
    url          = {http://arxiv.org/abs/1505.04597},
    eprinttype    = {arXiv},
    eprint       = {1505.04597},
    timestamp    = {Mon, 13 Aug 2018 16:46:52 +0200},
    biburl       = {https://dblp.org/rec/journals/corr/RonnebergerFB15.bib},
    bibsource    = {dblp computer science bibliography, https://dblp.org}
    }
    ```
## Intended Use
- **Primary intended uses**: The model is intended to be used for semantic segmentation tasks.
- **Primary intended users**:
    - **Farmers and Agriculturists**: They can use this system to monitor crop health and growth, identify areas of concern in their fields, and make informed decisions about irrigation, fertilization, and pest control;
    - **Drone Service Providers**: Companies that offer drone imaging services for agriculture can use this system to provide value-added services like crop segmentation and health analysis;
    - **Government Agencies**: Government agricultural departments can use this system for large scale monitoring of crop health and yield prediction. This can aid in planning and policy-making related to food security.
- **Out-of-scope use cases**: 
    - **Urban Planners and Landscape Architects**: While not the primary users, these professionals could potentially use this system to analyze green spaces in urban areas. The system could help identify different types of vegetation and their distribution, aiding in the design and maintenance of urban green spaces;
    - **Golf Course Managers**: This might be somewhat unexpected, but managers of golf courses could use this system to monitor the health of the grass and other plants on the course. This could help in timely maintenance and ensure optimal playing conditions;
    - **Real Estate Developers**: Real estate developers could use this system to analyze aerial images of potential development sites. The system could help identify areas with dense vegetation, which could impact construction plans.

## Factors
- **Relevant Factors**: The model’s performance may be influenced by the quality and diversity of the training data. Factors such as lighting conditions, image resolution, and the presence of occlusions could potentially affect the model’s performance.


## Metrics
- **Model Performance Measures**: To evaluate the effectiveness of the model on new data, two measures commonly adopted for semantic segmentation have been employed
    - **Pixel accuracy** is the percentage of pixels in a predicted mask that have been correctly classified; although this metric is commonly reported separately for each class, a global performance indicator across all classes can be expressed through the accuracy. While pixel accuracy is quite immediate, it is worth noting that high pixel accuracy doesn’t always imply better segmentation abilities; in fact, this type of metric may provide misleading results when the class representation is small within the image, as the measure will be biased in mainly reporting how well you identify negative cases, i.e., where the class is not present. Unfortunately, the issue of unbalanced classes is prevalent in many real-world datasets;
    - **Intersection-Over-Union (IoU)** might be useful to overcome the issue of unbalanced classes. Also known as the Jaccard Index, IoU is the most commonly used metrics in semantic segmentation as well as the most effective. The IoU is essentially a method to quantify the overlap percentage between the target and predicted masks; the overlap can be quantified as the intersection of the predicted and ground-truth segmentations over their union, instead. For binary or multi-class segmentation problems, the mean IoU of the image is calculated by averaging the IoU of each class.
- **Variation Approaches**: The obtained results have been numerically evaluated using the metrics described earlier. Evaluation metrics have been applied to the masks predicted by the trained model and successively a comparison was made between the traditional approaches and the proposed model.

## Training Data
- **[Dataset](https://github.com/se4ai2324-uniba/CropSegmentation/blob/main/data/README.md)**: The model was trained on the ```UAV Sugar beets 2015-16```, which includes images of sugar beets from two fields under various conditions;
- **Pre-processing**: Images were divided into smaller patches, background was removed using a selective mask and the K-means algorithm was applied for segmentation. Random black masks were also applied to 20% of the images to help the model learn additional patterns. This resulted in 35,504 images with corresponding *pseudo-labels*.

## Evaluation Data
- **[Dataset](https://github.com/se4ai2324-uniba/CropSegmentation/blob/main/data/README.md)**: The testing dataset ```Remote Sensing 2018 Weed Map``` contains several sugar beet fields pictures taken from an aerial point of view and whose pixels have been manually labeled, <span style="color:lightgreen">green</span> for crops and <span style="color:red">red</span> for weeds; successively, tiles of size ```360x480``` pixels were extracted from each image. For the experiments, the images taken from ```Red Edge 003``` field were used, for a total number of 204 images.
- <a name="preprocessing"></a>**Pre-processing**: The labels ```crops``` and ```weeds``` were merged into a single category, as the distinction was not relevant for the purposes of our experiments. Moreover, considering that the test set contains many totally black images due to the tiles division, those images containing only black irrelevant areas close to the edges were eliminated from the calculation of the metrics for the purpose of a higher precision and a more genuine evaluation.

## Quantitative Analyses

The model proved to perform effectively, producing highly accurate predictions and successfully identifying patterns within the images of the testing set that included the edges of the field, which contained darkened and irrelevant spaces.

As stated in sub-section [Pre-processing](preprocessing), after removing totally black images containing irrelevant areas, the proposed model achieves a mean accuracy of **76.22**, surpassing the traditional techniques of <span style="color:lightgreen">**5.44%**</span>; as concernes the mean IoU, our model scores **63.00**, once again surpassing the traditional techniques by <span style="color:lightgreen">**9.58%**</span>.


| | Traditiona Techniques | Proposed Method | Gain (%) |
|--------|---|---|---|
|Mean Accuracy|83.93|89.57|+3.04|
|Mean Accuracy (no black)|72.29|76.22|+5.44|
|Mean IoU|79.94|83.76|+4.78|
|Mean IoU (no black)|57.50|63.00|+9.58|


Comparing the model’s results with the current state-of-the-art results in crop segmentation for different deep learning models, the proposed method is able to outperform those results by <span style="color:lightgreen">**4.60%**</span> over the mean value. In this case the gain was calculated with respect to the average value of the results produced by each method.

| | Average IoU (%) | Gain (%) |
|---|---|---|
|ERM|60.21|+1.81| 
|IBN|55.79    |-2.61    | 
|ISW| 57.12| -1.28| 
|pAdaIN|58.92|+0.52| 
|XDED| 59.93| +1.53|
|**U-NET** | **63.00** | **+4.60**|

## Ethical Considerations
Any semantic segmentation system, including this U-NET model, should be used responsibly. User privacy should be respected, and the technology should not be used for activities that infringe on people’s rights. Furthermore, care should be taken to ensure that the technology does not perpetuate bias or discrimination. It’s also important to consider the ethical implications of using this model in sensitive applications, as the model’s predictions could potentially have significant real-world impacts. Users should be aware of these considerations and use the model responsibly.

## Caveats and Recommendations
The model’s performance may be influenced by the quality and diversity of the training data. Factors such as lighting conditions, image resolution, and the presence of occlusions could potentially affect the model’s performance. Users should be aware that while the model has been trained to perform semantic segmentation tasks accurately, it may still make mistakes or fail to generalize well to new, unseen data. It’s recommended to perform additional tests and validations when applying this model to new datasets or use cases. This will help ensure that the model meets specific needs and expectations.

## Carbon Footprint
The model has a reported carbon footprint of 2.04g CO2e for its pre-training phase, as estimated by CodeCarbon. This metric reflects the environmental impact associated with the computational resources used during the model’s development. Specifically, the training was conducted in Apulia, Italy, utilizing a hardware configuration of Apple M2 Max. It is essential for users to be aware of these details to make informed decisions regarding the sustainability of their machine learning operations.

<p align="center">
  <img src="https://cdn.albertovalerio.com/datasets/crop_segmentation/dataset/emissions.jpg" alt="U-NET Model Carbon Footprint" title="Carbon Footprint Infographic" width="400"/>
</p>