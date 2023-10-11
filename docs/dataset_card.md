U-NET for Semantic Crop Segmentation - Dataset Card
==============================

# Training Dataset

## Dataset Description
The training dataset is composed of images of sugar beets from two different fields, Field A and Field B, taken in different weeks of growth and in different atmospheric conditions (cloudy, sunny, overcast) for a total of 675 images of various sizes.

The dataset does not provide any form of labels or masks, so the primary intended use is for unsupervised learning models.

### Dataset Summary

- **Name:** UAV Sugarbeets 2015-16
- **Homepage:** [UAV Sugarbeets 2015-16 Datasets](https://www.ipb.uni-bonn.de/data/uav-sugarbeets-2015-16/)

## Dataset Structure

The following table gives an overview of the datasets:

| **Field** | **Ses.** | **Date** | **# of images** | **Crop size** | **Weather** |
|:---------:|:--------:|:--------:|:---------------:|:-------------:|:-----------:|
| A         | 1        | May 20   | 45              | 7 cm          | cloudy      |
| A         | 2        | May 27   | 175             | 10 cm         | sunny       |
| A         | 3        | June 17  | 121             | 15 cm         | overcast    |
| A         | 4        | June 22  | 140             | 20 cm         | cloudy      |
| B         | 1        | May 8    | 99              | 5 cm          | sunny       |
| B         | 2        | June 5   | 95              | 15 cm         | cloudy      |

### Data Instances

#### Field A
For Field A, the dataset images were collected across four sessions using a DJI MATRICE 100 UAV. The flight altitude for each session is between 8 m to 12 m above the ground. The images were recorded using the Zenmuse X3 camera with an image resolution of `4000×2250` pixels having a ground sampling distance of 4 mm per pixel at a height of 10 m.

![field A](https://cdn.albertovalerio.com/datasets/crop_segmentation/dataset/field_A.png)

#### Field B
For Field B, a DJI PHANTOM 4 UAV was used across two sessions recorded almost one month apart. The UAV was equipped with a GoPro camera set up to take an image every second at a resolution of `3840×2880`. The flight altitude for the two sessions varied between 10 m and 18 m above the ground having a ground sampling distance of 9 mm per pixel at 15 m height. The average plant sizes in the fields range from 5 cm to 20 cm in diameter across the crop season.

![field B](https://cdn.albertovalerio.com/datasets/crop_segmentation/dataset/field_B.png)

## Additional Information

### Contributions
Thanks to [Raghav Khanna](https://scholar.google.com/citations?user=1faSRIkAAAAJ) and [Frank Liebisch](https://scholar.google.com/citations?user=oYY_g2YAAAAJ) for assisting with the data acquisition campaign and [ETH Zurich Crop Science](https://kp.ethz.ch/) group for providing access to the fields.


</br>
</br>
</br>

# Testing Dataset

## Dataset Description
The testing dataset contains several sugarbeet field images taken from aerial point of view whose pixels have been manually labeled, <span style="color:lightgreen">green</span> for crops and <span style="color:red">red</span> for weeds, then from each image tiles have been extracted of size `360x480` pixels.

The primary intended use is for large-scale semantic weed mapping framework using Deep Neural Network for precision farming.

### Dataset Summary

- **Name:** Remote Sensing 2018 Weed Map
- **Homepage:** [Remote Sensing 2018 Weed Map](https://projects.asl.ethz.ch/datasets/doku.php?id=weedmap:remotesensing2018weedmap)

## Dataset Structure

Data was collected from sugar beet fields in Eschikon, Switzerland, and Rheinbach, Germany, with a time interval of five months (see the table below) using two commercial quadrotor UAV platforms carrying multispectral cameras, i.e., RedEdge-M and Sequoia from MicaSense.

| **Description** |  **Field A**  |   **Field B**   |
|:---------------:|:-------------:|:---------------:|
| Location        | Eschikon (CH) | Rheinbach (GER) |
| Date            | 5-8 May, 2017 | 18 Sep 2017     |
| Time            | 12.00 PM      | 9:00-18:40 AM   |
| Aeral Platform  | Mavic Pro     | Inspire 2       |
| Sensor          | Sequoia       | RedEdge         |
| Crop            | Sugar beet    | Sugar beet      |
| Altitude        | 10m           | 10m             |

### Data Instances

The datasets consist of 129 directories and 18,746 image files, and this section explains their folder structure.

![folder structure](https://cdn.albertovalerio.com/datasets/crop_segmentation/dataset/folderstructure.png)

After the file extraction, you can find two folders named Orthomosaic and Tiles that contain orthomosaic maps and the corresponding tiles (portions of the region in an image with the same size as that of the input image). Multiple tiles were cropped from an orthomosaic map by sliding a window over it until the entire map is covered.

Orthomosaic folder contains RedEdge and Sequoia subfolders that include 8 orthomosaic maps. Each of the subfolders indexed 000-007 and contains `composite-png`, `groundtruth`, `reflectance-tif` folders. Similar to Sequoia folder, there are 8 tiles and each contains `groundtruth`, `mask` and `tile` folders.

![folder structure](https://cdn.albertovalerio.com/datasets/crop_segmentation/dataset/weedmap.jpg)

## Dataset Creation

### Source Data

The datasets are indexed from 000 to 007, and follow a straightforward naming convention. For example, the groundtruth folder from `Orthomosaic>Rededge>000` contains orthomosaic image of dataset 000, i.e., RedEdge dataset.

The same conventions applied to Tiles folder, e.g., 000-007 indicating dataset indices, but groundtruth, mask, tile require more explanation.

The groundtruth folder contains manually labeled images and the original RGB (or CIR) images. The labelled images are in two formats; color and indexed images. The file convention of the former is `XXX_frameYYYY_GroundTruth_color.png` and `XXX_frameYYYY_GroundTruth_iMap.png` for the latter. In these file names, `XXX` indicates the dataset index, i.e., ranging from 000 to 007, and `YYYY` is a four-digit frame number that starts from 0000 (not 0001).

The mask folder contains binary masks (<span style="color:white">white</span>: valid, <span style="color:gray">black</span>: invalid) of the tile images. The file naming convention is `frameYYYY`, where `YYYY` indicates the frame number mentioned above.

The tile folders consist of 8 and 6 subfolders, RedEdge and Sequoia respectively. Each folder corresponds to a single channel and the file naming convention follows same as mask such that `frameYYYY` where `YYYY` indicates the frame number.

## Additional Information

### Contributions
In case of questions, please feel to contact the Flourish team via e-mail:
 - inkyu.sa@mavt.ethz.ch
 - raghav.khanna@mavt.ethz.ch
 - marija.popovic@mavt.ethz.ch.