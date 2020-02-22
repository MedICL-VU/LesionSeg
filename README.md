![](https://img.shields.io/badge/language-python-brightgreen.svg)
[![](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://github.com/HuahongZhang/LesionSeg/blob/master/LICENSE)

# Multiple Sclerosis Lesion Segmentation

The repository is the implementation of the paper "Multiple Sclerosis Lesion Segmentation with Tiramisu and 2.5D Stacked Slices". 
The paper link is https://link.springer.com/chapter/10.1007/978-3-030-32248-9_38

Currently, the model has the state-of-the-art performance for MS lesion segmentation and it was evaluated on the ISBI 2015 dataset. 
The overall score of 93.1 placed the L2-loss variant of our method in the first position on the leaderboard, 
while the focal-loss variant has obtained the best Dice coefficient and lesion-wise true positive rate with 69.3% and 60.2%, respectively. 
The ISBI 2015 challenge website is https://smart-stats-tools.org/lesion-challenge

If you would like to cite our work, the reference is
```
@inproceedings{zhang_multiple_2019,
	address = {Cham},
	series = {Lecture {Notes} in {Computer} {Science}},
	title = {Multiple {Sclerosis} {Lesion} {Segmentation} with {Tiramisu} and 2.5D {Stacked} {Slices}},
	isbn = {978-3-030-32248-9},
	doi = {10.1007/978-3-030-32248-9_38},
	booktitle = {Medical {Image} {Computing} and {Computer} {Assisted} {Intervention} – {MICCAI} 2019},
	publisher = {Springer International Publishing},
	author = {Zhang, Huahong and Valcarcel, Alessandra M. and Bakshi, Rohit and Chu, Renxin and Bagnato, Francesca and Shinohara, Russell T. and Hett, Kilian and Oguz, Ipek},
	editor = {Shen, Dinggang and Liu, Tianming and Peters, Terry M. and Staib, Lawrence H. and Essert, Caroline and Zhou, Sean and Yap, Pew-Thian and Khan, Ali},
	year = {2019},
	keywords = {Deep learning, Multiple sclerosis, Segmentation},
	pages = {338--346}
}
```

## Prerequisites
* NVIDIA GPU + CUDA + cuDNN

## Installation
We suggest installing the dependencies using Anaconda
* create the environment and activate (replace DL with your environment name)
```shell script
conda create --name DL python=3.6
```
* Install PyTorch with the official guide from http://pytorch.org (we used the CUDA version 10.0), 
and then install other dependencies:
```shell script
conda activate DL
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install nibabel statsmodels visdom jsonpatch dominate scikit-image -c conda-forge -c anaconda
```

## Datasets

You can put your images in any folder, but to run this model, 
we assume the format of file names in the pattern like {PREFIX}\_{PATIENT-ID}\_{TIMEPOINT-ID}\_{MODALITY(MASK)}.{SUFFIX},
e.g. training_01_01_flair.nii, training_03_05_mask1.nii :
```
PREFIX: can be any string
PATIENT-ID: number id of the patient
TIMEPOINT-ID: number id of the timepoint
MODALITY, MASK: t1, flair, t2, pd, mask1, mask2, etc
SUFFIX: nii, nii.gz, etc
```

You need to specify the name of MODALITIES, MASKS and SUFFIX in the configuration.py under the root folder.

## Data Conversion

This data conversion is to convert the different kinds of datasets into the same structure. 
To illustrate the process of training and validation, we use the ISBI-2015 challenge data as an example.
The steps are as follows:
1. Copy all of the preprocessed training data of ISBI challenge dataset into the target dataset folder. 
The folder is ~/Documents/Datasets/sample_dataset
2. Modify the configuration file (configuration.py). The positions that need attention are marked with TODO. 
    * The dataset folder, e.g. PATH_DATASET = os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets', 'sample_dataset').
    * The modalities available with this dataset, e.g. MODALITIES = ['t1', 'flair', 't2', 'pd'].
    * The delineations available with this dataset, e.g. MASKS = ['mask1', 'mask2'].
    * The suffix of the files, usually 'nii' or 'nii.gz'
    * (This is not necessary for now) The axis corresponding to axial, sagittal and coronal, respectively. 
    For the ISBI dataset, it is [2, 0, 1].
3. Rename all the files to comply with the pattern as mentioned in the Dataset section. For example, since we use 't1'
in the MODALITIES, simply rename 'training01_01_mprage_pp.nii' to 'training_01_01_t1.nii'. 
The following commands that might be useful:
    ```shell script
    rename 's/training/training_/g' *; rename 's/_pp.nii/.nii/g' *; rename 's/mprage/t1/g' *; 
    ```
4. Run the data\_conversion file. This function will move your files under the sample_dataset folder
into its subfolder ./raw and generate two JSON files (ids.json, properties.json). 

   The ids.json contains the new data paths based on PATIENT-ID, TIMEPOINT-ID, MODALITY or MASK.
   It is needed for cross-validation to avoid splitting the scan from the same patient into 
   both Train and Val (or Test) folder. An example is
   ``` 
   "1": {
    "1": {
      "modalities": {
        "t1": "~/Documents/Datasets/sample_dataset/raw/training_01_01_t1.nii",
        "flair": "~/Documents/Datasets/sample_dataset/raw/training_01_01_flair.nii",
        "t2": "~/Documents/Datasets/sample_dataset/raw/training_01_01_t2.nii",
        "pd": "~/Documents/Datasets/sample_dataset/raw/training_01_01_pd.nii"
      },
      "mask": {
        "mask1": "~/Documents/Datasets/sample_dataset/raw/training_01_01_mask1.nii",
        "mask2": "~/Documents/Datasets/sample_dataset/raw/training_01_01_mask2.nii"
      }
    },
    "2": {
    ...
    }
    ...
   }
   ```
   
   The properties.json contains the peak of each modality using kernel density estimation. 
   It is saved in the JSON file so that we don't need to calculate it repeatedly during the training process. An example is
    ```shell script
    "406576b1b92f6740b0e20a29016952ae1fa6c4cf": {
    "path": "~/Documents/Datasets/sample_dataset/raw/training_01_01_t1.nii",
    "peak": 178855.95956321745
    }
    ```

## Training and Validation

Before you run and training and validation, you can decide what kind of cross-validation strategy you want to use.
At default, we use 5-fold cross-validation. 
We implemented three strategies, which can be set using test_mode option: 
* 'val': validation only, no test. For each fold, 4/5 of data for training and 1/5 of data for validation.
The models with the best dice score will be saved as 'latest' models. 5 'latest' models will be preserved.
The training will stop after 160 epoch of no improvement or the fixed number of epochs as provided in the training options, whichever comes first. 
Since all the ISBI data we placed into sample\_dataset are from the training dataset 
(which means the test data will be provided separately), we use this mode. 
* 'test': test only, no validation. The fold can be set using test_index option. Only one model will be generated. 
The program will not automatically save the 'latest' model because the model performance is determined using the validation set.
The program will run fixed epochs and stop.
* 'val_test': do both validation and test. The test fold is set to the last fold at default (1/5 of data) 
and will not be seen by the model during the training/validation process.
For the remaining 4/5 of data, training takes 3/5 and validation takes 1/5. 
In such a way, 4 'latest' model will be saved and they will be finally tested with the hold test set.

The visdom is needed to run before the training function if you want to visualize the results
```shell script
conda activate DL
python -m visdom.server
```

An example to run the training is
```shell script
conda activate DL
python train.py --loss_to_use focal --input_nc 3 --trainSize 128 --test_mode val --name experiment_name --eval_val
```
where:
* loss_to_use: you can choose from ['focal', 'l2', 'dice']. You can also use combinations of them, e.g. 'focal_l2'.
If you use more than one loss functions, you can set the weight of each loss by setting lambda\_L2, lambda\_focal and lambda\_dice.
* input_nc: the number of slices in a stack. As mentioned in the paper, using 3 achieved the best results.
* trainSize: for ISBI dataset, the size is 181x217x181. We set the trainSize to be 128 so that the program will crop
the slice during the training process. If you set it to 256, the program will pad the slices.
* name: the experiment name. The checkpoint of this name will be automatically created under the ../Checkpoint folder.
Also, in the test phase, the mask prediction has the same suffix as this value.
*eval_val: use the eval mode for validation during the training process.

_If you got "RuntimeError: unable to open shared memory object", 
use PyCharm to run the code instead of using terminal_

## Testing

In the testing phase, you can create a new folder for the test dataset. 
For example, if you want to use the test set of ISBI challenge, create a new subfolder ./challenge
under sample\_dataset (parallel to train, val, test).
The naming of files does not need to follow the strict pattern described in Dataset and Data Conversion, 
but the files should end with {MODALITY(MASK)}.{SUFFIX}. The reference mask files are not needed.
If the reference mask files are provided, the segmentation metrics will be output.

Example:
```shell script
conda activate DL
python test.py --load_str val0,val1,val2,val3,val4 --input_nc 3 --testSize 256 --name experiment_name --epoch latest --phase challenge --eval
```
where:
* load_str: in the training phase, the lastest models are saved due to cross-validation. 
This string describes the models (folds) you want to load.
With it is set to 'val0,val1,val2,val3,val4', latest models from all the 5 folds are loaded 
(you can check the model names by exploring the checkpoint folder).
* testSize: the default value is 256. The slices will be padded to this size in the inference stage. 
If you have any dimension larger than 256, simply change it to a number that larger than all the dimension sizes.
* epoch: the default value is latest. You can set it to a specific number, but it is not recommended.
* phase: the new subfolder of the test set.

You can find the options files under the option folder to get more flexibility of running the experiments.
