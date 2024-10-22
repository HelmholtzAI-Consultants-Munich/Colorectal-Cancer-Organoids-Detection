# Colorectal Cancer Organoids Detection

This reposotory is a Napari based tool that allows to annotate organoids in brightfiled microscopy images with bounding boxes. To make the annotation process fastest we implemented an AI-assisted labelling system which uses the predictions of [GOAT](https://github.com/msknorr/goat-public) as a baseline, lets the annotator correct them as needed, and stores the corrected annotations.
The program requires in input only the path to the dataset folder containig the images to annotate. This dataset is required to have a specific structure: the images need to be stored in a subfolder named `images`, while the annotations will be stored in a speparate subfolder named `annotations`. The images can be stored in a structured manner separating for example images coming form different patients and/or different treatments, this structured will be automatically replicated in the `annotations` folder.

```
├── dataset name
            ├── images
                    ├── subfolder_1
                    ├── subfolder_2
                    ├── ...
            ├── annotations
                    ├── subfolder_1
                    ├── subfolder_2
                    ├── ...
                
```

In addition, the program wil keep track of the images that have been annotated, this way every time the program is stopped and started again it will restart where the user left off during the last session.
Finally, it is also  possible to review the annotations made and to apply further corrections.

An analogous framowork has been developed to annotate fibroblasts on the images. However, in this case the annotation consists in poits centered inthe nuclei of the fibroblast and there is no model that prides a baselin for the annotations.

## Installation

1. Clone the repository on your local machine.
    ```shell
    git clone https://github.com/HelmholtzAI-Consultants-Munich/Colorectal-Cancer-Organoids-Detection.git
    ```
2. Install minocoda on you machine, you can find the offical installation instructions [here](https://docs.anaconda.com/miniconda/miniconda-install/).
3. Create a new conda environment:
    ```shell
    conda create -n orga python=3.12
    ``` 
4. Activate the newly created environment
    ```shell
    conda activate orga
    ``` 
5. Navigate to the project directory in your terminal.
6. Install the required dependencies by running in you terminal the folling command:
    ```shell
    pip install -e .
    ```

7. Download the GOAT weights from [here](https://drive.google.com/file/d/1AcrYCBR5-kg91C61boj221t1X_SVX8Hv/view) and save them in the `model` folder without renaming it.

## Usage Instructions

To use the tool, first thing open the terminal and activate the conda environment (`conda activate orga`). The package contains different tools that can be used to perform the following functions:

1. Annotate a given dataset of images.
2. Merge the annotations genereted by two different annotators.
3. Correct the merged annotation.

Next, we describe in details what these tools do and how to use them, for more detailed information regarding the usage you can always refer to their help function:
```shell
<command_name> --help
```

### Image Annotation

Given a dataset of microscopy images, these tools allow to annotate them with the aid of a GUI based on Napari. The user can generate bounding box annotations of the organoids and point annotations for the fibroblasts. When annotation organoids each image will be presented to the user with the predictions made by GOAT, which can be used as a baseline for the annotation. For clarity the baseline boxes are colored in <code style="color : blue">**blue**</code>, while the bpoxes manually added by the user are represented in <code style="color : magenta">**magenta**</code>.

1. **Annotate:** 

    To run the annotation pipeline write the following command in the terminal and replace `dataset_path` with the actiual path on you machine. For annotating organoids run:
    ```shell
    annotate_organoids -d=dataset_path
    ```
    while for annotating fibrobalsts run:
    ```shell
    annotate_fibroblasts -d=dataset_path
    ```
    During the execution type "**s**" to save the annotation and go to the next image, and type "**e**" to end the program.

2. **Review**:

    To revie the images previously annotated the commands are similar to above, for organoids annotations run:
    ```shell
    annotate_organoids -d=dataset_path -r
    ```
    and for fibroblasts annotations run:
    ```shell
    annotate_fibroblasts -d=dataset_path -r
    ```

    As before, during the execution type "**s**" to save the annotation and go to the next image, and type "**e**" to end the program.

### Merging Multiple Annotations

If two different annotators independelty annotated the same dataset it is possible to merge the two annotations. The tool uses the Hungarian algorithm to match boxes form the two annotators that have maximal Intesection over Union (IoU) score nad merges the "matching" boxes by averaging the edges. In additon, it is possible to set a minimum threshold for the IoU score below which two boxes are not matched by the algorithm (```-iou``` parameter). Finally, the user can decide whether to include or to discard the unmatched bounding boxes in the final annotation (```--keep``` or ```--drop``` flags).

To run the tool write one of the following command in the terminal depending if you want to keep or discard unamtched boxes, and replace the dataset paths and the iou threshold with the desidered ones:
- **keep** the unmatched boxes: ```merge_annotations -a1=annotator_1_dataset_path -a2=annotator_1_dataset_path -o=output_dataset_path -iou=  --keep```
- **discard** the unmatched boxes: ```merge_annotations -a1=annotator_1_dataset_path -a2=annotator_1_dataset_path -o=output_dataset_path -iou=  --drop```

**Remark**: this tool does not support fibroblasts annotations

### Correct Merged Annotations

After the annotations have been merged, it is posible to undergo a second round to further manually correct the annotations. This tool allows to navigate the dataset generated with ```merge_annotations``` and permorm the necessary manual corrections adn consists in a GUI based on Napari. Similar to above, the boxes are colored to increase the clarity: the matched boxes are represented in <code style="color : green">**green**</code>, the unmatched boxed are represented in <code style="color : red">**red**</code> , and the new manually added boxes in <code style="color : magenta">**magenta**</code>.

To run this tool write the following command in the terminal:

```shell
annotate_organoids -d=dataset_path
```

To review the previously annotaed images just add the ```-r``` flag as above.

**Remark**: this tool does not support fibroblasts annotations

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License.
