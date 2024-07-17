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

## Usage

To use the tool open the terminal and activate the conda environment (`conda activate orga`), then use on of the following commands:

1. **Annotation:** 

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
    review_organoids -d=dataset_path
    ```
    and for fibroblasts annotations run:
     ```shell
    review_fibroblasts -d=dataset_path
    ```

    As before, during the execution type "**s**" to save the annotation and go to the next image, and type "**e**" to end the program.


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License.
