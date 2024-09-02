# pytorch_template
[Original Template](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template)
## Installation
1. Make a new repository
[Click this](https://github.com/new?owner=4ILab-SSU&template_name=pytorch-template&template_owner=4ILab-SSU)
or
Click "Use this template"
or
    ```sh
    gh repo create (repo_name) [--public | --private] -p https://github.com/4ILab-SSU/pytorch-template.git
    ```
1. Clone the Repository
2. Make Anaconda3 environment and execute
    ```sh
    conda create -n (env_name) python==3.8.0
    conda activate (env_name)
    pip install -r requirements.txt
    ```

## Structure
```
.
├── callbacks // here you can create your custom callbacks
├── checkpoint // were we store the trained models
├── data // here we define our dataset
│ └── transformation // custom transformation, e.g. resize and data augmentation
├── logger.py // were we define our logger
├── losses // custom losses
├── main.py
├── models // here we create our models
│ └── utils.py
├── Project.py // a class that represents the project structure
├── README.md
├── requirements.txt
└── utils.py // utilities functions
```