# iProphIT
<p align="center">
  <img src="https://github.com/user-attachments/assets/c7aa2e32-0611-477b-bc9c-1cc821b67545" alt="iProphIT Logo">
</p>
<p align="center">
  <strong>A deep learning approach that identifies the inducible activity of prophages from their DNA sequences.</strong>
</p>

## Requirements

System and software requirements:

- **Linux**
- **Python 3.x**    *(Any Python version compatible with PyTorch，Tested with Python 3.12.)*
- **biopython**
- **numpy**
- **pandas**
- **tqdm**
- **urllib3**
- **pytorch**    *(If you want to enable GPU acceleration, please install the appropriate GPU-enabled PyTorch version from the official PyTorch website.)*

## Installation

**1.** You need to download **`classifier.py`** and **`iProphIT_model-v1.pth`** into your working directory.  
(download the model weight file **`iProphIT_model-v1.pth`** from Zenodo (https://doi.org/10.5281/zenodo.21457317), or use the command-line argument **`--download_model`** for automatic download.)   

**2.** Download tool

**Prerequisites: Create Conda Environment**  
Create a conda environment and install required packages:
```bash
conda create -n iprophit python=3.12
conda activate iprophit
conda install -c conda-forge biopython numpy tqdm pandas urllib3
conda install pytorch
```

**Method 1: Manual Installation**  
```bash
git clone https://github.com/HongboZhang-z3d/iProphIT-FAFU.git
```

**Method 2: Bioconda Installation**
```bash
conda install -c bioconda iprophit
```
```bash
#or use mamba for faster installation:
mamba install -c bioconda iprophit
```


## Run iProphIT
### Method 1: To run iProphIT via GitHub download, follow these steps:
**1.** Download **`classifier.py`**， **`iProphIT_model-v1.pth`** and put them in your working path.   

**2.** Run **`classifier.py`**

```bash
python classifier.py -i test_iProphIT.fasta -m iProphIT_model-v1.pth -o ./Result.tsv -t 16
```


### Method2: To install via Bioconda, run the following command directly:
```bash
iprophit -i test_iProphIT.fasta -m iProphIT_model-v1.pth -o ./Result.tsv -t 16
```


## Usage

```bash
usage: iprophit [-h] [-i INPUT] [-m MODEL] [-o OUTPUT] [-t THREADS] [-b BATCH_SIZE] [--download_model DIR]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input FASTA file (required for prediction)
  -m MODEL, --model MODEL
                        Path to the trained model file (default: ./iProphIT_model-v1.pth)
  -o OUTPUT, --output OUTPUT
                        Output TSV file path (default: ./Result.tsv)
  -t THREADS, --threads THREADS
                        Number of CPU threads for DataLoader (default: 4)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for prediction (default: 4). Larger values accelerate inference.
                        GPU: Increase for speedup until CUDA OOM, then reduce.
                        CPU: Can use larger values due to more RAM.
  --download_model DIR  Download the pre-trained iProphIT model from Zenodo to the specified directory.
                        Directory will be created if it does not exist. After downloading, the script exits.

```

## Typical output
- Find result in Result.tsv  

  | ID | Predict | Confidence |
  |----------|:--------:|:--------:|
  | prophage1     | active  | 0.9981  |
  | prophage2  | dormant   | 0.9221  |

- Explanation  
1.**`ID`** is the content of the description line in the genome file.  
2.**`Predict`** is the result of identification (**`active`**->inducible prophage, **`dormant`**->non-inducible prophage).

## Using testing data
genome file: **`OY731326.1`** and **`OY731419.1`**,   
source: Dahlman S. et al., Nature (2025), https://doi.org/10.1038/s41586-025-09614-7  
  
- Run **`iProphIT-classifier.py`**

```bash
iprophit -i test_iProphIT.fasta -m iProphIT_model-v1.pth -o ./Result.tsv -t 16
```

- Output of the test
```bash
ID	Predict	Confidence
OY731326.1	active	0.9989
OY731419.1	active	0.9927
```

## Notes
- Input can accept genome files in formats such as `.fasta`, `.fa`, `.fna`, etc.
- iProphIT will automatically use the GPU if available, as long as you have installed a `PyTorch` version with CUDA support. 
## Copyright
Hongbo Zhang, Chen Liu, Hanpeng Liao, Fujian Provincial Key Laboratory of Soil Environmental Health and Regulation, College of Resources and Environment, Fujian Agriculture and Forestry University, Fuzhou, 350002, China.
