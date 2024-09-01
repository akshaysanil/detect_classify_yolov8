# OBJECT DETECTION AND CLASSIFICATION

## Overview

Detect Gym equipment Bands and identiy the color

## Table of Contents


1. [Installation](#installation)
2. [Running](#running)




## Installation

Follow these steps to install and set up the project:

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/your-repository.git
```

2. **Navigate to the project directory:**

```bash
cd image_detect_identify_color
```

3. **Set up the environment:**

If you are using a conda environment:
```bash
conda create -n py39band python=3.9 -y
```
If you are using a python environment:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use `venv\Scripts\activate`
```

4. **Install dependencies:**

Make sure you have a `requirements.txt` file with all the necessary packages. Then:

```bash
pip install -r requirements.txt
```

### Running 

To run the project, use:

```bash
python app.py
```
For augmentation, use:

```bash
python augmentation.py