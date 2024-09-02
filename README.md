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
git clone https://github.com/akshaysanil/detect_classify_yolov8.git
```

2. **Navigate to the project directory:**

```bash
cd detect_classify_yolov8
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
streamlit run app.py
```
For augmentation, use:

```bash
python augmentation.py
```

### Project Document
For more, refer - [Gym_band_detection_and_color_identification_report ](https://github.com/akshaysanil/detect_classify_yolov8/blob/master/Gym%20Band%20Detection%20and%20Color%20Identification%20Project.pdf)