# FIX6DSENSE.AI

TODO - PIC HERE

## Introduction

This repository is a submission for MLDA Deep Learning Week Hackathon 2022 on 30 September 2022 - 2 October 2022.

This project is created by:

- Dhairya Rungta
- Joshua Adrian Cahyono
- Karan Andrade
- Michael Andrew Chan

## Project Overview

TODO

## Demonstration Images & Video Link

Click [here](TODO-CREATE LINK HERE) to see the video demonstration.

## How To Run

1. Clone this repository and make sure you have all the files downloaded

2. Set your environment and install the required python packages

    It is recommended for you to use Anaconda, and specifically `python version 3.9.0` (newer version of python has some compatibility issues related to the `collections` package, other older python version is untested and might have some compatibility issues as well). Then, install all the required python packages by typing:

    ```powershell
        pip install -r requirements.txt
    ```

3. Execute `image_detection.py` file

    If you want to use an external camera, you can connect via IP Webcam Pro and edit the `url` variable when running the model. By default, it is connected to your PC / Laptop default camera. You can also toggle the `USE_SPEECH` global variable to `True` or `False`. If it is `True`, it will use speech recognition to choose and change the mode, or else it will manually prompt you for the mode input.
