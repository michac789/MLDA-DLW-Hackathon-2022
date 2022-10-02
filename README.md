# MLDA DLW Hackathon 2022

![title_logo](assets/title_logo.png?raw=true)

## Introduction

This repository is a submission for MLDA Deep Learning Week Hackathon 2022 held on 30 September 2022 - 3 October 2022.

This project is created by:

- Dhairya Rungta [(@dhairyarungta)](https://github.com/dhairyarungta)
- Joshua Adrian Cahyono [(@JvThunder)](https://github.com/JvThunder)
- Karan Andrade [(@kj-andrade)](https://github.com/kj-andrade)
- Michael Andrew Chan [(@michac789)](https://github.com/michac789)

## Project Overview

### General Description

`FIX6DSENSE.AI` is an AI-Powered real-time virtual assistant that enables the visually impaired to see the world by hearing.

### Features

- Detecting 80 different categorical object types with their confidence level
- Priority-based detection
- Distance prediction & scaling algorithm
- Mobile phone activated camera for various items
- Speech-activated mode selection
- Multi-threading usage for asynchronous voice warnings and feedbacks

### Modes

- Aware mode: continuously speak out loud 3 items with the topmost priority
- Warn mode: only shout out warnings if any item is too close based on priority
- Search mode: focus on searching a particular item until it is found

## Resources Links

Please click the link below to view our other resources regarding to this project submission.

1. [Submission Website](https://devpost.com/software/fix6dsense-ai)
2. [Demonstration Video](TODO)
3. [Presentation Slides](TODO)

## How To Run

1. Clone this repository and make sure you have all the files downloaded

2. Set your environment and install the required python packages

    It is recommended for you to use Anaconda, and specifically `python version 3.9.0` (newer version of python has some compatibility issues related to the `collections` package, other older python version is untested and might have some compatibility issues as well). Then, install all the required python packages by typing:

    ```powershell
        pip install -r requirements.txt
    ```

3. Execute `image_detection.py` file

    If you want to use an external camera, you can connect via IP Webcam Pro and edit the `url` variable when running the model. By default, it is connected to your PC / Laptop default camera. You can also toggle the `USE_SPEECH` global variable to `True` or `False`. If it is `True`, it will use speech recognition to choose and change the mode, or else it will manually prompt you for the mode input.
