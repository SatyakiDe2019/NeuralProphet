# Restore old video with Python

## About this app

This app will perform training a model using Neural Prophet to predict IoT Time series analysis with better accuracy. This is an improvement over Facebook's Prophet API.


## How to run this app

(The following instructions apply to Posix/bash. Windows users should check
[here](https://docs.python.org/3/library/venv.html).)

First, clone this repository and open a terminal inside the root folder.

Create and activate a new virtual environment (recommended) by running
the following:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the model training for Neural-Prophet Forecast:

```bash
python readingIoT.py
```
This will generate the model by consuming historical IoT data & then performing various KPIs.

## Screenshots

![demo.GIF](demo.GIF)

## Resources

- To learn more about Open-CV, check out our [documentation](https://opencv.org/opencv-free-course/).
- To learn more about Matplotlib, check out our [documentation](https://matplotlib.org/stable/contents.html).
- To learn more about Neural-Prophet, check out our [documentation](https://neuralprophet.com/html/quickstart.html).
