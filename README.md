# ParkiTrace — AI-assisted handwriting screening research prototype

ParkiTrace is a **research/academic prototype** that classifies handwriting images into two classes (`healthy` / `parkinson`) using an EfficientNetV2-S + Vision Transformer feature-fusion model and visualizes the result with Grad-CAM.

> **Medical disclaimer:** This project is not a medical device and does not diagnose Parkinson's disease. Results are for research/demo purposes and must not be used as a substitute for a qualified clinician.

## 1. Recommended Windows setup

Use **Python 3.11** for the smoothest PyTorch/torchvision compatibility.

1. Install Python 3.11 from https://www.python.org/downloads/
2. During installation, tick **Add Python to PATH**.
3. Extract this project, for example to `C:\parkitrace`.
4. Open the project folder, click the address bar, type `cmd`, and press Enter.

Create and activate the virtual environment:

```bat
python -m venv venv
venv\Scripts\activate
```

Upgrade pip:

```bat
python -m pip install --upgrade pip
```

Install dependencies:

```bat
pip install -r requirements.txt
```

## 2. Test the complete pipeline without medical data

Run:

```bat
python make_demo_data.py
```

This creates **synthetic drawings only**. They are not real patient data and must not be presented as clinical results.

You should see:

```text
dataset\healthy\
dataset\parkinson\
```

## 3. Train the model

For a normal laptop/CPU, start with:

```bat
python train.py --data dataset --epochs 5 --freeze-backbones
```

For better training when you have a GPU:

```bat
python train.py --data dataset --epochs 15
```

The first pretrained run downloads ImageNet weights. An internet connection is required for that download.

After training, the project creates:

- `model\parkitrace.pt` — trained model
- `model\metrics.json` — accuracy, precision, recall and F1
- `model\confusion_matrix.png` — validation confusion matrix

**Do not invent accuracy numbers for a report.** Use the values in `model\metrics.json` from your actual dataset.

### Real dataset

For academic experiments, use a properly licensed public handwriting dataset such as a Parkinson's spiral/wave dataset. Check the dataset's license and research-use conditions yourself.

Arrange files as:

```text
dataset\
├── healthy\
│   ├── image1.png
│   └── ...
└── parkinson\
    ├── image1.png
    └── ...
```

Keep the classes balanced where possible and document the dataset source, number of participants/images, and train/validation split in your report.

## 4. Run the web application

After training:

```bat
python app.py
```

On the laptop, open:

```text
http://127.0.0.1:5000
```

Upload a drawing, choose **Analyse handwriting**, and the application displays:

- predicted class
- model confidence
- risk band
- original handwriting
- Grad-CAM visualization
- downloadable PDF report

## 5. Use the phone camera

1. Connect laptop and phone to the same Wi-Fi.
2. In the project Command Prompt, run:

```bat
ipconfig
```

3. Find the laptop's **IPv4 Address** under the active Wi-Fi adapter.
4. Start the app:

```bat
python app.py
```

5. On the phone, open:

```text
http://YOUR_IPV4_ADDRESS:5000
```

Example:

```text
http://192.168.1.20:5000
```

6. Tap **Take a photo** and capture the drawing.

If Windows Firewall asks whether Python should communicate on the network, allow it on **Private networks** if this is your trusted home/college network.

## 6. Next time

You do not need to train again unless you change the dataset or model.

```bat
cd C:\parkitrace
venv\Scripts\activate
python app.py
```

## Troubleshooting

### `python is not recognized`
Reinstall Python 3.11 and select **Add Python to PATH**. You can also try:

```bat
py --version
```

If `py` works, create the environment with:

```bat
py -3.11 -m venv venv
```

### `No trained model found`
Run training first:

```bat
python train.py --data dataset --epochs 5 --freeze-backbones
```

### `Expected exactly these dataset folders`
The dataset must contain exactly:

```text
dataset\healthy
dataset\parkinson
```

### Training is too slow
Use a GPU/Google Colab, reduce the number of epochs, and use:

```bat
python train.py --data dataset --epochs 5 --freeze-backbones
```

### Phone cannot open the website
Check:

- phone and laptop are on the same Wi-Fi
- `python app.py` is still running
- you used the laptop's current IPv4 address
- Windows Firewall allows Python on the private network
- your Wi-Fi is not using client/AP isolation

## Project structure

```text
parkitrace/
├── app.py
├── make_demo_data.py
├── parkitrace_model.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   └── .gitkeep
└── templates/
    └── index.html
```

## Suggested academic report sections

1. Introduction
2. Problem statement
3. Motivation
4. Literature survey
5. Dataset and preprocessing
6. Proposed architecture
7. EfficientNetV2 feature extraction
8. Vision Transformer feature extraction
9. Feature fusion and classification
10. Grad-CAM explainability
11. Web application and PDF reporting
12. Experimental results
13. Limitations
14. Future scope
15. Conclusion

For a publication or clinical claim, substantially stronger validation is required than this demonstration pipeline, including appropriate participant-level splits, independent testing, leakage controls, and clinical evaluation.
