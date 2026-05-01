# Colorful Fruit Detection and Counting System

A computer vision based fruit detection and counting system developed using Python, OpenCV, NumPy, and Gradio. The system detects fruits based on color segmentation in HSV color space and counts individual fruits from uploaded images.

## Features
- Detects fruits using HSV color segmentation
- Supports multiple fruit colors:
  - Red
  - Green
  - Yellow
  - Blue
- Applies morphological operations for noise reduction
- Uses contour detection and connected component analysis
- Handles overlapping fruits using:
  - Distance Transform
  - Watershed Segmentation
  - K-Means Clustering
- Displays intermediate processing steps
- Interactive Gradio-based web interface

## Technologies Used
- Python
- OpenCV
- NumPy
- Gradio
- Matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
```

Install required packages:

```bash
pip install opencv-python numpy matplotlib gradio
```

## Run the Project

```bash
python image_final_project.py
```

The application will open automatically in your browser.

## Project Workflow
1. Upload image
2. Convert image to HSV color space
3. Generate color masks
4. Apply preprocessing and hole filling
5. Detect and separate overlapping fruits
6. Count and label detected fruits
7. Display annotated output and intermediate processing results

## Screenshots
Project  input images are available in the `imagelab` folder.
Project screenshots and output images are available in the `output` folder.

## Author
Sanzida Moin Tithi
```
