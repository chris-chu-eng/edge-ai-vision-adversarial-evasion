# Edge AI Vision System & Adversarial Evasion Study

This project is a real-time computer vision pipeline designed to test the robustness of modern object detection systems. The application establishes a "naked eye" baseline using YOLOv8 to later demonstrate how standard AI models can be defeated by physical adversarial attacks.

This project was built to demonstrate proficiency in computer vision pipelines, edge-optimization, and defense-oriented systems engineering.

---

## Features
- **Real-Time Inference**: Uses the YOLOv8-Nano architecture to prioritize low-latency performance, mimicking the constraints of edge hardware (drones/sentries).
- **Live HUD**: A custom visualization layer built with OpenCV/CvZone that renders bounding boxes, object labels, and confidence scores in real-time.
- **Logic Filtering**: Implements a custom confidence threshold (>0.45) to filter noise and strictly track "Blue Force" targets (Humans).
- **Modular Pipeline**: The codebase (`main.py`) separates hardware initialization, inference, and rendering logic for clean maintainability.

---

## How to Run

### Prerequisites
- Python 3.10+
- A standard USB Webcam or Laptop Camera.

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chris-chu-eng/edge-ai-vision-adversarial-evasion.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd edge-ai-vision-adversarial-evasion
    ```
3.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    ```
    - **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    - **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution
- To run the program:
    ```bash
    python main.py
    ```
    *(Note: AI model weights will auto-download on first run)*

---

## Project Roadmap
- [x] **Version 1: Visible Spectrum Tracker** (Current Baseline).
- [ ] **Version 2: Multi-Spectral Simulation** (Implementing synthetic Thermal/IR view).
- [ ] **Version 3: Adversarial Attack** (Testing system robustness against physical evasion patches & more).

---

## Technologies Used
- **Core**: Python, NumPy
- **Inference Engine**: Ultralytics YOLOv8 (Nano)
- **Computer Vision**: OpenCV, CvZone
- **Version Control**: Git / GitHub
