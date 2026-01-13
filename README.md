A real-time person detection system built using Classical Computer Vision techniques in Python and OpenCV.

    Project Overview
This project demonstrates the application of fundamental image processing algorithms to solve industrial safety challenges. It detects workers in the blind spots of machinery without requiring high-end GPU hardware.

    Technical Analysis: Classical (MOG2) vs. Deep Learning
This system is intentionally built using MOG2 (Gaussian Mixture-based Background/Foreground Segmentation) to meet specific requirements. However, it is essential to understand the limitations of this pixel-based approach compared to modern Deep Learning (DL) detectors like YOLO.

    The Limitations of MOG2 (The "Downfall")
Motion Dependency: MOG2 detects pixel changes, not people. If a worker stands perfectly still, they are absorbed into the background model and "disappear." DL models detect features (limbs, heads), so they see static people.

Ego-Motion Sensitivity: If the machine (camera) moves, the entire background shifts. MOG2 interprets this global change as motion, triggering massive false positives. DL models are generally invariant to background shifts.

Semantic Blindness: To MOG2, a falling box and a walking human are both just "white blobs." It lacks the semantic understanding to classify what the object is.

    Our Mitigation Strategy
Since we cannot rely on semantic understanding, we implemented a robust Geometric Filtering Pipeline to distinguish humans from noise:

Solidity Filter: Calculates Area / (Width * Height). Humans are "solid" shapes; noise/reflections are often hollow or scattered.

Aspect Ratio Constraint: Rejects horizontal objects (like pallets). Valid detections must be vertical (Height > Width).

Dynamic ROI: Automatically ignores motion in the upper half of the frame (e.g., high shelves), focusing on the floor level.

    Processing Pipeline
The code follows a strict 5-step pipeline:

Pre-processing: Gaussian Blur to reduce sensor noise.

Motion Detection: MOG2 with a high threshold (250) to ignore vibrations.

Morphology: Aggressive Erosion (7x7 kernel) to remove debris, followed by Dilation to merge body parts.

Segmentation: Connected Components Analysis to extract metrics.

Logic Filtering: Applying the geometric rules described above.

    Installation & Usage
Prerequisites
* Python 3.x
* A webcam or a test video file (already installed)

Quick Start
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/Forklift-Safety-Vision.git](https://github.com/yourusername/Forklift-Safety-Vision.git)
    cd Forklift-Safety-Vision
    ```

2.  **Create and activate a virtual environment (Recommended):**
    * *Windows:*
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    * *Mac/Linux:*
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the system:**
    ```bash
    python main.py
    ```
    *Note: If no video file is found in `data/`, the application will automatically open your webcam.*
    
      Academic Context
This project was developed for the Computer Vision course, demonstrating mastery of fundamental algorithms independently of "Black Box" AI libraries.
