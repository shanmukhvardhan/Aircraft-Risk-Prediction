# 🛸 Aircraft Predictive Maintenance: An XAI-Powered Digital Twin

## 📌 Project Overview
I developed this project to bridge the gap between "Black Box" machine learning and safety-critical aviation operations. This is a real-time **Digital Twin** cockpit simulation that predicts engine risk using a Random Forest model and explains those predictions using **LIME** and **SHAP**. 

By integrating **Polars** for high-speed data engineering and **Great Expectations** for data integrity, I've built a pipeline that isn't just accurate, but also robust and transparent for pilots and engineers.

---

## 🏗️ Project Architecture & Workflow

### 1. Data Engineering: The Rust-Powered Speed
I started by generating 4,000+ records of synthetic flight telemetry. To handle this efficiently, I chose **Polars** over Pandas.
* **Why:** Polars is written in Rust and uses parallel execution. When dealing with high-frequency sensor data, every millisecond counts.
* **Concept:** I used Polars to clean, transform, and prepare the dataset, ensuring the pipeline can scale to massive telemetry streams.



### 2. Data Quality: Great Expectations (GE)
In aviation, "garbage in, garbage out" can be fatal. I implemented **Great Expectations** to "unit test" my data.
* **What I did:** I created a suite of validations to ensure `Engine Vibration` stayed within physical limits (0-6 mm/s) and `RPM` was always positive. 
* **Outcome:** This ensures that the model never makes a prediction based on faulty or corrupted sensor data.

### 3. The Model Evolution: Solving Bias
The core of the intelligence layer went through a significant evolution during my development process:
* **The Starting Point:** I initially used **Logistic Regression**. While the accuracy was decent, I noticed the model was **biased**. It struggled with non-linear relationships (e.g., high RPM during Takeoff is normal, but high RPM during Taxi is a risk).
* **The Solution:** I pivoted to a **Random Forest Classifier**. This ensemble method handled the complex feature interactions much better. I wrapped it in a **Scikit-Learn Pipeline** with a `StandardScaler` to ensure the model remains generalized and stable.

### 4. Explainable AI (XAI): SHAP vs. LIME
I believe a pilot should never be told "Engine Risk" without a reason. I implemented two XAI methods to solve this:
* **SHAP (Global Insights):** I used SHAP in the research phase (`aircraft_vibration_xai.ipynb`) to see how each feature impacts the model globally. 
* **LIME (Real-Time Explanation):** For the live dashboard, I chose **LIME**. It’s much faster than SHAP, allowing the system to provide an instant "Pilot Advisory" (e.g., "Critical: Vibration Hike detected, check engine bearings").



---

## 💻 Tech Stack I Used
* **Frontend:** Streamlit (For the Digital Twin Cockpit)
* **Backend:** FastAPI (To serve the model as a production API)
* **Data Engine:** Polars
* **Validation:** Great Expectations
* **ML Framework:** Scikit-Learn
* **XAI:** SHAP & LIME
* **Visualization:** Plotly (Real-time Gauges)

---

## 🚀 How I Built the Simulation
I didn't want a static dashboard, so I built a **Mission Phase Simulator**:
1.  **Phase Logic:** The sensors (RPM, Temp, Vibration) change behavior based on whether you are in **Taxi, Takeoff, Cruise, or Approach**.
2.  **Anomaly Injector:** I wrote a script to randomly "spike" vibration or temperature. This allows the user to see exactly how the AI reacts to an engine failure.
3.  **Sensor Failure Mode:** I added an integrity check. If a sensor fails (returns `NULL`), the UI displays a "Sensor Bus Failure" warning, simulating a real hardware fault.



---

## 🔄 Alternatives & Future Scope
While I am proud of the current build, there are always other paths:
* **Model:** I could have used **LSTMs** (Long Short-Term Memory) for time-series, but they are harder to explain to a pilot than Random Forests.
* **XAI:** **SHAP** is more mathematically sound but was too slow for a real-time 1Hz dashboard update, which is why I stuck with **LIME** for the UI.

---

## 🛠️ Installation
1.  Clone the repo.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Start the API: `python api.py`.
4.  Launch the Cockpit: `streamlit run simulation.py`.

---

##Live Demo : https://aircraft-risk-prediction-nzn76paph4pmtauvpummlr.streamlit.app/
