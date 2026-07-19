# AI-Powered Smart Parking System & Virtual Kiosk

This repository contains an intelligent system developed for the automated management of a parking lot[cite: 1]. The primary objective is to monitor parking space availability in real-time and digitize the payment process through virtual ticketing[cite: 1].

<div align="center">
  <img src="assets/kiosk-main.png" width="400" alt="LabVIEW Main UI" />
  <img src="assets/kiosk-checkout.png" width="400" alt="LabVIEW Checkout UI" />
</div>

## System Architecture
To guarantee system stability and prevent heavy mathematical processing from freezing the user's checkout screen, the software tasks are divided into three distinct technological areas[cite: 1]:

* **Computer Vision:** Utilizes Python and the YOLO artificial intelligence model to detect vehicles via cameras[cite: 1].
* **Industrial Interface:** Utilizes LabVIEW as the point of sale (POS) where the user interacts, views available spaces, and calculates fees[cite: 1].
* **Connectivity & Cloud:** Integrates network tools so the local system can back up data to the internet and clients can access their tickets from mobile devices[cite: 1].

## API Implementation
Because Python, LabVIEW, and mobile devices operate independently, a local API was built using a Flask server to act as a communication bridge[cite: 1]. 

* **Internal Communication:** Allows the LabVIEW interface to constantly query the Python engine for parking availability[cite: 1]. The API takes the coordinates of the vehicles detected by the camera and delivers them to LabVIEW packaged in a JSON format[cite: 1].
* **External Communication:** Saves ticket information automatically to a Google Sheets database[cite: 1]. It also receives requests from clients' cell phones when they scan the QR code, delivering a dynamic virtual ticket and managing email dispatches[cite: 1].

## Data Flow & Operational Phases

The lifecycle of the data is strictly divided into 8 operational phases:

### Phase 1: Environment Initialization
The data flow begins in the LabVIEW main panel[cite: 1]. Pressing the start button executes an automatic CMD command (`cmd /c python ... Deteccion_Estacionamiento2.py`) that launches the Python server in the background, establishing bidirectional communication and placing the system into a While Loop[cite: 1].

### Phase 2: Visual Mapping & Interface
The YOLO AI model processes the video frames and transmits the resulting coordinates to LabVIEW[cite: 1]. The main interface renders the real-time video stream, superimposing bounding boxes over each detected vehicle to confirm the vision engine's operation[cite: 1]. The interface features an "IMPRIMIR TICKET" (Print Ticket) button that triggers the database interaction[cite: 1].

### Phase 3: Data Acquisition
The AI continuously updates the coordinates in a local JSON file[cite: 1]. LabVIEW executes a `For` loop that continuously reads this file to extract positions and display the detected vehicles on an indicator[cite: 1].

### Phase 4: Entry Registration & Ticket Generation
When a new user requires access, selecting the "Imprimir Ticket" option triggers two simultaneous events[cite: 1]. LabVIEW visually generates a QR code with a unique ID[cite: 1]. Parallelly, the API sends a data packet to Google Cloud, where a Google Sheets database registers the ticket ID, captures the exact arrival timestamp, and assigns an initial "PENDIENTE" (Pending) status[cite: 1].

### Phase 5: Web Portal & Mobile Experience
The user scans the printed QR code using their mobile device camera[cite: 1]. Through an `ngrok` network tunnel, the phone securely connects to the local server[cite: 1]. The system delivers a dynamic webpage displaying the virtual ticket information, including the entry time and identifier[cite: 1].

### Phase 6: Cloud Notification System
LabVIEW crosses the coordinates of each detected vehicle with the geometric area of each parking space using two nested `For` loops[cite: 1]. If an intersection occurs (at least one vehicle inside a space), the spot is marked in red (occupied) on the interface's aerial view[cite: 1]. 
Within the ticket web portal, users can input their email to request a map of available spaces[cite: 1]. The Flask API receives this parameter via an HTTP POST request and executes a Google Cloud script that automatically emails the availability map to the user's inbox[cite: 1].

### Phase 7: Exit Scanning & Optical Reading
For checkout, a secondary LabVIEW panel designed for the toll booth is used[cite: 1]. The user presents their QR code to a physically connected camera[cite: 1]. Using a LabVIEW vision and decoding module, the software reads the QR matrix and extracts the unique ticket identifier[cite: 1].

### Phase 8: Dynamic Calculation & Transaction Closure
With the extracted identifier, the local system queries the Google Sheets database to retrieve the specific client's entry time[cite: 1]. LabVIEW subtracts the registered entry time from the current timestamp to calculate the total elapsed time, displaying the dynamic fee on the screen[cite: 1]. Once payment is confirmed, a final HTTP request overwrites the Google Sheets cell, changing the status from "PENDIENTE" to "PAGADO" (Paid) and closing the data lifecycle[cite: 1].
