# EmbryoML: Federated Learning for IVF Clinics

A federated learning system for embryo viability prediction. This system allows multiple IVF clinics to collaboratively train a machine learning model for embryo classification while keeping their patient data private and secure.

## Features

- **Privacy-First Approach**: All patient data remains at each clinic - only model updates are shared
- **Collaborative Learning**: Multiple clinics can contribute to a single powerful model
- **Real-time Visualization**: Monitor training progress through an interactive UI
- **External Connectivity**: Connect clinics across different networks using ngrok
- **Real Embryo Data Support**: Works with real human embryo images

## System Components

The system consists of several key components:

1. **Core Utilities** (`embryo_fl_utils.py`): Common functions and classes for the federated learning system
2. **Federated Learning Demo** (`federated_embryo_demo.py`): The server and client implementation
3. **Integrated App** (`integrated_embryo_fl_app.py`): A user-friendly Gradio interface
4. **Sample Data Creation** (`create_sample_data.py`): Script to generate test data
5. **Real Data Preparation** (`prepare_real_embryo_data.py`): Scripts to prepare real embryo images

## Setup Instructions

### 1. Install Dependencies

```bash
pip install torch torchvision flwr gradio pyngrok numpy pillow matplotlib
```

### 2. Prepare Data

Either use synthetic data:
```bash
python create_sample_data.py
```

Or use real embryo data:
```bash
python download_real_data.py    # Download real embryo data from Google Drive
python prepare_real_embryo_data.py    # Process the data for binary classification
```

### 3. Run the Integrated App

```bash
python integrated_embryo_fl_app.py
```

This will open a Gradio interface in your browser (typically at http://127.0.0.1:7860).

## Using the App

### For the Host Clinic:

1. **Start Server**:
   - Go to the "Server Control" tab
   - Set the server port (default: 8090)
   - Check "Use ngrok for external connections" to allow other clinics to connect over the internet
   - Click "Start Server"

2. **Share Connection Details**:
   - Copy the connection command shown in the Server Output box
   - Share this command with other participating clinics

3. **Start Local Test Clinics** (Optional):
   - Go to the "IVF Clinic Control" tab
   - Enter the server address (usually 127.0.0.1:8090)
   - Set a Clinic ID (start with 1)
   - Click "Start Local IVF Clinic"
   - Repeat with different Clinic IDs if desired

4. **Monitor Progress**:
   - Go to the "Visualization" tab to see training progress
   - Check "System Status" periodically to see connected clinics and training status

### For Participating Clinics:

1. **Get the Required Files**:
   - `federated_embryo_demo.py`
   - `embryo_fl_utils.py`
   - `create_sample_data.py` (to generate sample data)

2. **Prepare Local Data**:
   - Generate sample data: `python create_sample_data.py`
   - Or use their own embryo images in the required format

3. **Connect to the Server**:
   - Run the command provided by the host clinic:
   ```
   python federated_embryo_demo.py client --server_address=X.ngrok.io:YYYY --client_id=Z
   ```
   - Replace Z with a unique clinic ID (2, 3, 4, etc.)

## How Federated Learning Works

1. **Model Definition**: A CNN model architecture for embryo classification is defined
2. **Server Initialization**: The central server coordinates but never sees patient data
3. **Client Training**: Each clinic trains on their local data
4. **Aggregation**: The server combines all clinic model updates into a global model
5. **Distribution**: The improved global model is sent back to all clinics
6. **Repeat**: This process continues for multiple rounds, improving accuracy

## Acknowledgments

This project is designed to showcase federated learning in a healthcare setting, particularly for IVF clinics where privacy concerns are paramount. The system demonstrates how clinics can leverage collective data insights while respecting patient privacy.

## License

This project is open source and available under the MIT License. 