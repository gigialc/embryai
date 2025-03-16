# EmbryoML Federated Learning Demo

This project demonstrates a federated learning system for embryo image classification. Multiple computers can collaborate to train a shared model while keeping all data local and private.

## Quick Start

1. **Setup**: Run the setup script to install all required dependencies:
   ```
   python setup_embryo_fl.py
   ```

2. **Server Setup**: On one computer (the server), run:
   ```
   python federated_embryo_demo.py server --port=8080
   ```
   Note the IP address displayed, as clients will need to connect to this address.

3. **Client Setup**: On each participating computer, run:
   ```
   python federated_embryo_demo.py client --server_address=<SERVER_IP>:8080 --client_id=<UNIQUE_ID>
   ```
   Replace `<SERVER_IP>` with the IP address of the server and `<UNIQUE_ID>` with a unique identifier for each client (e.g., 1, 2, 3).

## Data Preparation

Each client needs a local dataset of embryo images in the `embryo_data` directory:

- Images should follow the naming convention `*label_0*.png` or `*label_1*.png`
- `label_1` indicates embryo survival
- `label_0` indicates non-survival

## How It Works

1. **Server**: Coordinates the training process and aggregates model updates
2. **Clients**: Train the model on their local data and send model updates (not the data) to the server
3. **Aggregation**: The server combines all client model updates into a global model
4. **Distribution**: The updated global model is sent back to all clients

This approach provides:
- **Privacy**: Raw image data never leaves each computer
- **Collaboration**: The model benefits from diverse data sources
- **Personalization**: Each client can further fine-tune the final model on their own data

## Requirements

- Python 3.6+
- PyTorch
- Flower (flwr)
- Pillow
- NumPy

All dependencies are installed by the setup script.

## Troubleshooting

If you encounter network issues:
- Ensure port 8080 (or your chosen port) is open on the server's firewall
- Make sure all computers are on the same network
- Try using the server's local IP address if connecting within the same network

For data loading issues:
- Check that your images follow the correct naming convention
- Ensure the `embryo_data` directory contains valid image files

## Advanced Options

- Change the port by modifying the `--port` argument on the server
- Adjust the number of training rounds by changing the `num_rounds` parameter in the `start_server` function
- Modify the model architecture by editing the `Net` class in the code 