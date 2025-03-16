#!/usr/bin/env python3
"""
Integrated EmbryoML Federated Learning App

This application combines:
1. The federated learning functionality directly (not through subprocess calls)
2. A Gradio visualization interface
3. Ngrok support for exposing the server to the internet

This allows for a seamless federated learning experience where friends can connect
to your server from anywhere without port forwarding.
"""

import os
import sys
import time
import threading
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import gradio as gr
from typing import Dict, List, Any, Tuple
import socket
import multiprocessing
from pyngrok import ngrok, conf

# Import utilities instead of duplicating
from embryo_fl_utils import (
    get_local_ip, find_image_files, EmbryoDataset, Net,
    DEVICE, DATA_FOLDER, BATCH_SIZE,
    train, test, get_parameters, set_parameters
)

# Global variables
server_process = None
client_processes = {}
ngrok_tunnel = None
server_log_file = "server_output.log"
training_metrics = {
    "rounds": [],
    "loss": [],
    "accuracy": [],
    "clients": {}
}

def setup_ngrok(port):
    """Set up ngrok tunnel for the given port."""
    # Configure ngrok (optional)
    # If you have an auth token, uncomment and add it here
    # conf.get_default().auth_token = "YOUR_AUTHTOKEN"
    
    # Create a new tunnel
    try:
        # Close any existing tunnels
        ngrok.kill()
        
        # Create a new TCP tunnel
        tunnel = ngrok.connect(port, "tcp")
        print(f"Ngrok tunnel established: {tunnel.public_url}")
        
        # Extract the ngrok address
        # Format: tcp://X.ngrok.io:YYYYY -> X.ngrok.io:YYYYY
        public_url = tunnel.public_url.replace("tcp://", "")
        
        return tunnel, public_url
    except Exception as e:
        print(f"Failed to establish ngrok tunnel: {str(e)}")
        return None, f"Ngrok error: {str(e)}"

def close_ngrok():
    """Close any open ngrok tunnels."""
    ngrok.kill()
    print("Ngrok tunnels closed")

def start_server_process(port=8090, use_ngrok=False):
    """Start the federated learning server in a separate process and optionally expose it via ngrok."""
    global server_process, ngrok_tunnel, server_log_file
    
    # Clean up old log file
    if os.path.exists(server_log_file):
        os.remove(server_log_file)
    
    # Start the server process
    server_log = open(server_log_file, 'w')
    cmd = f"python federated_embryo_demo.py server --port={port}"
    server_process = subprocess.Popen(
        cmd, shell=True, stdout=server_log, 
        stderr=subprocess.STDOUT, text=True
    )
    
    # Give the server time to start
    time.sleep(3)
    
    # Get local IP
    local_ip = get_local_ip()
    server_address = f"{local_ip}:{port}"
    
    # Set up ngrok if requested
    ngrok_url = None
    if use_ngrok:
        tunnel, ngrok_url = setup_ngrok(port)
        ngrok_tunnel = tunnel
    
    # Build the response message
    message = f"Server started at {server_address}\n"
    message += "Command for local IVF clinics:\n"
    message += f"python federated_embryo_demo.py client --server_address={server_address} --client_id=X\n"
    
    if ngrok_url:
        message += "\nNgrok tunnel established for external connections:\n"
        message += f"Connection address: {ngrok_url}\n"
        message += f"Command for external IVF clinics:\n"
        message += f"python federated_embryo_demo.py client --server_address={ngrok_url} --client_id=X\n"
    
    return message

def start_client_process(server_address, client_id):
    """Start a federated learning client (IVF clinic) in a separate process."""
    global client_processes
    
    # Check if this client ID already exists
    if str(client_id) in client_processes:
        return f"IVF Clinic {client_id} is already running"
    
    # Create a log file for this client
    client_log_file = f"clinic{client_id}_output.log"
    if os.path.exists(client_log_file):
        os.remove(client_log_file)
    
    # Start the client process
    client_log = open(client_log_file, 'w')
    cmd = f"python federated_embryo_demo.py client --server_address={server_address} --client_id={client_id}"
    process = subprocess.Popen(
        cmd, shell=True, stdout=client_log, 
        stderr=subprocess.STDOUT, text=True
    )
    
    # Store the client information
    client_processes[str(client_id)] = {
        "process": process,
        "log_file": client_log_file,
        "start_time": datetime.now(),
        "server_address": server_address
    }
    
    return f"Started IVF Clinic {client_id} connecting to {server_address}"

def stop_all_processes():
    """Stop the server and all client processes."""
    global server_process, client_processes, ngrok_tunnel
    
    # Stop all clients
    for client_id, info in client_processes.items():
        try:
            info["process"].terminate()
        except:
            pass
    client_processes = {}
    
    # Stop the server
    if server_process:
        try:
            server_process.terminate()
            server_process = None
        except:
            pass
    
    # Close ngrok tunnel if it exists
    if ngrok_tunnel:
        close_ngrok()
        ngrok_tunnel = None
    
    return "Server and all IVF clinics stopped. Ngrok tunnel closed."

def parse_server_log():
    """Parse the server log file to extract training metrics."""
    metrics = {
        "rounds": [],
        "loss": [],
        "accuracy": [],
        "status": "Unknown"
    }
    
    if not os.path.exists(server_log_file):
        metrics["status"] = "Server not started"
        return metrics
    
    try:
        with open(server_log_file, 'r') as f:
            content = f.read()
            
            # Determine status
            if "[SUMMARY]" in content:
                metrics["status"] = "Training complete"
            elif "[ROUND 3]" in content:
                metrics["status"] = "Round 3 in progress"
            elif "[ROUND 2]" in content:
                metrics["status"] = "Round 2 in progress"
            elif "[ROUND 1]" in content:
                metrics["status"] = "Round 1 in progress"
            elif "Starting Flower server" in content:
                metrics["status"] = "Server started, waiting for clients"
            
            # Extract rounds
            round_markers = ["[ROUND 1]", "[ROUND 2]", "[ROUND 3]"]
            for i, marker in enumerate(round_markers):
                if marker in content:
                    metrics["rounds"].append(i+1)
            
            # Extract accuracy data
            if "History (metrics, distributed, evaluate)" in content:
                acc_section = content.split("History (metrics, distributed, evaluate):")[1].split("\n")[1]
                if "accuracy" in acc_section:
                    # Parse the accuracy tuple list like [(1, 0.55), (2, 0.45), (3, 0.675)]
                    acc_str = acc_section.strip().replace("'accuracy': [", "").replace("]", "").replace("(", "").replace(")", "")
                    acc_pairs = acc_str.split(", ")
                    for pair in acc_pairs:
                        if "," in pair:
                            round_num, acc = pair.split(",")
                            try:
                                # metrics["rounds"].append(int(round_num))
                                metrics["accuracy"].append(float(acc))
                            except ValueError:
                                pass
            
            # Extract loss values
            if "History (loss, distributed)" in content:
                loss_section = content.split("History (loss, distributed):")[1]
                loss_lines = loss_section.strip().split("\n")
                for line in loss_lines:
                    if "round" in line and ":" in line:
                        try:
                            loss_value = float(line.split(":")[1].strip())
                            metrics["loss"].append(loss_value)
                        except (ValueError, IndexError):
                            pass
            
            # Extract connected clients
            if "aggregate_fit: received " in content:
                line = [l for l in content.split("\n") if "aggregate_fit: received " in l][-1]
                try:
                    num_clients = int(line.split("aggregate_fit: received ")[1].split(" ")[0])
                    metrics["num_clients"] = num_clients
                except (ValueError, IndexError):
                    metrics["num_clients"] = 0
            
    except Exception as e:
        metrics["status"] = f"Error parsing log: {str(e)}"
    
    return metrics

def parse_client_logs():
    """Parse all client logs to extract training metrics."""
    client_metrics = {}
    
    for client_id, info in client_processes.items():
        log_file = info["log_file"]
        metrics = {
            "epochs": [],
            "loss": [],
            "accuracy": [],
            "status": "Unknown"
        }
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
                # Extract training metrics
                epoch_lines = [line for line in content.split('\n') if "Epoch 1: train loss" in line]
                
                for line in epoch_lines:
                    try:
                        loss_part = line.split("train loss")[1].split(",")[0].strip()
                        acc_part = line.split("accuracy")[1].strip()
                        metrics["epochs"].append(len(metrics["epochs"]) + 1)
                        metrics["loss"].append(float(loss_part))
                        metrics["accuracy"].append(float(acc_part))
                    except (IndexError, ValueError):
                        pass
                
                # Determine status
                if "Disconnect and shut down" in content:
                    metrics["status"] = "Training complete"
                elif len(metrics["epochs"]) > 0:
                    metrics["status"] = f"Training in progress (Epoch {len(metrics['epochs'])})"
                elif "Model created on" in content:
                    metrics["status"] = "Connected, waiting for training"
                
        except Exception as e:
            metrics["status"] = f"Error: {str(e)}"
        
        client_metrics[client_id] = metrics
    
    return client_metrics

def get_system_status():
    """Get the current status of the federated learning system."""
    server_metrics = parse_server_log()
    client_metrics = parse_client_logs()
    
    # Format server status
    status = f"Server Status: {server_metrics['status']}\n"
    if 'num_clients' in server_metrics:
        status += f"IVF clinics connected: {server_metrics['num_clients']}\n"
    else:
        status += f"IVF clinics connected: {len(client_processes)}\n"
    
    if server_metrics["rounds"]:
        status += f"Completed rounds: {len(server_metrics['rounds'])}\n"
    
    # Format client status
    if client_metrics:
        status += "\nIVF Clinic Status:\n"
        for client_id, metrics in client_metrics.items():
            status += f"IVF Clinic {client_id}: {metrics['status']}\n"
            
            # Add performance metrics if available
            if metrics['epochs'] and metrics['loss'] and metrics['accuracy']:
                latest_epoch = metrics['epochs'][-1]
                latest_loss = metrics['loss'][-1]
                latest_acc = metrics['accuracy'][-1]
                status += f"  Last epoch: {latest_epoch}, Loss: {latest_loss:.4f}, Accuracy: {latest_acc:.4f}\n"
    
    return status

def create_plots():
    """Create training progress plots for server and clients."""
    server_metrics = parse_server_log()
    client_metrics = parse_client_logs()
    
    # Create server plot
    fig_server = plt.figure(figsize=(10, 5))
    
    # If we have data, create detailed plots
    if server_metrics["rounds"] and (server_metrics["loss"] or server_metrics["accuracy"]):
        # Create a subplot for accuracy and loss
        ax1 = fig_server.add_subplot(111)
        
        # Plot accuracy if available
        if server_metrics["accuracy"]:
            rounds = list(range(1, len(server_metrics["accuracy"])+1))
            ax1.plot(rounds, server_metrics["accuracy"], 'o-', color='blue', label='Accuracy')
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Accuracy', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim([0, 1])
        
        # Plot loss if available on a second y-axis
        if server_metrics["loss"]:
            ax2 = ax1.twinx()
            rounds = list(range(1, len(server_metrics["loss"])+1))
            ax2.plot(rounds, server_metrics["loss"], 'o-', color='red', label='Loss')
            ax2.set_ylabel('Loss', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        # Add a title
        plt.title('Global Model Progress')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        if server_metrics["loss"]:
            lines2, labels2 = ax2.get_legend_handles_labels()
            fig_server.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            fig_server.legend(lines1, labels1, loc='upper right')
    else:
        # If no data yet, show a placeholder
        plt.figtext(0.5, 0.5, "Waiting for training data...", ha="center", va="center", fontsize=14)
        plt.title("Server Training Progress")
    
    plt.tight_layout()
    
    # Create client plots if we have clients
    if client_metrics:
        # Create a figure with a subplot for each client
        num_clients = len(client_metrics)
        fig_clients, axes = plt.subplots(num_clients, 1, figsize=(10, 4*num_clients))
        
        # Handle the case with just one client
        if num_clients == 1:
            axes = [axes]
        
        # Create a plot for each client
        for i, (client_id, metrics) in enumerate(client_metrics.items()):
            ax = axes[i]
            ax.set_title(f"IVF Clinic {client_id} Training Progress")
            
            if metrics["epochs"] and metrics["loss"] and metrics["accuracy"]:
                # Create two y-axes
                ax2 = ax.twinx()
                
                # Plot loss and accuracy
                ax.plot(metrics["epochs"], metrics["accuracy"], 'o-', color='blue', label='Accuracy')
                ax2.plot(metrics["epochs"], metrics["loss"], 'o-', color='red', label='Loss')
                
                # Set labels
                ax.set_xlabel("Training Round")
                ax.set_ylabel("Accuracy", color='blue')
                ax2.set_ylabel("Loss", color='red')
                
                # Set colors
                ax.tick_params(axis='y', labelcolor='blue')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Add a legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax.text(0.5, 0.5, "Waiting for training data...", 
                        ha="center", va="center", transform=ax.transAxes, fontsize=14)
        
        plt.tight_layout()
        return fig_server, fig_clients
    
    # If no clients, return only the server plot
    return fig_server, None

def launch_app():
    """Launch the integrated Gradio app."""
    with gr.Blocks(title="EmbryoML Federated Learning App") as app:
        gr.Markdown("# EmbryoML Federated Learning System")
        gr.Markdown("""
        This application allows you to run a federated learning system and connect with IVF clinics around the world.
        
        **How it works:**
        1. Start the server (optionally with ngrok for external connections)
        2. Share the connection details with other clinics
        3. They run the client command on their computers
        4. You can also start local test clinics
        5. Monitor training progress in real-time
        """)
        
        with gr.Tabs():
            with gr.TabItem("Server Control"):
                with gr.Row():
                    with gr.Column():
                        port_input = gr.Number(value=8090, label="Server Port", precision=0)
                        use_ngrok = gr.Checkbox(label="Use ngrok for external connections")
                        gr.Markdown("*Using ngrok allows clinics to connect without port forwarding*")
                        start_server_btn = gr.Button("Start Server", variant="primary")
                        server_output = gr.Textbox(label="Server Output", lines=8)
                        
                        stop_btn = gr.Button("Stop All Processes", variant="stop")
                    
                    with gr.Column():
                        gr.Markdown("### System Status")
                        status_output = gr.Textbox(label="Current Status", lines=15)
                        refresh_btn = gr.Button("Refresh Status")

            with gr.TabItem("IVF Clinic Control"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Start Local Test IVF Clinic")
                        server_address = gr.Textbox(label="Server Address", placeholder="127.0.0.1:8090")
                        client_id = gr.Number(value=1, label="Clinic ID", precision=0)
                        start_client_btn = gr.Button("Start Local IVF Clinic")
                        client_output = gr.Textbox(label="Clinic Output", lines=4)
                    
                    with gr.Column():
                        gr.Markdown("### IVF Clinic Connection Instructions")
                        gr.Markdown("""
                        To connect from another clinic:
                        
                        1. Share the connection address from the Server Control tab
                        2. Make sure the other clinic has all required files
                        3. They should run the command shown in the Server Output box
                        4. Each clinic should use a unique Clinic ID
                        
                        *The ngrok address allows connections across different networks without port forwarding*
                        """)

            with gr.TabItem("Visualization"):
                gr.Markdown("### Training Progress Visualization")
                with gr.Row():
                    server_plot = gr.Plot(label="Global Model Progress")
                
                client_plots = gr.Plot(label="IVF Clinic Training Progress")
                
                vis_refresh_btn = gr.Button("Refresh Plots")
        
        # Set up event handlers
        start_server_btn.click(
            fn=start_server_process,
            inputs=[port_input, use_ngrok],
            outputs=[server_output]
        )
        
        start_client_btn.click(
            fn=start_client_process,
            inputs=[server_address, client_id],
            outputs=[client_output]
        )
        
        stop_btn.click(
            fn=stop_all_processes,
            inputs=[],
            outputs=[server_output]
        )
        
        # Update status function
        def update_status():
            return get_system_status()
        
        refresh_btn.click(
            fn=update_status,
            inputs=[],
            outputs=[status_output]
        )
        
        # Update plots
        def update_plots():
            server_fig, client_figs = create_plots()
            return server_fig, client_figs if client_figs else gr.update(value=None)
        
        vis_refresh_btn.click(
            fn=update_plots,
            inputs=[],
            outputs=[server_plot, client_plots]
        )
        
        # Set up automatic refresh (every 5 seconds)
        def auto_refresh():
            status = update_status()
            server_fig, client_figs = create_plots()
            return status, server_fig, client_figs if client_figs else gr.update(value=None)
        
        # Remove timer functionality - use manual refresh only
        # Comment on how to refresh: Add instruction text
        gr.Markdown("**Note:** Click 'Refresh All' to update both status and plots.")
        
        # Add a 'Refresh All' button
        refresh_all_btn = gr.Button("Refresh All", variant="primary")
        refresh_all_btn.click(
            fn=auto_refresh,
            inputs=[],
            outputs=[status_output, server_plot, client_plots]
        )
        
    # Launch the app
    app.launch(share=True)

if __name__ == "__main__":
    # Check if data directory exists and has files
    try:
        files = find_image_files()
        print(f"Found {len(files)} embryo images for training")
    except ValueError:
        print("No embryo images found. Running create_sample_data.py to generate test data...")
        try:
            subprocess.run([sys.executable, "create_sample_data.py"], check=True)
            print("Test data created successfully")
        except Exception as e:
            print(f"Error creating test data: {str(e)}")
            print("Please run create_sample_data.py manually before starting this app")
            sys.exit(1)
    
    # Launch the app
    print("Starting Integrated EmbryoML Federated Learning App...")
    launch_app() 