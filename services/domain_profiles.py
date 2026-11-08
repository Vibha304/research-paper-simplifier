"""Engineering domain profiles used by the built-in engine to tailor templates.

Each profile supplies sensible defaults (tech stack, modules, metrics, skills,
an everyday analogy) that are combined with facts extracted from the paper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    name: str
    signals: tuple[str, ...]
    analogy: str
    tech_stack: tuple[str, ...]
    modules: tuple[str, ...]
    metrics: tuple[str, ...]
    skills: tuple[str, ...]
    deployment: str
    base_difficulty: float  # added to heuristic difficulty (0..1.5)


DOMAINS: tuple[DomainProfile, ...] = (
    DomainProfile(
        name="Computer Vision / Deep Learning",
        signals=("image", "convolution", "cnn", "pixel", "object detection", "segmentation", "vision", "resnet", "yolo"),
        analogy="Think of it like a student who studies thousands of labelled photos until they can recognise new ones on their own.",
        tech_stack=("Python", "PyTorch", "torchvision", "OpenCV", "NumPy", "Matplotlib", "Streamlit"),
        modules=("data_loader: download, augment and batch the images", "model: the network architecture from the paper",
                 "train: training loop with checkpoints", "evaluate: metrics, confusion matrix and plots",
                 "app: Streamlit demo that classifies uploaded images"),
        metrics=("Accuracy", "Precision / Recall / F1", "Inference time per image", "Model size (MB)"),
        skills=("Deep learning with PyTorch", "Image preprocessing & augmentation", "Model evaluation", "Experiment tracking"),
        deployment="Export the trained model to ONNX or TorchScript, wrap it in a FastAPI/Streamlit service and containerise it with Docker.",
        base_difficulty=1.0,
    ),
    DomainProfile(
        name="Natural Language Processing",
        signals=("language model", "nlp", "text", "token", "transformer", "bert", "translation", "sentiment", "corpus", "llm"),
        analogy="Think of it like a reader who has seen millions of sentences and learns which words tend to belong together.",
        tech_stack=("Python", "Hugging Face Transformers", "PyTorch", "scikit-learn", "Pandas", "Streamlit"),
        modules=("data_loader: load and clean the text corpus", "tokenizer: tokenisation and vocabulary handling",
                 "model: the language model / classifier from the paper", "train: fine-tuning loop",
                 "evaluate: metrics and error analysis", "app: Streamlit demo for typing in text"),
        metrics=("Accuracy / F1", "BLEU or ROUGE (for generation)", "Perplexity", "Latency per request"),
        skills=("Text preprocessing", "Transformers & fine-tuning", "Hugging Face ecosystem", "Evaluation of NLP models"),
        deployment="Serve the model behind a FastAPI endpoint, cache tokenizers, and quantise the model to run on CPU.",
        base_difficulty=1.0,
    ),
    DomainProfile(
        name="Cybersecurity",
        signals=("attack", "intrusion", "malware", "security", "threat", "anomaly", "phishing", "vulnerability", "adversar"),
        analogy="Think of it like a security guard who learns what normal visitors look like so that unusual behaviour stands out.",
        tech_stack=("Python", "scikit-learn", "Pandas", "XGBoost", "Wireshark/Scapy", "Streamlit"),
        modules=("data_loader: load and clean traffic/log datasets", "features: feature engineering from raw records",
                 "detector: the detection model from the paper", "evaluate: detection metrics and ROC curves",
                 "dashboard: live alert dashboard"),
        metrics=("Detection rate (Recall)", "False positive rate", "F1-score", "ROC-AUC"),
        skills=("Security data analysis", "Anomaly detection", "Feature engineering", "Imbalanced-data handling"),
        deployment="Run the detector as a service that consumes logs (e.g. from Kafka), with alerts pushed to a dashboard.",
        base_difficulty=0.6,
    ),
    DomainProfile(
        name="Networking & IoT",
        signals=("network", "iot", "sensor", "wireless", "routing", "packet", "protocol", "edge", "5g", "vehicular", "sdn"),
        analogy="Think of it like a city traffic system that decides which roads cars should take so everyone arrives quickly.",
        tech_stack=("Python", "NS-3 or Mininet", "MQTT", "Raspberry Pi / ESP32", "Pandas", "Matplotlib"),
        modules=("topology: build the network/simulation topology", "protocol: implement the proposed scheme",
                 "traffic: workload / sensor data generator", "metrics: collect throughput, delay and energy",
                 "visualize: plots and comparison with baseline"),
        metrics=("Throughput", "End-to-end delay", "Packet delivery ratio", "Energy consumption"),
        skills=("Network simulation", "Protocol design", "IoT hardware basics", "Performance analysis"),
        deployment="Prototype on Raspberry Pi/ESP32 nodes with MQTT, then compare measured results with the simulation.",
        base_difficulty=0.7,
    ),
    DomainProfile(
        name="Blockchain",
        signals=("blockchain", "smart contract", "ledger", "consensus", "ethereum", "hyperledger", "decentralized"),
        analogy="Think of it like a class notebook that everyone has a copy of, so no single person can secretly change the records.",
        tech_stack=("Solidity", "Hardhat/Ganache", "Ethereum testnet", "web3.py", "React or Streamlit"),
        modules=("contracts: smart contracts from the paper", "deploy: deployment scripts for a local chain",
                 "client: web3 client / API", "benchmark: gas cost and latency measurement", "ui: simple front-end"),
        metrics=("Transaction latency", "Gas cost", "Throughput (TPS)", "Security properties checked"),
        skills=("Smart contract development", "Web3 integration", "Cryptography basics", "Decentralised app design"),
        deployment="Deploy contracts to a public testnet and host the front-end statically; never put real funds at risk.",
        base_difficulty=0.8,
    ),
    DomainProfile(
        name="Signal Processing & Communications",
        signals=("signal", "frequency", "channel", "modulation", "ofdm", "mimo", "filter", "spectrum", "antenna", "noise"),
        analogy="Think of it like tuning a radio: separating the song you want from the static and the other stations.",
        tech_stack=("Python", "NumPy", "SciPy", "MATLAB/Octave", "Matplotlib", "GNU Radio (optional)"),
        modules=("signal_gen: generate or load signals", "channel: channel / noise model", "algorithm: proposed processing method",
                 "receiver: detection and decoding", "analysis: BER/SNR curves"),
        metrics=("Bit error rate (BER)", "Signal-to-noise ratio (SNR)", "Spectral efficiency", "Computational complexity"),
        skills=("Digital signal processing", "Simulation with NumPy/SciPy", "Communication theory", "Scientific plotting"),
        deployment="Package the simulator as a reusable library with notebooks; optionally test on an SDR (e.g. RTL-SDR).",
        base_difficulty=0.9,
    ),
    DomainProfile(
        name="Robotics & Control",
        signals=("robot", "control", "controller", "trajectory", "kinematic", "uav", "drone", "autonomous", "pid", "slam"),
        analogy="Think of it like riding a bicycle: constantly sensing your balance and making small corrections to stay on track.",
        tech_stack=("Python", "ROS 2", "Gazebo", "NumPy", "OpenCV", "Matplotlib"),
        modules=("model: robot / plant dynamics model", "controller: the proposed control algorithm", "perception: sensing and state estimation",
                 "simulation: Gazebo / Python simulation", "analysis: tracking error plots"),
        metrics=("Tracking error", "Settling time / overshoot", "Success rate", "Computation time per control step"),
        skills=("Control theory", "ROS 2", "Simulation", "Sensor fusion"),
        deployment="Validate in simulation first, then run on a low-cost robot kit with safety limits in place.",
        base_difficulty=1.0,
    ),
    DomainProfile(
        name="Machine Learning & Data Science",
        signals=("machine learning", "classifier", "regression", "prediction", "random forest", "svm", "decision tree",
                 "xgboost", "gradient boosting", "k-means", "naive bayes"),
        analogy="Think of it like learning to predict tomorrow's weather by studying patterns in years of past weather records.",
        tech_stack=("Python", "scikit-learn", "Pandas", "NumPy", "Matplotlib/Seaborn", "Streamlit"),
        modules=("data_loader: load and clean the dataset", "features: feature engineering and selection",
                 "model: the proposed model", "evaluate: cross-validation and metrics", "app: Streamlit prediction demo"),
        metrics=("Accuracy", "Precision / Recall / F1", "RMSE / MAE (regression)", "Training time"),
        skills=("Data cleaning with Pandas", "Classical ML with scikit-learn", "Model evaluation", "Data visualisation"),
        deployment="Save the model with joblib, serve it through FastAPI or Streamlit, and log predictions for monitoring.",
        base_difficulty=0.4,
    ),
)

GENERAL = DomainProfile(
    name="General Engineering / Software",
    signals=(),
    analogy="Think of it like improving a recipe: the authors change a few key steps so the dish turns out better and faster.",
    tech_stack=("Python", "NumPy", "Pandas", "Matplotlib", "Streamlit"),
    modules=("data: input data handling", "core: the main algorithm from the paper", "baseline: a simple method to compare against",
             "evaluate: experiments and metrics", "app: small demo interface"),
    metrics=("Correctness on test cases", "Runtime", "Memory usage", "Improvement over baseline (%)"),
    skills=("Reading research papers", "Algorithm implementation", "Experimental evaluation", "Technical writing"),
    deployment="Package the code as a Python library with a CLI and a small web demo; add CI tests.",
    base_difficulty=0.3,
)


def detect_domain(text: str) -> DomainProfile:
    lowered = text.lower()
    best, best_score = GENERAL, 0
    for domain in DOMAINS:
        score = sum(len(re.findall(rf"\b{re.escape(signal)}", lowered)) for signal in domain.signals)
        if score > best_score:
            best, best_score = domain, score
    return best if best_score >= 5 else GENERAL
