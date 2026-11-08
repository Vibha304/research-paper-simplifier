"""A realistic multi-section paper (written for testing) rendered to PDF with PyMuPDF."""

from __future__ import annotations

import pymupdf as fitz

TITLE = "LightIDS: A Lightweight Deep Learning Intrusion Detection System for IoT Networks"

SECTIONS: list[tuple[str, str]] = [
    ("Abstract", (
        "Abstract—The rapid growth of Internet of Things (IoT) devices has created a large attack surface for "
        "cyber attacks. Existing intrusion detection systems are often too computationally expensive to run on "
        "resource-constrained IoT gateways. In this paper, we propose LightIDS, a lightweight intrusion detection "
        "system that combines a compact convolutional neural network with feature selection. We evaluate LightIDS "
        "on the CICIDS2017 and UNSW-NB15 datasets. Experimental results show that LightIDS achieves 98.7% accuracy "
        "while reducing inference time by 63% compared with state-of-the-art deep learning baselines.")),
    ("Index Terms", "Index Terms—Intrusion detection, Internet of Things, deep learning, feature selection, edge computing."),
    ("1 Introduction", (
        "IoT devices are deployed in homes, hospitals and factories, and they frequently lack basic security "
        "protections. A major challenge is that attacks such as botnets and denial of service spread quickly "
        "across poorly protected devices. However, most existing deep learning detectors suffer from high memory "
        "usage and latency, which makes them difficult to deploy on low-power gateways. Traditional signature-based "
        "systems cannot detect previously unseen attacks. "
        "To address these problems, we make the following contributions. We propose a compact convolutional "
        "neural network architecture with only 48,000 parameters for traffic classification. We introduce a "
        "mutual-information based feature selection step that reduces the number of input features from 78 to 20. "
        "We demonstrate on two public benchmarks that our model outperforms larger baselines in both accuracy and "
        "inference time. The rest of this paper is organised as follows.")),
    ("2 Related Work", (
        "Several studies have applied machine learning to intrusion detection. Random forest and support vector "
        "machine classifiers were widely used in early work. Recent approaches use recurrent neural networks and "
        "LSTM models to capture temporal patterns in network traffic, but these models are computationally heavy.")),
    ("3 Proposed Method", (
        "The proposed LightIDS framework consists of three stages: preprocessing, feature selection and "
        "classification. First, raw network flows are cleaned, missing values are removed and every feature is "
        "normalised to the range zero to one. Then, mutual information between each feature and the attack label "
        "is computed, and the twenty most informative features are kept. Next, the selected features are reshaped "
        "into a small two-dimensional grid so that convolution layers can learn local correlations between "
        "features. The network architecture contains two convolutional layers with batch normalization, followed "
        "by global average pooling and a softmax output layer. Finally, the model is trained with the Adam "
        "optimizer and a weighted cross-entropy loss function to handle class imbalance. Dropout with a rate of "
        "0.3 is applied to reduce overfitting. The trained model is quantised to eight-bit integers so that "
        "inference can run efficiently on a Raspberry Pi gateway.")),
    ("4 Experiments", (
        "We implemented LightIDS in Python using PyTorch and scikit-learn. All experiments were run on a laptop "
        "with an Intel Core i5 processor and on a Raspberry Pi 4. The CICIDS2017 dataset contains benign traffic "
        "and fourteen attack types captured over five days. The UNSW-NB15 dataset contains 2.5 million records "
        "with nine attack families. We report accuracy, precision, recall, F1-score and inference time. "
        "LightIDS achieves an F1-score of 0.985 on CICIDS2017 and 0.962 on UNSW-NB15. Compared with an LSTM "
        "baseline, our model is 63% faster and uses five times less memory. An ablation study shows that "
        "feature selection improves accuracy by 1.4% while halving the training time.")),
    ("5 Limitations", (
        "Our study has several limitations. The model was evaluated only on two offline datasets, which may not "
        "reflect the traffic of real deployments. The approach assumes that labelled attack data is available "
        "for training, which is expensive to obtain. LightIDS cannot detect encrypted payload attacks because it "
        "only uses flow-level statistics.")),
    ("6 Conclusion and Future Work", (
        "In this paper we presented LightIDS, a lightweight deep learning intrusion detection system for IoT "
        "gateways. In the future, we plan to evaluate the system on live traffic from a smart-home testbed. "
        "Future work will also explore federated learning so that gateways can collaboratively improve the "
        "model without sharing raw traffic. The approach could be extended to detect zero-day attacks using "
        "unsupervised anomaly detection.")),
    ("References", "[1] A. Author, An old intrusion detection paper, 2015. [2] B. Author, Another paper, 2019."),
]


def make_realistic_paper_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 40, 545, 100), TITLE, fontsize=16, align=1)
    y = 110.0
    for heading, body in SECTIONS:
        blocks = [body] if heading in ("Abstract", "Index Terms") else [heading, body]
        for index, text in enumerate(blocks):
            is_heading = index == 0 and len(blocks) == 2
            fontsize = 11 if is_heading else 9.5
            height = 22 if is_heading else max(30, 13 * (len(text) // 95 + 2))
            if y + height > 800:
                page = doc.new_page(width=595, height=842)
                y = 50
            page.insert_textbox(fitz.Rect(50, y, 545, y + height), text, fontsize=fontsize)
            y += height + 6
    data = doc.tobytes()
    doc.close()
    return data
