#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import argparse
import requests
from tqdm import tqdm
import zipfile


class DataPreprocessor:
    """Handles data preprocessing for MIMIC-CXR and IU-Xray datasets"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def download_mimic_cxr(self, url: str = None):
        """Download MIMIC-CXR dataset (placeholder - requires PhysioNet access)"""
        print("MIMIC-CXR Download Instructions:")
        print("1. Request access at: https://physionet.org/content/mimic-cxr/2.0.0/")
        print("2. Download the dataset to data/mimic-cxr/")
        print("3. Run: python data/preprocess.py --dataset mimic-cxr")

    def download_iu_xray(self, url: str = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_dcm.tgz"):
        """Download IU-Xray dataset"""
        print("Downloading IU-Xray dataset...")

        # Create IU-Xray directory
        iu_dir = self.data_dir / "iu_xray"
        iu_dir.mkdir(exist_ok=True)

        # Download the dataset
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(iu_dir / "NLMCXR_dcm.tgz", 'wb') as f, tqdm(
            desc="IU-Xray",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

        print("IU-Xray dataset downloaded successfully!")

    def preprocess_mimic_cxr(self, csv_path: str, img_dir: str):
        """Preprocess MIMIC-CXR data"""
        print("Preprocessing MIMIC-CXR data...")

        # Load metadata
        df = pd.read_csv(csv_path)

        # Filter for PA/AP views
        valid_views = ['PA', 'AP']
        df = df[df['ViewPosition'].isin(valid_views)]

        # Create disease labels (simplified CheXpert format)
        diseases = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion',
            'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

        # Initialize disease columns
        for disease in diseases:
            df[disease] = 0

        # Save processed metadata
        output_path = self.data_dir / "mimic_cxr_processed.csv"
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

    def preprocess_iu_xray(self, reports_path: str, images_path: str):
        """Preprocess IU-Xray data"""
        print("Preprocessing IU-Xray data...")

        # Load reports
        reports_df = pd.read_csv(reports_path)

        # Basic preprocessing
        reports_df['findings'] = reports_df['findings'].fillna('')
        reports_df['impression'] = reports_df['impression'].fillna('')

        # Save processed data
        output_path = self.data_dir / "iu_xray_processed.csv"
        reports_df.to_csv(output_path, index=False)
        print(f"Processed IU-Xray data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Data preprocessing for BrainDead-Solution")
    parser.add_argument("--dataset", choices=["mimic-cxr", "iu-xray"], required=True)
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess dataset")
    parser.add_argument("--data_dir", default="data", help="Data directory")

    args = parser.parse_args()

    preprocessor = DataPreprocessor(args.data_dir)

    if args.download:
        if args.dataset == "mimic-cxr":
            preprocessor.download_mimic_cxr()
        elif args.dataset == "iu-xray":
            preprocessor.download_iu_xray()

    if args.preprocess:
        if args.dataset == "mimic-cxr":
            # These paths need to be adjusted based on actual data location
            preprocessor.preprocess_mimic_cxr(
                csv_path="data/mimic-cxr/mimic-cxr-2.0.0-metadata.csv",
                img_dir="data/mimic-cxr/images/"
            )
        elif args.dataset == "iu-xray":
            preprocessor.preprocess_iu_xray(
                reports_path="data/iu_xray/reports.csv",
                images_path="data/iu_xray/images/"
            )


if __name__ == "__main__":
    main()