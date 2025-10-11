"""
DICOM Volume Loader for MicroCT Mouse Brain Data
Loads a series of DICOM files from a folder and converts to 3D numpy array
"""

import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple, Optional
import glob
from tqdm import tqdm


def load_dicom_volume(dicom_folder: str, normalize: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Load DICOM series from a folder into a 3D volume.
    
    Parameters:
    -----------
    dicom_folder : str
        Path to folder containing DICOM (.dcm) files
    normalize : bool
        If True, normalize volume to [0, 1] range
        
    Returns:
    --------
    volume : np.ndarray
        3D array of shape (height, width, slices)
    metadata : dict
        Dictionary containing DICOM metadata
    """
    
    dicom_folder = Path(dicom_folder)
    
    if not dicom_folder.exists():
        raise FileNotFoundError(f"Directory not found: {dicom_folder}")
    
    # Find all DICOM files
    dcm_files = sorted(glob.glob(str(dicom_folder / "*.dcm")))
    
    if len(dcm_files) == 0:
        raise ValueError(f"No DICOM files found in {dicom_folder}")
    
    print(f"Loading {len(dcm_files)} DICOM files from: {dicom_folder}")
    
    # Read first file to get dimensions
    first_slice = pydicom.dcmread(dcm_files[0])
    img_shape = first_slice.pixel_array.shape
    
    # Initialize volume array
    volume = np.zeros((img_shape[0], img_shape[1], len(dcm_files)), dtype=np.float64)
    
    # Load all slices
    metadata = {}
    for i, dcm_file in enumerate(tqdm(dcm_files, desc="Loading DICOM slices")):
        ds = pydicom.dcmread(dcm_file)
        volume[:, :, i] = ds.pixel_array.astype(np.float64)
        
        # Store metadata from first slice
        if i == 0:
            metadata = {
                'PixelSpacing': getattr(ds, 'PixelSpacing', None),
                'SliceThickness': getattr(ds, 'SliceThickness', None),
                'ImageOrientationPatient': getattr(ds, 'ImageOrientationPatient', None),
                'ImagePositionPatient': getattr(ds, 'ImagePositionPatient', None),
                'Rows': ds.Rows,
                'Columns': ds.Columns,
                'NumberOfSlices': len(dcm_files)
            }
    
    # Normalize if requested
    if normalize:
        volume = volume / np.max(volume)
    
    print(f"Volume loaded successfully. Dimensions: {volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]}")
    
    return volume, metadata


def save_volume_as_tif(volume: np.ndarray, output_path: str, bit_depth: int = 16):
    """
    Save 3D volume as multi-page TIFF file.
    
    Parameters:
    -----------
    volume : np.ndarray
        3D volume array
    output_path : str
        Output file path
    bit_depth : int
        Bit depth for output (8 or 16)
    """
    from tifffile import imwrite
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if bit_depth == 16:
        volume_out = (volume * 65535).astype(np.uint16)
    elif bit_depth == 8:
        volume_out = (volume * 255).astype(np.uint8)
    else:
        raise ValueError("bit_depth must be 8 or 16")
    
    print(f"Saving volume to: {output_path}")
    imwrite(output_path, volume_out, compression='none')
    print(f"Successfully saved {volume_out.shape[2]} slices")


if __name__ == "__main__":
    import sys
    from tkinter import Tk, filedialog
    
    # GUI folder selection
    root = Tk()
    root.withdraw()
    
    dicom_folder = filedialog.askdirectory(
        title="Select folder with DICOM files",
        initialdir="C:/DATA/MFA/uCT"
    )
    
    if not dicom_folder:
        print("No folder selected. Exiting.")
        sys.exit(0)
    
    # Load volume
    volume, metadata = load_dicom_volume(dicom_folder)
    
    # Save as TIF
    output_path = filedialog.asksaveasfilename(
        title="Save as TIF",
        defaultextension=".tif",
        filetypes=[("TIFF files", "*.tif")]
    )
    
    if output_path:
        save_volume_as_tif(volume, output_path)