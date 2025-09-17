import argparse
import os
import sys

# Add the directory containing read_iml_hyp to the Python path
# This makes the script runnable from anywhere, assuming read_iml_hyp.py is in the same directory as this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(SCRIPT_DIR) # Not strictly necessary if run from the same dir or if the package is installed

try:
    from read_iml_hyp import read_iml_hyp, visualize_hyperspectral_image
except ImportError:
    print(f"Error: Could not import from read_iml_hyp.py. Ensure it's in the same directory ({SCRIPT_DIR}) or in PYTHONPATH.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Load and visualize a hyperspectral image using functions from read_iml_hyp.py.'
    )
    
    parser.add_argument('--image-dir', type=str, default='.',
                        help='Directory containing the hyperspectral image files (e.g., where .bil and .bil.hdr are located). Default is current directory.')
    
    parser.add_argument('--image-name', type=str, required=True,
                        help='Base name of the hyperspectral image (e.g., "Arctia caja_1D_030717" if files are Arctia caja_1D_030717.bil and Arctia caja_1D_030717.bil.hdr). Do NOT include the .bil extension.')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Optional directory to save visualization plots. If None, plots are displayed interactively (if supported by backend).')
    
    args = parser.parse_args()
    
    # Verify image directory exists
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)
        
    # Optional: Warn if expected files don't seem to be there based on naming convention
    # read_iml_hyp itself will handle the actual file checking.
    expected_bil_path = os.path.join(args.image_dir, args.image_name + ".bil")
    expected_hdr_path = os.path.join(args.image_dir, args.image_name + ".bil.hdr")

    if not os.path.isfile(expected_bil_path):
        print(f"Warning: Expected BIL file ({expected_bil_path}) not found. read_iml_hyp may fail.")
    if not os.path.isfile(expected_hdr_path):
        print(f"Warning: Expected HDR file ({expected_hdr_path}) not found. read_iml_hyp may fail.")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
    return args

def main():
    """Main function to load and visualize hyperspectral data."""
    args = parse_arguments()

    print(f"Attempting to load hyperspectral image: '{args.image_name}' from directory: '{args.image_dir}'")

    try:
        # Call read_iml_hyp from the existing script
        # 'address' is the directory, 'name' is the base name WITHOUT .bil extension
        im, wvls, scan_info, gain = read_iml_hyp(address=args.image_dir, name=args.image_name)
        
        print("Image loaded successfully.")
        if scan_info:
            print(f"  Dimensions (lines, samples, bands): ({scan_info.get('lines', 'N/A')}, {scan_info.get('samples', 'N/A')}, {scan_info.get('bands', 'N/A')})")
        if wvls is not None:
            print(f"  Number of wavelengths: {len(wvls)}")
        if gain is not None:
            print(f"  Gain: {gain}")

        print("\nVisualizing hyperspectral image...")
        # Call visualize_hyperspectral_image from the existing script
        visualize_hyperspectral_image(image=im, wavelengths=wvls, output_dir=args.output_dir)
        
        if args.output_dir:
            print(f"Visualization complete. Plots should be saved in '{args.output_dir}'.")
        else:
            print("Visualization complete. Plots should have been displayed.")

    except FileNotFoundError as e:
        print(f"\nError: A required file was not found during loading.")
        print(f"Details: {e}")
        print("Please ensure the .bil and .bil.hdr files exist and match the expected naming convention (basename.bil, basename.bil.hdr).")
    except ValueError as e:
        print(f"\nError: There was an issue with the data or parameters during loading.")
        print(f"Details: {e}")
    except IOError as e:
        print(f"\nError: An I/O problem occurred during loading.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
