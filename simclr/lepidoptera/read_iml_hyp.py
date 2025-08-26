import os
import numpy as np
from pathlib import Path


def read_iml_hyp(address, name):
    """
    Read hyperspectral file in IML format with .bil extension

    Parameters:
    -----------
    address : str
        Path to the folder containing the IML file(s)
    name : str
        Name of the file to read (without extension)

    Returns:
    --------
    im : numpy.ndarray
        The hyperspectral image
    wvls : numpy.ndarray
        Wavelengths of measurement
    scan : dict
        Dictionary with information about scan
    gain : float
        Gain for current scan
    """
    # Build file paths
    file_path = os.path.join(address, name)
    hdr_file_path = f"{file_path}.bil.hdr"
    bil_file_path = f"{file_path}.bil"  # Explicitly add .bil extension for the image file
    
    # Define reference wavelengths (same as in the MATLAB script)
    wvls_temp = np.array([
        348.3931, 349.5263, 350.6595, 351.7927, 352.9259, 354.0591, 355.1923, 356.3255,
        357.4587, 358.5919, 359.7251, 360.8583, 361.9915, 363.1247, 364.2579, 365.3911,
        366.5243, 367.6575, 368.7907, 369.9239, 371.0571, 372.1903, 373.3235, 374.4567,
        375.5899, 376.7231, 377.8563, 378.9895, 380.1227, 381.2559, 382.3891, 383.5223,
        384.6555, 385.7887, 386.9219, 388.0551, 389.1883, 390.3215, 391.4547, 392.5879,
        393.7211, 394.8543, 395.9875, 397.1207, 398.2539, 399.3871, 400.5203, 401.6535,
        402.7867, 403.9199, 405.0531, 406.1863, 407.3195, 408.4527, 409.5859, 410.7191,
        411.8523, 412.9855, 414.1187, 415.2519, 416.3851, 417.5183, 418.6515, 419.7847,
        420.9179, 422.0511, 423.1843, 424.3175, 425.4507, 426.5839, 427.7171, 428.8503,
        429.9835, 431.1167, 432.2499, 433.3831, 434.5163, 435.6495, 436.7827, 437.9159,
        439.0491, 440.1823, 441.3155, 442.4487, 443.5819, 444.7151, 445.8483, 446.9815,
        448.1147, 449.2479, 450.3811, 451.5143, 452.6475, 453.7807, 454.9139, 456.0471,
        457.1803, 458.3135, 459.4467, 460.5799, 461.7131, 462.8463, 463.9795, 465.1127,
        466.2459, 467.3791, 468.5123, 469.6455, 470.7787, 471.9119, 473.0451, 474.1783,
        475.3115, 476.4447, 477.5779, 478.7111, 479.8443, 480.9775, 482.1107, 483.2439,
        484.3771, 485.5103, 486.6435, 487.7767, 488.9099, 490.0431, 491.1763, 492.3095,
        493.4427, 494.5759, 495.7091, 496.8423, 497.9755, 499.1087, 500.2419, 501.3751,
        502.5083, 503.6415, 504.7747, 505.9079, 507.0411, 508.1743, 509.3075, 510.4407,
        511.5739, 512.7071, 513.8403, 514.9735, 516.1067, 517.2399, 518.3731, 519.5063,
        520.6395, 521.7727, 522.9059, 524.0391, 525.1723, 526.3055, 527.4387, 528.5719,
        529.7051, 530.8383, 531.9715, 533.1047, 534.2379, 535.3711, 536.5043, 537.6375,
        538.7707, 539.9039, 541.0371, 542.1703, 543.3035, 544.4367, 545.5699, 546.7031,
        547.8363, 548.9695, 550.1027, 551.2359, 552.3691, 553.5023, 554.6355, 555.7687,
        556.9019, 558.0351, 559.1683, 560.3015, 561.4347, 562.5679, 563.7011, 564.8343,
        565.9675, 567.1007, 568.2339, 569.3671, 570.5003, 571.6335, 572.7667, 573.8999,
        575.0331, 576.1663, 577.2995, 578.4327, 579.5659, 580.6991, 581.8323, 582.9655,
        584.0987, 585.2319, 586.3651, 587.4983, 588.6315, 589.7647, 590.8979, 592.0311,
        593.1643, 594.2975, 595.4307, 596.5639, 597.6971, 598.8303, 599.9635, 601.0967,
        602.2299, 603.3631, 604.4963, 605.6295, 606.7627, 607.8959, 609.0291, 610.1623,
        611.2955, 612.4287, 613.5619, 614.6951, 615.8283, 616.9615, 618.0947, 619.2279,
        620.3611, 621.4943, 622.6275, 623.7607, 624.8939, 626.0271, 627.1603, 628.2935,
        629.4267, 630.5599, 631.6931, 632.8263, 633.9595, 635.0927, 636.2259, 637.3591,
        638.4923, 639.6255, 640.7587, 641.8919, 643.0251, 644.1583, 645.2915, 646.4247,
        647.5579, 648.6911, 649.8243, 650.9575, 652.0907, 653.2239, 654.3571, 655.4903,
        656.6235, 657.7567, 658.8899, 660.0231, 661.1563, 662.2895, 663.4227, 664.5559,
        665.6891, 666.8223, 667.9555, 669.0887, 670.2219, 671.3551, 672.4883, 673.6215,
        674.7547, 675.8879, 677.0211, 678.1543, 679.2875, 680.4207, 681.5539, 682.6871,
        683.8203, 684.9535, 686.0867, 687.2199, 688.3531, 689.4863, 690.6195, 691.7527,
        692.8859, 694.0191, 695.1523, 696.2855, 697.4187, 698.5519, 699.6851, 700.8183,
        701.9515, 703.0847, 704.2179, 705.3511, 706.4843, 707.6175, 708.7507, 709.8839,
        711.0171, 712.1503, 713.2835, 714.4167, 715.5499, 716.6831, 717.8163, 718.9495,
        720.0827, 721.2159, 722.3491, 723.4823, 724.6155, 725.7487, 726.8819, 728.0151,
        729.1483, 730.2815, 731.4147, 732.5479, 733.6811, 734.8143, 735.9475, 737.0807,
        738.2139, 739.3471, 740.4803, 741.6135, 742.7467, 743.8799, 745.0131, 746.1463,
        747.2795, 748.4127, 749.5459, 750.6791, 751.8123, 752.9455, 754.0787, 755.2119,
        756.3451, 757.4783, 758.6115, 759.7447, 760.8779, 762.0111, 763.1443, 764.2775,
        765.4107, 766.5439, 767.6771, 768.8103, 769.9435, 771.0767, 772.2099, 773.3431,
        774.4763, 775.6095, 776.7427, 777.8759, 779.0091, 780.1423, 781.2755, 782.4087,
        783.5419, 784.6751, 785.8083, 786.9415, 788.0747, 789.2079, 790.3411, 791.4743,
        792.6075, 793.7407, 794.8739, 796.0071, 797.1403, 798.2735, 799.4067, 800.5399,
        801.6731, 802.8063, 803.9395, 805.0727, 806.2059, 807.3391, 808.4723, 809.6055
    ])
    
    wvls = wvls_temp

    # Read the header file to extract metadata
    scan = {}
    gain = None
    
    try:
        with open(hdr_file_path, 'r') as hdr_file:
            lines = hdr_file.readlines()
            
            # Extract relevant parameters
            for i, line in enumerate(lines):
                # print(line.strip())  # Display line (like in MATLAB)
                
                if "lines" in line.lower() and "=" in line:
                    scan['lines'] = int(line.split('=')[1].strip())
                elif "samples" in line.lower() and "=" in line:
                    scan['samples'] = int(line.split('=')[1].strip())
                elif "bands" in line.lower() and "=" in line:
                    scan['bands'] = int(line.split('=')[1].strip())
                elif "gain" in line.lower() and "=" in line:
                    gain = float(line.split('=')[1].strip())
    
    except Exception as e:
        raise IOError(f"Error reading header file: {e}")
    
    # Check if we have all required parameters
    required_params = ['lines', 'samples', 'bands']
    for param in required_params:
        if param not in scan:
            raise ValueError(f"Missing required parameter in header file: {param}")
    
    # Read the hyperspectral image
    try:
        # Verify the .bil file exists
        if not os.path.exists(bil_file_path):
            raise FileNotFoundError(f"BIL file not found: {bil_file_path}")
        
        # Read the BIL format image using our helper function
        im, adjusted_dims = read_bil_file(bil_file_path, scan['lines'], scan['samples'], scan['bands'])
        
        # Update scan dimensions if they were adjusted during reading
        if adjusted_dims is not None:
            scan['lines'], scan['samples'], scan['bands'] = adjusted_dims
        
    except Exception as e:
        raise IOError(f"Error reading hyperspectral image: {e}")
    
    return im, wvls, scan, gain


def read_bil_file(file_path, lines, samples, bands, data_type=np.uint16, byte_order='<'):
    """
    Helper function to read a BIL (Band Interleaved by Line) format file
    
    Parameters:
    -----------
    file_path : str
        Path to the BIL file
    lines : int
        Number of lines in the image
    samples : int
        Number of samples (columns) in the image
    bands : int
        Number of spectral bands
    data_type : numpy.dtype
        Data type of the image (default: np.uint16)
    byte_order : str
        Byte order ('<' for little-endian, '>' for big-endian)
        
    Returns:
    --------
    image : numpy.ndarray
        3D array containing the hyperspectral image data
    adjusted_dimensions : tuple or None
        If dimensions were adjusted to match file size, returns (lines, samples, bands)
        Otherwise returns None
    """
    # Element size in bytes
    element_size = 2  # bytes per uint16 element
    
    # BIL format: data is stored as [lines][bands][samples]
    shape = (lines, bands, samples)
    
    # Calculate expected file size in bytes
    expected_size = lines * bands * samples * element_size
    
    # Get actual file size
    actual_size = os.path.getsize(file_path)
    
    adjusted_dimensions = None
    
    # Check if there's a size mismatch
    if actual_size != expected_size:
        # Try to recover by calculating correct dimensions
        total_elements = actual_size // element_size
        
        # Add a flag to indicate whether we should force dimensions adjustment
        force_adjust = False
        
        # If bands is likely correct (typically fixed by the hardware)
        # Try to adjust lines and samples to fit
        if total_elements % bands == 0:
            area = total_elements // bands
            
            # Try to maintain aspect ratio if possible
            if lines > 0 and samples > 0:
                ratio = lines / samples
                # Calculate new dimensions to match file size while keeping aspect ratio
                new_samples = int(np.sqrt(area / ratio))
                new_lines = int(area / new_samples)
                
                # Make sure we don't have leftover elements
                while new_lines * new_samples < area:
                    new_samples += 1
                while new_lines * new_samples > area:
                    new_lines -= 1
                    if new_lines <= 0:
                        new_lines = 1
                        break
                
                # Update shape with new dimensions
                lines, samples = new_lines, new_samples
                shape = (lines, bands, samples)
                adjusted_dimensions = (lines, samples, bands)
                # print(f"Warning: File size mismatch. Adjusted dimensions to ({lines}x{samples}x{bands}) to match file size.")
            else:
                # If we can't maintain aspect ratio, just use a simple approach to determine dimensions
                force_adjust = True
        else:
            # If we can't evenly divide by bands, try a more aggressive approach
            force_adjust = True
            
        # Force adjustment as a last resort
        if force_adjust:
            # Calculate total pixels and try to maintain a reasonable aspect ratio
            total_pixels = total_elements // bands
            
            # Try to keep original aspect ratio if possible
            if lines > 0 and samples > 0:
                ratio = lines / samples
                new_samples = int(np.sqrt(total_pixels / ratio))
                new_lines = total_pixels // new_samples
                
                # Ensure we use all available data
                while new_lines * new_samples < total_pixels:
                    new_samples += 1
                while new_lines * new_samples > total_pixels:
                    new_lines -= 1
                    if new_lines <= 0:
                        new_lines = 1
                        break
                
                # print(f"Warning: Forced dimension adjustment to ({new_lines}x{new_samples}x{bands}) to match file size.")
                lines, samples = new_lines, new_samples
                shape = (lines, bands, samples)
                adjusted_dimensions = (lines, samples, bands)
            else:
                print(f"Warning: Cannot calculate dimensions with invalid lines/samples. Using square dimensions.")
                # Just use square dimensions as a fallback
                side = int(np.sqrt(total_elements / bands))
                lines = samples = side
                shape = (lines, bands, samples)
                adjusted_dimensions = (lines, samples, bands)
    
    try:
        # Create a memory map to the file
        fp = np.memmap(file_path, dtype=byte_order + 'u2', mode='r', shape=shape)
        
        # Transpose to get [lines][samples][bands] format, which is more commonly used
        # This matches the MATLAB multibandread behavior with 'bil' format
        return np.transpose(fp, (0, 2, 1)), adjusted_dimensions
        
    except ValueError as e:
        # If still getting an error, provide detailed message
        error_msg = f"Memory mapping error with dimensions ({lines}x{samples}x{bands}). "
        error_msg += f"File size: {actual_size} bytes, expected: {lines*samples*bands*element_size} bytes."
        if 'mmap length' in str(e):
            raise ValueError(error_msg) from e
        else:
            raise e


def visualize_hyperspectral_image(image, wavelengths, output_dir=None):
    """
    Visualize a hyperspectral image in multiple ways
    
    Parameters:
    -----------
    image : numpy.ndarray
        The hyperspectral image as a 3D array (lines, samples, bands)
    wavelengths : numpy.ndarray
        Array of wavelengths corresponding to the bands
    output_dir : str, optional
        Directory to save the visualization plots, if None plots are displayed
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create an RGB composite using wavelengths close to RGB
    rgb_indices = []
    # Find indices for wavelengths closest to red (650nm), green (550nm), and blue (450nm)
    target_wavelengths = [650, 550, 450]
    for target in target_wavelengths:
        idx = np.argmin(np.abs(wavelengths - target))
        rgb_indices.append(idx)
    
    # Create RGB composite
    rgb_composite = np.zeros((image.shape[0], image.shape[1], 3))
    for i, idx in enumerate(rgb_indices):
        # Normalize band to 0-1 range
        band = image[:, :, idx].astype(float)
        min_val = np.percentile(band, 2)  # Use percentiles to avoid outliers
        max_val = np.percentile(band, 98)
        rgb_composite[:, :, i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
    
    # 2. Create false color composite using different bands (e.g., NIR, red, green)
    # Try to find near-infrared band (~800nm)
    nir_idx = np.argmin(np.abs(wavelengths - 800))
    false_color_indices = [nir_idx, rgb_indices[0], rgb_indices[1]]  # NIR, Red, Green
    
    false_color = np.zeros((image.shape[0], image.shape[1], 3))
    for i, idx in enumerate(false_color_indices):
        band = image[:, :, idx].astype(float)
        min_val = np.percentile(band, 2)
        max_val = np.percentile(band, 98)
        false_color[:, :, i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
    
    # 3. Visualize specific bands
    # Select a few interesting bands across the spectrum
    band_indices = []
    target_bands = [400, 500, 600, 700, 800]  # UV, blue, green, red, NIR
    for target in target_bands:
        idx = np.argmin(np.abs(wavelengths - target))
        band_indices.append((idx, f"{wavelengths[idx]:.1f}nm"))
    
    # Create the figure
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3)
    
    # RGB Composite
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(rgb_composite)
    ax1.set_title(f"RGB Composite\n(R:{wavelengths[rgb_indices[0]]:.1f}nm, G:{wavelengths[rgb_indices[1]]:.1f}nm, B:{wavelengths[rgb_indices[2]]:.1f}nm)")
    ax1.axis('off')
    
    # False Color
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(false_color)
    ax2.set_title(f"False Color\n(NIR:{wavelengths[false_color_indices[0]]:.1f}nm, R:{wavelengths[false_color_indices[1]]:.1f}nm, G:{wavelengths[false_color_indices[2]]:.1f}nm)")
    ax2.axis('off')
    
    # Selected bands
    for i, (idx, label) in enumerate(band_indices):
        if i < 4:  # First 4 bands in grid
            row, col = 1, i % 2
            ax = plt.subplot(gs[row, col])
        else:  # Last band in the top right
            ax = plt.subplot(gs[0, 2])
        
        band = image[:, :, idx].astype(float)
        min_val = np.percentile(band, 2)
        max_val = np.percentile(band, 98)
        normalized_band = np.clip((band - min_val) / (max_val - min_val), 0, 1)
        
        # Use a suitable colormap for individual bands
        ax.imshow(normalized_band, cmap='viridis')
        ax.set_title(f"Band: {label}")
        ax.axis('off')
    
    # 4. Add spectral profile of center pixel
    center_x, center_y = image.shape[0] // 2, image.shape[1] // 2
    spectral_profile = image[center_x, center_y, :]
    
    ax_spec = plt.subplot(gs[1, 2])
    ax_spec.plot(wavelengths, spectral_profile)
    ax_spec.set_xlabel('Wavelength (nm)')
    ax_spec.set_ylabel('Intensity')
    ax_spec.set_title(f'Spectral Profile at Pixel ({center_x}, {center_y})')
    ax_spec.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'hyperspectral_visualization.png'), dpi=300)
    else:
        plt.show()
    
    # Create a wavelength-intensity heatmap
    plt.figure(figsize=(12, 8))
    
    # Take a horizontal line through the center of the image
    line_profile = image[center_x, :, :]
    
    # Create a meshgrid for the plot
    x = np.arange(line_profile.shape[0])
    y = wavelengths
    X, Y = np.meshgrid(x, y)
    
    # Transpose line_profile to match meshgrid shape
    Z = line_profile.T
    
    # Normalize Z for better visualization
    vmin = np.percentile(Z, 5)
    vmax = np.percentile(Z, 95)
    
    plt.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap='inferno')
    plt.colorbar(label='Intensity')
    plt.ylabel('Wavelength (nm)')
    plt.xlabel('Pixel Position')
    plt.title('Wavelength-Intensity Profile Along Center Horizontal Line')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'wavelength_intensity_profile.png'), dpi=300)
    else:
        plt.show()


def test_load_and_visualize(image_path, image_name, output_dir=None):
    """
    Test function to load and visualize a hyperspectral image
    
    Parameters:
    -----------
    image_path : str
        Path to the directory containing the hyperspectral image
    image_name : str
        Name of the image file without extension
    output_dir : str, optional
        Directory to save visualization results
    """
    try:
        # Load the hyperspectral image
        image, wavelengths, scan_info, gain = read_iml_hyp(image_path, image_name)
        
        print(f"Successfully loaded hyperspectral image: {os.path.join(image_path, image_name)}")
        print(f"Image shape: {image.shape} (lines, samples, bands)")
        print(f"Wavelength range: {wavelengths.min():.2f}nm to {wavelengths.max():.2f}nm")
        print(f"Scan info: {scan_info}")
        print(f"Gain value: {gain}")
        
        # Visualize the image
        visualize_hyperspectral_image(image, wavelengths, output_dir)
        
        return True
    
    except Exception as e:
        print(f"Error loading or visualizing hyperspectral image: {e}")
        return False


if __name__ == "__main__":
    # Example usage:
    # 1. Load a hyperspectral image
    # image, wavelengths, scan_info, gain_value = read_iml_hyp("/path/to/folder", "filename_without_extension")
    
    # To test visualization:
    test_load_and_visualize("/Volumes/shared/cooney_lab/Shared/data/hyperspectral/lepidoptera", "Abraxas fulvobasalis_1D_040917", "./")
    pass
