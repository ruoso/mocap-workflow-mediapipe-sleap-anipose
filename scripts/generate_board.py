#!/usr/bin/env python3
"""
Generate ChArUco calibration board and multi-format PDFs.

Generates a high-resolution ChArUco board image and creates printable PDFs
for different paper sizes including single-page A3 and multi-page A4/Letter.

Usage:
  generate_board.py [options]

Options:
  --marker-size FLOAT     Marker size in mm (default: 60)
  --squares-x INT         Number of squares horizontally (default: 7)
  --squares-y INT         Number of squares vertically (default: 5)
  --output-name STR       Output base name (default: board)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

try:
    from sleap_anipose.calibration import draw_board
except ImportError:
    print("Error: sleap-anipose not installed")
    print("Install with: pip install sleap-anipose")
    sys.exit(1)


def generate_charuco_board(
    output_path: str,
    squares_x: int = 7,
    squares_y: int = 5,
    marker_length_mm: float = 40.0,
    dpi: int = 300
):
    """
    Generate high-resolution ChArUco board image.

    Args:
        output_path: Path to save the board image
        squares_x: Number of squares horizontally
        squares_y: Number of squares vertically
        marker_length_mm: Size of ArUco markers in millimeters
        dpi: Resolution in dots per inch
    """
    # Convert mm to meters for sleap-anipose
    marker_length_m = marker_length_mm / 1000.0

    # Square should be larger than marker (typically 125-133%)
    square_length_m = marker_length_m * 1.3

    # ArUco dictionary parameters (5x5 bits, 50 markers)
    marker_bits = 5
    dict_size = 50

    # Calculate image size in pixels based on DPI
    # Size in inches = (squares * square_size_mm) / 25.4
    width_mm = squares_x * (square_length_m * 1000)
    height_mm = squares_y * (square_length_m * 1000)

    img_width = int((width_mm / 25.4) * dpi)
    img_height = int((height_mm / 25.4) * dpi)

    print(f"Generating ChArUco board:")
    print(f"  Squares: {squares_x}x{squares_y}")
    print(f"  Square size: {square_length_m*1000:.1f}mm")
    print(f"  Marker size: {marker_length_mm}mm")
    print(f"  Image size: {img_width}x{img_height} pixels ({dpi} DPI)")
    print(f"  Physical size: {width_mm:.1f}x{height_mm:.1f}mm")

    # Generate board configuration file
    config_path = output_path.replace('.png', '.toml').replace('.jpg', '.toml')

    # Draw the board
    draw_board(
        board_name=output_path,
        board_x=squares_x,
        board_y=squares_y,
        square_length=square_length_m,
        marker_length=marker_length_m,
        marker_bits=marker_bits,
        dict_size=dict_size,
        img_width=img_width,
        img_height=img_height,
        save=config_path
    )

    # Convert JPG to PNG if needed
    if output_path.endswith('.png') and not Path(output_path).exists():
        jpg_path = output_path.replace('.png', '.jpg')
        if Path(jpg_path).exists():
            img = Image.open(jpg_path)
            img.save(output_path)
            print(f"  Converted to PNG: {output_path}")

    print(f"✓ Board image saved: {output_path}")
    print(f"✓ Configuration saved: {config_path}")

    return {
        'width_mm': width_mm,
        'height_mm': height_mm,
        'square_length_mm': square_length_m * 1000,
        'marker_length_mm': marker_length_mm,
        'config_path': config_path
    }


def draw_scale_rulers_and_info(c, paper_w_mm, paper_h_mm, board_info, x_start, y_start, width, height, page_info=None, is_single_page=True):
    """
    Draw scale rulers, dimensions, and page info on a PDF canvas.
    Places horizontal rulers at the top alongside the title for better space usage.

    Args:
        c: ReportLab canvas object
        paper_w_mm: Paper width in mm
        paper_h_mm: Paper height in mm
        board_info: Dict with marker_length_mm, square_length_mm
        x_start: X position of board content (in mm)
        y_start: Y position of board content (in mm)
        width: Width of board content (in mm)
        height: Height of board content (in mm)
        page_info: Optional string for page info (e.g., "Page 1/4")
        is_single_page: True for single-page PDFs, False for multi-page
    """
    from reportlab.lib.pagesizes import mm

    # Board specifications at top left
    c.setFont("Helvetica-Bold", 8)

    if board_info:
        marker_mm = board_info.get('marker_length_mm', 0)
        square_mm = board_info.get('square_length_mm', 0)
        if page_info:
            spec_text = f"Marker: {marker_mm:.0f}mm | Square: {square_mm:.0f}mm | {page_info}"
        else:
            spec_text = f"Marker: {marker_mm:.0f}mm | Square: {square_mm:.0f}mm | 100% Scale"
    else:
        spec_text = page_info or "Calibration Board"

    c.drawString(3*mm, (paper_h_mm - 3)*mm, spec_text)

    # Draw scale rulers at the top
    c.setFont("Helvetica", 6)
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)

    # Horizontal ruler at top - millimeters
    ruler_y_mm = (paper_h_mm - 7)*mm
    ruler_start_x = 3*mm
    ruler_length = min(150, paper_w_mm - 10)  # Max 150mm ruler or paper width
    ruler_end_x = ruler_start_x + ruler_length*mm

    c.line(ruler_start_x, ruler_y_mm, ruler_end_x, ruler_y_mm)
    for i in range(0, int(ruler_length) + 1, 10):
        tick_x = ruler_start_x + i*mm
        if tick_x <= ruler_end_x:
            if i % 50 == 0:
                c.line(tick_x, ruler_y_mm, tick_x, ruler_y_mm - 2*mm)
                c.drawString(tick_x - 3*mm, ruler_y_mm + 1*mm, str(i))
            elif i % 10 == 0:
                c.line(tick_x, ruler_y_mm, tick_x, ruler_y_mm - 1*mm)

    c.drawString(ruler_end_x + 1*mm, ruler_y_mm - 0.5*mm, "mm")

    # Horizontal ruler at top - inches
    ruler_y_inch = ruler_y_mm - 4*mm
    ruler_length_inch = min(6, ruler_length / 25.4)  # Corresponding inches
    ruler_end_x_inch = ruler_start_x + ruler_length_inch * 25.4*mm

    c.line(ruler_start_x, ruler_y_inch, ruler_end_x_inch, ruler_y_inch)
    for i in range(0, int(ruler_length_inch) + 1):
        tick_x = ruler_start_x + i * 25.4*mm
        if tick_x <= ruler_end_x_inch:
            c.line(tick_x, ruler_y_inch, tick_x, ruler_y_inch - 2*mm)
            c.drawString(tick_x - 2*mm, ruler_y_inch + 1*mm, f'{i}"')

    c.drawString(ruler_end_x_inch + 1*mm, ruler_y_inch - 0.5*mm, "in")

    # Vertical ruler - millimeters
    if is_single_page:
        # For single-page: place at left margin
        ruler_x = 2*mm
        ruler_start_y = max(12*mm, y_start*mm)
        ruler_end_y = min((paper_h_mm - 15)*mm, (y_start + height)*mm)
    else:
        # For multi-page: place on whichever margin has more space
        left_space = x_start
        right_space = paper_w_mm - (x_start + width)

        if left_space >= 8:
            # Use left margin
            ruler_x = max(2*mm, x_start*mm - 3*mm)
            ruler_start_y = max(12*mm, y_start*mm)
            ruler_end_y = min((paper_h_mm - 15)*mm, (y_start + height)*mm)
        elif right_space >= 8:
            # Use right margin
            ruler_x = (x_start + width + 3)*mm
            ruler_start_y = max(12*mm, y_start*mm)
            ruler_end_y = min((paper_h_mm - 15)*mm, (y_start + height)*mm)
        else:
            # Not enough space for vertical ruler
            ruler_x = None

    if ruler_x is not None:
        c.line(ruler_x, ruler_start_y, ruler_x, ruler_end_y)
        for i in range(0, int(height) + 1, 10):
            tick_y = ruler_start_y + i*mm
            if tick_y <= ruler_end_y:
                if i % 50 == 0:
                    c.line(ruler_x, tick_y, ruler_x + 2*mm, tick_y)
                    c.drawString(ruler_x + 3*mm, tick_y - 1*mm, str(i))
                elif i % 10 == 0:
                    c.line(ruler_x, tick_y, ruler_x + 1*mm, tick_y)


def create_pdf_single_page(image_path: str, output_pdf: str, paper_size: tuple, board_info: dict = None):
    """
    Create a single-page PDF with the board centered.
    Automatically uses landscape orientation if board fits better.
    Adds scale rulers and dimension information.

    Args:
        image_path: Path to board image
        output_pdf: Path to save PDF
        paper_size: (width_mm, height_mm) of paper
        board_info: Optional dict with marker_length_mm, square_length_mm
    """
    from reportlab.lib.pagesizes import mm
    from reportlab.pdfgen import canvas

    paper_w_mm, paper_h_mm = paper_size

    # Load image to get dimensions
    img = Image.open(image_path)
    img_w_px, img_h_px = img.size

    # Assume image was rendered at 300 DPI
    img_w_mm = (img_w_px / 300) * 25.4
    img_h_mm = (img_h_px / 300) * 25.4

    margin_mm = 5  # Safe for most consumer printers

    # Check both portrait and landscape orientations
    # Portrait
    available_w_portrait = paper_w_mm - 2 * margin_mm
    available_h_portrait = paper_h_mm - 2 * margin_mm
    scale_portrait = min(available_w_portrait / img_w_mm, available_h_portrait / img_h_mm)

    # Landscape
    available_w_landscape = paper_h_mm - 2 * margin_mm
    available_h_landscape = paper_w_mm - 2 * margin_mm
    scale_landscape = min(available_w_landscape / img_w_mm, available_h_landscape / img_h_mm)

    # Choose orientation with better scale (closer to 1.0)
    if scale_landscape > scale_portrait:
        # Use landscape
        paper_w_mm, paper_h_mm = paper_h_mm, paper_w_mm
        scale = scale_landscape
        orientation = "landscape"
    else:
        scale = scale_portrait
        orientation = "portrait"

    # Don't upscale
    if scale > 1.0:
        scale = 1.0

    final_w = img_w_mm * scale
    final_h = img_h_mm * scale

    # Center on page
    x = (paper_w_mm - final_w) / 2
    y = (paper_h_mm - final_h) / 2

    # Create PDF
    c = canvas.Canvas(output_pdf, pagesize=(paper_w_mm*mm, paper_h_mm*mm))
    c.drawImage(image_path, x*mm, y*mm, width=final_w*mm, height=final_h*mm)

    # Add rulers and board information using shared function
    draw_scale_rulers_and_info(c, paper_w_mm, paper_h_mm, board_info, x, y, final_w, final_h)

    c.save()
    print(f"✓ PDF saved: {output_pdf}")
    print(f"  Paper: {paper_w_mm}x{paper_h_mm}mm ({orientation}), Scale: {scale*100:.1f}%")


def create_pdf_multipage(image_path: str, output_pdf: str, paper_size: tuple, board_info: dict = None):
    """
    Create a multi-page PDF with the board split across pages.
    Tiles edge-to-edge without overlap for precise assembly.
    Adds scale rulers and dimension information to each page.

    Args:
        image_path: Path to board image
        output_pdf: Path to save PDF
        paper_size: (width_mm, height_mm) of paper
        board_info: Optional dict with marker_length_mm, square_length_mm
    """
    from reportlab.lib.pagesizes import mm
    from reportlab.pdfgen import canvas
    import math

    paper_w_mm, paper_h_mm = paper_size
    margin_mm = 5  # Safe for most consumer printers

    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    img_h_px, img_w_px = img_array.shape[:2]

    # Assume 300 DPI
    dpi = 300
    img_w_mm = (img_w_px / dpi) * 25.4
    img_h_mm = (img_h_px / dpi) * 25.4

    # Calculate tiles needed (1:1 scale, no overlap)
    tile_w_mm = paper_w_mm - 2 * margin_mm
    tile_h_mm = paper_h_mm - 2 * margin_mm

    # Number of tiles needed (edge-to-edge tiling)
    n_tiles_x = math.ceil(img_w_mm / tile_w_mm)
    n_tiles_y = math.ceil(img_h_mm / tile_h_mm)

    print(f"Creating multi-page PDF: {n_tiles_x}x{n_tiles_y} = {n_tiles_x * n_tiles_y} pages")

    # Create PDF
    c = canvas.Canvas(output_pdf, pagesize=(paper_w_mm*mm, paper_h_mm*mm))

    for tile_y in range(n_tiles_y):
        for tile_x in range(n_tiles_x):
            # Calculate region in mm (edge-to-edge, no overlap)
            start_x_mm = tile_x * tile_w_mm
            start_y_mm = tile_y * tile_h_mm
            end_x_mm = min(start_x_mm + tile_w_mm, img_w_mm)
            end_y_mm = min(start_y_mm + tile_h_mm, img_h_mm)

            # Convert to pixels
            start_x_px = int((start_x_mm / 25.4) * dpi)
            start_y_px = int((start_y_mm / 25.4) * dpi)
            end_x_px = int((end_x_mm / 25.4) * dpi)
            end_y_px = int((end_y_mm / 25.4) * dpi)

            # Extract tile (note: image Y is inverted)
            tile_array = img_array[start_y_px:end_y_px, start_x_px:end_x_px]

            # Save tile as temporary image
            tile_img = Image.fromarray(tile_array)
            tile_path = f"/tmp/tile_{tile_y}_{tile_x}.png"
            tile_img.save(tile_path)

            # Add to PDF
            tile_w_mm = (tile_array.shape[1] / dpi) * 25.4
            tile_h_mm = (tile_array.shape[0] / dpi) * 25.4

            # Draw tile (bottom-up coordinate system)
            c.drawImage(tile_path, margin_mm*mm, margin_mm*mm,
                       width=tile_w_mm*mm, height=tile_h_mm*mm)

            # Add rulers and board information using shared function
            page_num = tile_y * n_tiles_x + tile_x + 1
            page_info = f"Page {page_num}/{n_tiles_x * n_tiles_y} | Tile ({tile_x+1},{tile_y+1})"
            draw_scale_rulers_and_info(c, paper_w_mm, paper_h_mm, board_info,
                                      margin_mm, margin_mm, tile_w_mm, tile_h_mm, page_info, is_single_page=False)

            # Draw crop marks at corners
            c.setStrokeColorRGB(0.5, 0.5, 0.5)
            mark_len = 3
            # Top-left
            c.line(margin_mm*mm - mark_len*mm, margin_mm*mm + tile_h_mm*mm,
                   margin_mm*mm, margin_mm*mm + tile_h_mm*mm)
            c.line(margin_mm*mm, margin_mm*mm + tile_h_mm*mm,
                   margin_mm*mm, margin_mm*mm + tile_h_mm*mm + mark_len*mm)
            # Bottom-left
            c.line(margin_mm*mm - mark_len*mm, margin_mm*mm, margin_mm*mm, margin_mm*mm)
            c.line(margin_mm*mm, margin_mm*mm, margin_mm*mm, margin_mm*mm - mark_len*mm)
            # Top-right
            c.line(margin_mm*mm + tile_w_mm*mm, margin_mm*mm + tile_h_mm*mm,
                   margin_mm*mm + tile_w_mm*mm + mark_len*mm, margin_mm*mm + tile_h_mm*mm)
            c.line(margin_mm*mm + tile_w_mm*mm, margin_mm*mm + tile_h_mm*mm,
                   margin_mm*mm + tile_w_mm*mm, margin_mm*mm + tile_h_mm*mm + mark_len*mm)
            # Bottom-right
            c.line(margin_mm*mm + tile_w_mm*mm, margin_mm*mm,
                   margin_mm*mm + tile_w_mm*mm + mark_len*mm, margin_mm*mm)
            c.line(margin_mm*mm + tile_w_mm*mm, margin_mm*mm,
                   margin_mm*mm + tile_w_mm*mm, margin_mm*mm - mark_len*mm)

            c.showPage()

    c.save()
    print(f"✓ Multi-page PDF saved: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate ChArUco calibration board and printable PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--marker-size', type=float, default=40.0,
                       help='Marker size in millimeters (default: 40)')
    parser.add_argument('--squares-x', type=int, default=7,
                       help='Number of squares horizontally (default: 7)')
    parser.add_argument('--squares-y', type=int, default=5,
                       help='Number of squares vertically (default: 5)')
    parser.add_argument('--output-name', default='board',
                       help='Output base name (default: board)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Image resolution in DPI (default: 300)')

    args = parser.parse_args()

    print("="*70)
    print("ChArUco Board Generator")
    print("="*70)
    print()

    # Generate board image
    output_image = f"{args.output_name}.png"
    board_info = generate_charuco_board(
        output_path=output_image,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        marker_length_mm=args.marker_size,
        dpi=args.dpi
    )

    print()
    print("="*70)
    print("Generating PDFs")
    print("="*70)
    print()

    try:
        from reportlab.lib.pagesizes import mm
        from reportlab.pdfgen import canvas
    except ImportError:
        print("Warning: reportlab not installed, skipping PDF generation")
        print("Install with: pip install reportlab")
        return 0

    # Paper sizes in mm
    paper_sizes = {
        'A3': (297, 420),
        'A4': (210, 297),
        'Letter': (215.9, 279.4)
    }

    # Generate A3 single page (will auto-select landscape if better)
    create_pdf_single_page(output_image, f"{args.output_name}_A3.pdf", paper_sizes['A3'], board_info)

    print()

    # For A4 and Letter, check if board fits on single page first
    # (useful for smaller boards or landscape orientation)
    board_w = board_info['width_mm']
    board_h = board_info['height_mm']

    # A4: Check if fits on single page (with auto-orientation)
    a4_w, a4_h = paper_sizes['A4']
    margin = 10  # 5mm margin on each side
    if (board_w <= a4_w - margin and board_h <= a4_h - margin) or \
       (board_w <= a4_h - margin and board_h <= a4_w - margin):
        create_pdf_single_page(output_image, f"{args.output_name}_A4.pdf", paper_sizes['A4'], board_info)
    else:
        # Multi-page A4
        create_pdf_multipage(output_image, f"{args.output_name}_A4_multipage.pdf", paper_sizes['A4'], board_info)

    print()

    # Letter: Check if fits on single page
    letter_w, letter_h = paper_sizes['Letter']
    if (board_w <= letter_w - margin and board_h <= letter_h - margin) or \
       (board_w <= letter_h - margin and board_h <= letter_w - margin):
        create_pdf_single_page(output_image, f"{args.output_name}_Letter.pdf", paper_sizes['Letter'], board_info)
    else:
        # Multi-page Letter
        create_pdf_multipage(output_image, f"{args.output_name}_Letter_multipage.pdf", paper_sizes['Letter'], board_info)

    print()
    print("="*70)
    print("✓ Complete!")
    print("="*70)
    print()
    print("Generated files:")
    print(f"  - {output_image} (High-resolution source image)")
    print(f"  - {board_info['config_path']} (Board configuration)")
    print(f"  - {args.output_name}_A3.pdf (Single page A3)")
    print(f"  - {args.output_name}_A4_multipage.pdf (Multi-page A4)")
    print(f"  - {args.output_name}_Letter_multipage.pdf (Multi-page US Letter)")
    print()
    print("Printing instructions:")
    print("  1. Use single-page PDF for A3 printing (easiest)")
    print("  2. For multi-page: print at 100% scale, do not scale to fit")
    print("  3. Trim pages at crop marks and align edges precisely")
    print("  4. Tape pages together on rigid backing (foam board recommended)")
    print("  5. Measure actual marker size after printing to verify scale")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
