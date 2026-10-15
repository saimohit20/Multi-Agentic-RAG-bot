import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path
from PIL import Image
import fitz

from dc_available.common.helpers import get_operating_system, _check_path


def img_to_bytes(img_or_path: str | Path | Image.Image, encoding: str = "utf-8") -> str:
    if isinstance(img_or_path, Image.Image):
        img_bytes = img_or_path.tobytes()
    else:
        img_bytes = Path(img_or_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode(encoding)
    return encoded


def img_to_html(img_path: str | Path, styling: str) -> str:
    img_html = "<img src='data:image/png;base64,{0}' class='img-fluid' {1}>".format(
        img_to_bytes(img_path), styling
    )
    return img_html


def pdf_to_img_paths(pdf_path: str | Path, output_img_path: str | Path) -> list[Path]:
    assert (
        output_img_path is not None
    ), "You must provide a valid output path for this function"
    os.makedirs(output_img_path, exist_ok=True)
    output_img_path = Path(output_img_path)

    doc = fitz.open(pdf_path)
    image_paths = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap()

        image_path = output_img_path / f"page_{i + 1}.png"
        pix.save(image_path)
        image_paths.append(image_path.resolve())
    return image_paths


def pdf_to_img(
    pdf_path: str | Path, output_img_path: str | Path = None
) -> list[Image.Image]:
    if output_img_path is not None:
        os.makedirs(output_img_path, exist_ok=True)
        image_paths = pdf_to_img_paths(pdf_path, output_img_path)
        return [Image.open(ip) for ip in image_paths]

    else:
        doc = fitz.open(pdf_path)
        images = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap()
            image_path = "delete-me.png"
            pix.save(image_path)
            images.append(Image.open(image_path))
            if image_path == "delete-me.png":
                os.remove(image_path)
        return images


def force_close_powerpoint():
    """Force close all PowerPoint instances"""
    import subprocess
    import time

    print("Forcing PowerPoint to close...")

    # Try multiple methods to ensure PowerPoint is closed
    try:
        # Method 1: taskkill
        subprocess.run(['taskkill', '/F', '/IM', 'POWERPNT.EXE'],
                       capture_output=True,
                       shell=True)
        time.sleep(1)

        # Method 2: Using wmic (more thorough)
        subprocess.run(['wmic', 'process', 'where', 'name=\'POWERPNT.EXE\'', 'delete'],
                       capture_output=True,
                       shell=True)
        time.sleep(1)

        # Method 3: Check if processes still exist
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq POWERPNT.EXE'],
                                capture_output=True,
                                text=True)

        if 'POWERPNT.EXE' in result.stdout:
            # One final forceful attempt
            subprocess.run(['taskkill', '/F', '/IM', 'POWERPNT.EXE', '/T'],
                           capture_output=True,
                           shell=True)

    except Exception as e:
        print(f"Error during PowerPoint cleanup: {str(e)}")


def _ppt_to_pdf_with_powerpoint(
        input_pptx_file: str | Path,
        output_pdf_path: str | Path = None,
        platform: str = None,
) -> Path:
    if platform is None:
        platform = get_operating_system()
    assert platform == "Windows", "You can only use comtypes on Windows."

    if output_pdf_path is None:
        output_pdf_path = tempfile.mkdtemp()

    input_pptx_file = Path(input_pptx_file)
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.mkdir(parents=True, exist_ok=True)

    # Force close any existing PowerPoint instances before starting
    force_close_powerpoint()

    import comtypes.client
    powerpoint = None
    ppt = None

    try:
        print("Starting PowerPoint conversion...")
        powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
        # Don't try to hide PowerPoint as it causes errors

        input_path = str(input_pptx_file.absolute())
        output_path = str((output_pdf_path / (input_pptx_file.stem + ".pdf")).absolute())

        print("Opening presentation...")
        ppt = powerpoint.Presentations.Open(input_path)
        print("Converting to PDF...")
        ppt.SaveAs(output_path, 32)  # 32 = PDF format
        print("Closing presentation...")
        ppt.Close()

        result_path = _check_path(output_pdf_path / (input_pptx_file.stem + ".pdf"))
        print("Conversion completed.")
        return result_path

    except Exception as e:
        print(f"Error during PowerPoint conversion: {str(e)}")
        return None

    finally:
        print("Cleaning up PowerPoint...")
        try:
            if ppt:
                try:
                    ppt.Close()
                except:
                    pass
            if powerpoint:
                try:
                    powerpoint.Quit()
                except:
                    pass
        except:
            pass

        # Force close PowerPoint after conversion
        force_close_powerpoint()


def ppt_to_pdf(
        input_pptx_file: str | Path,
        output_pdf_path: str | Path = None,
        platform: str = None,
) -> Path:
    import time

    if platform is None:
        platform = get_operating_system()
    if output_pdf_path is None:
        output_pdf_path = tempfile.mkdtemp()

    input_pptx_file = Path(input_pptx_file)
    output_pdf_path = Path(output_pdf_path)

    if output_pdf_path.is_dir():
        assigned_output_path = output_pdf_path / (input_pptx_file.stem + ".pdf")
    else:
        assigned_output_path = output_pdf_path
        output_pdf_path = output_pdf_path.parent

    if assigned_output_path.exists():
        return assigned_output_path

    # Ensure clean PowerPoint state before starting
    #force_close_powerpoint()
    time.sleep(5)  # Give system time to clean up

    if platform == "Windows":
        try:
            file_created = _ppt_to_pdf_with_powerpoint(
                input_pptx_file, output_pdf_path, platform
            )
            if file_created:
                if assigned_output_path != file_created:
                    os.rename(file_created, assigned_output_path)
                return assigned_output_path
        except Exception as e:
            print(f"PowerPoint conversion failed: {str(e)}")
            force_close_powerpoint()

    # Use LibreOffice as fallback
    print("Falling back to LibreOffice conversion...")
    file_created = _ppt_to_pdf_with_libreoffice(
        input_pptx_file, output_pdf_path, platform
    )
    if file_created and file_created != assigned_output_path:
        os.rename(file_created, assigned_output_path)
    return assigned_output_path


def _ppt_to_pdf_with_libreoffice(
        input_pptx_file: str | Path,
        output_pdf_path: str | Path = None,
        platform: str = None,
) -> Path:
    if platform is None:
        platform = get_operating_system()
    if output_pdf_path is None:
        output_pdf_path = tempfile.mkdtemp()

    input_pptx_file = Path(input_pptx_file)
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.mkdir(parents=True, exist_ok=True)

    # Determine LibreOffice command based on platform
    if platform == "Windows":
        # Check common Windows installation paths
        possible_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            # ITOM specific paths
            r"C:\itom\LibreOffice\program\soffice.exe",
            r"C:\itom\tools\LibreOffice\program\soffice.exe"
        ]

        libre_office_command = None
        for path in possible_paths:
            if Path(path).exists():
                libre_office_command = f'"{path}"'
                break

        if not libre_office_command:
            print("LibreOffice not found in standard locations.")
            print("Please install LibreOffice from: https://de.libreoffice.org/download/download/")
            return None

    elif platform == "Mac":
        libre_office_command = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    else:
        libre_office_command = "libreoffice"

    print(f"Using LibreOffice command: {libre_office_command}")

    # Build and execute command
    command = f'{libre_office_command} --headless --convert-to pdf --outdir "{output_pdf_path.as_posix()}" "{input_pptx_file.as_posix()}"'

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        out, error = process.communicate()

        # Check for errors
        if error:
            error_text = error.lower()
            if "command not found" in error_text or "no such file" in error_text:
                print("LibreOffice not found. Please install LibreOffice:")
                if platform == "Mac":
                    print("brew install --cask libreoffice")
                elif platform == "Windows":
                    print("Download from: https://de.libreoffice.org/download/download/")
                else:
                    print(
                        "Install using your package manager or download from: https://de.libreoffice.org/download/download/")
                return None
            else:
                print(f"Warning during conversion: {error}")

        # Check if output file exists
        output_file = output_pdf_path / (input_pptx_file.stem + ".pdf")
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"Successfully created PDF: {output_file}")
            return _check_path(output_file)
        else:
            print(f"Error: Output file not created or empty: {output_file}")
            return None

    except Exception as e:
        print(f"Error during LibreOffice conversion: {str(e)}")
        return None


def video_to_audio(
    input_video_file: str | Path,
    output_audio_path: str | Path = None,
    platform: str = None,
) -> Path:
    if platform is None:
        platform = get_operating_system()

    return _video_to_audio_with_vlc(input_video_file, output_audio_path, platform)


def _video_to_audio_with_vlc(
        input_video_file: str | Path,
        output_audio_path: str | Path = None,
        platform: str = None,
) -> Path:
    if platform is None:
        platform = get_operating_system()
    input_video_file = Path(input_video_file)
    if output_audio_path is None:
        output_audio_path = Path(tempfile.mkdtemp())
    else:
        output_audio_path = Path(output_audio_path)

    if output_audio_path.is_dir():
        output_audio_path.mkdir(parents=True, exist_ok=True)
        output_audio_path /= input_video_file.stem + ".mp3"

    # Remove existing empty file if any
    if output_audio_path.exists() and output_audio_path.stat().st_size == 0:
        output_audio_path.unlink()
    elif output_audio_path.exists() and output_audio_path.stat().st_size > 0:
        print(f"Using existing file: {output_audio_path}")
        return output_audio_path

    # Verify VLC exists
    if platform == "Windows":
        vlc_command = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
        if not Path(vlc_command).exists():
            vlc_command = r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"
            if not Path(vlc_command).exists():
                print("VLC not found in standard locations!")
                return None

    print(f"\nStarting conversion process:")
    print(f"Input file: {input_video_file}")
    print(f"Output file: {output_audio_path}")
    print(f"Using VLC: {vlc_command}\n")

    try:
        # Use list format for command arguments
        command = [
            vlc_command,
            str(input_video_file.absolute()),  # Input file
            "--no-repeat",  # Don't repeat the playback
            "--no-loop",  # Don't loop the playback
            "--sout",
            f"#transcode{{acodec=mp3,ab=128,channels=2,samplerate=44100}}:standard{{access=file,mux=mp3,dst={output_audio_path.absolute()}}}",
            "--no-sout-video",  # Disable video output
            "--no-sout-rtp-sap",  # Disable streaming options
            "--no-sout-standard-sap",
            "--sout-keep",  # Keep the stream output open
            "vlc://quit"  # Quit after playback
        ]

        print("Starting VLC process with command:")
        print(" ".join(command))

        # Start VLC process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("VLC process started. Monitoring conversion...")

        # Monitor conversion progress
        start_time = time.time()
        last_size = 0
        no_progress_count = 0

        while process.poll() is None:  # While process is running
            time.sleep(2)  # Check every 2 seconds

            if output_audio_path.exists():
                try:
                    current_size = output_audio_path.stat().st_size
                    if current_size > last_size:
                        print(f"Converting... Size: {current_size / 1024 / 1024:.2f} MB")
                        last_size = current_size
                        no_progress_count = 0
                    else:
                        no_progress_count += 1
                except Exception as e:
                    print(f"Error checking file size: {e}")

            # Check for timeout
            if time.time() - start_time > 1200:  # 5 minutes timeout
                print("Conversion timeout reached")
                process.terminate()
                break

        # Wait for process to finish and get output
        stdout, stderr = process.communicate()
        print("\nVLC Output:", stdout if stdout else "No output")
        if stderr:
            print("VLC Errors:", stderr)

        # Verify the output file
        if output_audio_path.exists() and output_audio_path.stat().st_size > 0:
            print(f"Successfully created audio file: {output_audio_path}")
            return output_audio_path
        else:
            print("Failed to create audio file")
            return None

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        if 'process' in locals():
            process.terminate()
        return None
