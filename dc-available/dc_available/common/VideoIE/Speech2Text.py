import ast
import re
import os
import time
import subprocess
import tempfile
import glob
from pathlib import Path
import sys
import math
from typing import Union, List, Dict, Any
from openai import OpenAI

from .AzureClient import get_openai_client


def generate_speech_transcription(
        audio_file: Union[str, Path, object],
        client: OpenAI = None,
        model: str = "whisper",
        max_file_size_mb: float = 10,
        force_direct_processing: bool = False
) -> Any:
    """
    Generate speech transcription with improved handling of larger files

    Args:
        audio_file: Path to audio file or file object
        client: OpenAI client instance (optional)
        model: Model to use for transcription (default: "whisper")
        max_file_size_mb: Maximum file size in MB before using chunking (default: 10)
        force_direct_processing: If True, will process directly even if file is large

    Returns:
        Transcription result
    """
    if client is None:
        client = get_openai_client()

    # Check file size before processing
    file_size_bytes = 0
    file_path = ""
    is_file_object = False

    # Handle different input types
    if isinstance(audio_file, (str, Path)):
        file_path = str(audio_file)
        file_size_bytes = os.path.getsize(file_path)
    elif hasattr(audio_file, 'seek') and hasattr(audio_file, 'tell'):
        # Handle file object
        is_file_object = True
        current_pos = audio_file.tell()
        audio_file.seek(0, os.SEEK_END)
        file_size_bytes = audio_file.tell()
        audio_file.seek(current_pos)  # Reset position

    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Processing file of size: {file_size_mb:.2f} MB")

    # Automatically use chunking for large files unless explicitly told not to
    if file_size_mb > max_file_size_mb and not force_direct_processing:
        print(f"File size {file_size_mb:.2f}MB exceeds {max_file_size_mb}MB limit")
        print("Automatically switching to chunked processing...")

        # For file objects, we need to save to a temporary file first
        if is_file_object:
            print("Creating temporary file from file object")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file_path = temp_file.name

            # Reset file pointer and copy content
            audio_file.seek(0)
            temp_file.write(audio_file.read())
            temp_file.close()

            # Process with the temp file
            try:
                result = process_large_audio_file(temp_file_path, max_file_size_mb, client, model)
                return result
            finally:
                # Clean up the temp file
                try:
                    os.unlink(temp_file_path)
                except:
                    print(f"Warning: Failed to remove temporary file {temp_file_path}")
        else:
            # If it's a file path, use it directly
            return process_large_audio_file(file_path, max_file_size_mb, client, model)

    # Normal processing for small files or when forced
    try:
        # Start transcription based on input type
        print("Starting direct transcription request...")

        if isinstance(audio_file, (str, Path)):
            with open(audio_file, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
        else:
            # If it's already a file object
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        print("Transcription completed successfully")
        return transcription

    except Exception as e:
        print(f"Transcription error: {type(e).__name__}: {str(e)}")

        # If we encounter specific errors, and we're not already forced, try the chunked method
        if not force_direct_processing and ("too large" in str(e).lower() or
                                            "timeout" in str(e).lower() or "connection" in str(e).lower()):
            print("Detected API limitation. Automatically switching to chunked processing...")

            # Similar handling for file objects vs paths
            if is_file_object:
                print("Creating temporary file from file object")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file_path = temp_file.name

                # Reset file pointer and copy content
                audio_file.seek(0)
                temp_file.write(audio_file.read())
                temp_file.close()

                # Process with the temp file
                try:
                    result = process_large_audio_file(temp_file_path, max_file_size_mb, client, model)
                    return result
                finally:
                    # Clean up the temp file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        print(f"Warning: Failed to remove temporary file {temp_file_path}")
            else:
                # If it's a file path, use it directly
                return process_large_audio_file(file_path, max_file_size_mb, client, model)
        else:
            # Re-raise the exception if we can't handle it
            raise Exception(f"Transcription failed: {str(e)}")


def process_large_audio_file(
        file_path: Union[str, Path],
        max_size_mb: float = 10,
        client: OpenAI = None,
        model: str = "whisper",
        segment_time: int = 300  # segment time in seconds (5 minutes)
) -> Any:
    """
    Handles large audio files by splitting them into smaller chunks
    Uses FFmpeg instead of VLC for better reliability

    Args:
        file_path: Path to the audio file
        max_size_mb: Maximum size in MB for each chunk
        client: OpenAI client instance
        model: Whisper model to use
        segment_time: Time in seconds for each audio segment

    Returns:
        Combined transcription or list of individual transcriptions
    """
    file_path = str(file_path)  # Convert Path to string if needed
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if file_size_mb <= max_size_mb:
        # File is small enough, process normally
        print(f"File size ({file_size_mb:.2f} MB) is under the limit. Processing normally.")
        return generate_speech_transcription(file_path, client, model, force_direct_processing=True)

    print(f"File size ({file_size_mb:.2f} MB) exceeds the {max_size_mb} MB limit. Splitting into chunks.")

    # Create temporary directory for chunks
    with tempfile.TemporaryDirectory() as temp_dir:
        # Try to get audio duration using FFmpeg first, fallback to VLC
        try:
            duration_seconds = get_audio_duration_ffmpeg(file_path)
            if duration_seconds is None:
                duration_seconds = get_audio_duration_vlc(file_path)
                if duration_seconds is None:
                    duration_seconds = estimate_duration_from_filesize(file_path, file_size_mb)
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            duration_seconds = estimate_duration_from_filesize(file_path, file_size_mb)

        print(f"Audio duration: {duration_seconds} seconds")

        # Create chunks directory
        chunks_dir = os.path.join(temp_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)

        # Split the file using FFmpeg (preferred) or VLC
        chunk_files = split_audio_file(file_path, chunks_dir, segment_time, duration_seconds)

        # Filter out invalid chunks
        valid_chunk_files = []
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 1024:  # At least 1KB
                valid_chunk_files.append(chunk_file)
            else:
                print(
                    f"Skipping invalid chunk: {chunk_file} (size: {os.path.getsize(chunk_file) if os.path.exists(chunk_file) else 'N/A'})")

        if not valid_chunk_files:
            raise ValueError("No valid audio chunks were created. Check your audio file and splitting method.")

        print(f"Processing {len(valid_chunk_files)} valid chunks")

        # Process each chunk with rate limiting
        all_transcriptions = []
        failed_chunks = 0

        for i, chunk_file in enumerate(valid_chunk_files):
            print(f"Processing chunk {i + 1}/{len(valid_chunk_files)}")

            # Add delay to comply with rate limit (3 requests per minute)
            if i > 0:
                print("Waiting 20 seconds to respect rate limits...")
                time.sleep(20)

            try:
                # Verify chunk is valid before processing
                chunk_size_mb = os.path.getsize(chunk_file) / (1024 * 1024)
                print(f"Chunk {i + 1} size: {chunk_size_mb:.2f} MB")

                if chunk_size_mb < 0.001:  # Less than 1KB
                    print(f"Skipping chunk {i + 1}: too small ({chunk_size_mb:.4f} MB)")
                    failed_chunks += 1
                    continue

                # Process this chunk
                chunk_transcription = generate_speech_transcription(
                    chunk_file, client, model, force_direct_processing=True
                )
                all_transcriptions.append(chunk_transcription)
                print(f"Successfully transcribed chunk {i + 1}")
            except Exception as e:
                print(f"Error transcribing chunk {i + 1}: {e}")
                failed_chunks += 1
                print("Continuing with next chunk...")

        # Handle combining the results
        if not all_transcriptions:
            raise ValueError(
                f"No transcriptions were generated from the audio chunks. Failed chunks: {failed_chunks}/{len(valid_chunk_files)}")

        print(f"Successfully processed {len(all_transcriptions)} chunks, {failed_chunks} failed")

        # Combine transcriptions
        return combine_transcriptions(all_transcriptions, segment_time)


def get_audio_duration_ffmpeg(file_path: str) -> float:
    """Get audio duration using FFmpeg (more reliable than VLC)"""
    try:
        # Try to find ffmpeg or ffprobe
        ffprobe_cmd = None
        for cmd in ['ffprobe', 'ffmpeg']:
            try:
                subprocess.run([cmd, '-version'], check=True, capture_output=True)
                if cmd == 'ffmpeg':
                    ffprobe_cmd = [cmd, '-i', file_path, '-f', 'null', '-']
                else:
                    ffprobe_cmd = [cmd, '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', file_path]
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        if ffprobe_cmd is None:
            return None

        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)

        if 'ffprobe' in ffprobe_cmd[0]:
            # ffprobe output
            duration = float(result.stdout.strip())
        else:
            # ffmpeg output - parse from stderr
            duration_match = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', result.stderr)
            if duration_match:
                hours, minutes, seconds, centiseconds = map(int, duration_match.groups())
                duration = hours * 3600 + minutes * 60 + seconds + centiseconds / 100
            else:
                return None

        print(f"FFmpeg detected duration: {duration} seconds")
        return duration

    except Exception as e:
        print(f"FFmpeg duration detection failed: {e}")
        return None


def get_audio_duration_vlc(file_path: str) -> float:
    """Get audio duration using VLC (fallback method)"""
    try:
        vlc_path = find_vlc_path()
        if not vlc_path:
            return None

        duration_cmd = [
            vlc_path,
            "-vvv",
            "-I", "dummy",
            file_path,
            "--play-and-exit"
        ]

        result = subprocess.run(
            duration_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30
        )

        duration_match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+)(?:\.(\d+))?",
            result.stdout,
            re.IGNORECASE
        )

        if duration_match:
            hours, minutes, seconds = map(int, duration_match.groups()[:3])
            total_duration = hours * 3600 + minutes * 60 + seconds
            print(f"VLC detected duration: {total_duration} seconds")
            return total_duration

    except Exception as e:
        print(f"VLC duration detection failed: {e}")

    return None


def split_audio_file(file_path: str, output_dir: str, segment_time: int, total_duration: float) -> List[str]:
    """Split audio file using FFmpeg (preferred) or VLC"""

    # Try FFmpeg first
    chunk_files = split_with_ffmpeg(file_path, output_dir, segment_time, total_duration)
    if chunk_files:
        return chunk_files

    # Fallback to VLC
    print("FFmpeg failed, trying VLC...")
    return split_with_vlc(file_path, output_dir, segment_time, total_duration)


def split_with_ffmpeg(file_path: str, output_dir: str, segment_time: int, total_duration: float) -> List[str]:
    """Split audio using FFmpeg"""
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)

        chunk_files = []
        num_chunks = math.ceil(total_duration / segment_time)

        for i in range(num_chunks):
            start_time = i * segment_time
            output_file = os.path.join(output_dir, f"chunk_{i:03d}.mp3")
            chunk_files.append(output_file)

            # Use FFmpeg to extract segment
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # -y to overwrite existing files
                '-i', file_path,
                '-ss', str(start_time),
                '-t', str(segment_time),
                '-c:a', 'mp3',  # Ensure MP3 output
                '-b:a', '128k',  # Set bitrate
                output_file
            ]

            print(f"Creating chunk {i + 1}/{num_chunks} using FFmpeg...")

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                print(f"FFmpeg failed for chunk {i + 1}: {result.stderr}")
                return []  # Return empty list to trigger VLC fallback

            # Verify the chunk was created and has reasonable size
            if not os.path.exists(output_file) or os.path.getsize(output_file) < 1024:
                print(f"FFmpeg created invalid chunk {i + 1}")
                return []

        print(f"Successfully created {len(chunk_files)} chunks with FFmpeg")
        return chunk_files

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("FFmpeg not available or failed")
        return []


def split_with_vlc(file_path: str, output_dir: str, segment_time: int, total_duration: float) -> List[str]:
    """Split audio using VLC (fallback method with improved commands)"""
    try:
        vlc_path = find_vlc_path()
        if not vlc_path:
            raise FileNotFoundError("VLC not found")

        chunk_files = []
        num_chunks = math.ceil(total_duration / segment_time)

        for i in range(num_chunks):
            start_time = i * segment_time
            end_time = min(start_time + segment_time, total_duration)
            output_file = os.path.join(output_dir, f"chunk_{i:03d}.mp3")
            chunk_files.append(output_file)

            print(f"Creating chunk {i + 1}/{num_chunks} using VLC...")

            # Improved VLC command
            vlc_cmd = [
                vlc_path,
                "-I", "dummy",  # No interface
                "--no-video",  # Audio only
                "--play-and-exit",
                f"--start-time={start_time}",
                f"--stop-time={end_time}",
                f"--sout=#transcode{{acodec=mp3,ab=128}}:std{{access=file,mux=mp3,dst={output_file}}}",
                file_path
            ]

            try:
                result = subprocess.run(
                    vlc_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                # VLC might return non-zero even on success, so check file existence
                if not os.path.exists(output_file) or os.path.getsize(output_file) < 1024:
                    print(f"VLC failed to create valid chunk {i + 1}")
                    # Try alternative VLC command
                    alternative_vlc_extract(file_path, output_file, start_time, end_time)

            except subprocess.TimeoutExpired:
                print(f"VLC timeout for chunk {i + 1}")
                continue

        # Filter valid chunks
        valid_chunks = [f for f in chunk_files if os.path.exists(f) and os.path.getsize(f) > 1024]
        print(f"VLC created {len(valid_chunks)} valid chunks out of {len(chunk_files)}")

        return valid_chunks

    except Exception as e:
        print(f"VLC splitting failed: {e}")
        return []


def alternative_vlc_extract(input_file: str, output_file: str, start_time: float, end_time: float):
    """Alternative VLC extraction method"""
    vlc_path = find_vlc_path()
    if not vlc_path:
        return

    try:
        # Different VLC approach
        alt_cmd = [
            vlc_path,
            input_file,
            f"--start-time={start_time}",
            f"--stop-time={end_time}",
            "--intf=dummy",
            "--no-video",
            f"--sout=#std{{access=file,mux=mp3,dst={output_file}}}",
            "--play-and-exit"
        ]

        subprocess.run(alt_cmd, capture_output=True, timeout=300)

    except Exception as e:
        print(f"Alternative VLC method failed: {e}")


def combine_transcriptions(transcriptions: List[Any], segment_time: int) -> Dict[str, Any]:
    """Combine multiple transcription results into a single result"""
    if not transcriptions:
        return {}

    # Combine texts
    combined_text = " ".join(t.text for t in transcriptions if hasattr(t, 'text'))

    # Combine segments with adjusted timestamps
    combined_segments = []
    time_offset = 0

    for i, trans in enumerate(transcriptions):
        if not hasattr(trans, 'segments'):
            continue

        # Adjust segment timestamps
        for segment in trans.segments:
            # Create a copy of the segment with adjusted timestamps
            segment_data = {}

            # Extract segment attributes
            for attr in ['id', 'seek', 'start', 'end', 'text', 'tokens', 'temperature', 'avg_logprob',
                         'compression_ratio', 'no_speech_prob']:
                if hasattr(segment, attr):
                    segment_data[attr] = getattr(segment, attr)

            # Adjust the timestamps
            if 'start' in segment_data:
                segment_data['start'] = segment_data['start'] + time_offset
            if 'end' in segment_data:
                segment_data['end'] = segment_data['end'] + time_offset

            combined_segments.append(segment_data)

        # Update time offset for next chunk
        time_offset += segment_time

    # Create combined result
    return {
        "text": combined_text,
        "segments": combined_segments,
        "language": transcriptions[0].language if hasattr(transcriptions[0], 'language') else "unknown",
    }


def find_vlc_path():
    """Find the VLC executable based on the operating system"""
    if sys.platform.startswith('win'):
        # Common Windows installation paths
        potential_paths = [
            os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'VideoLAN', 'VLC', 'vlc.exe'),
            os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'VideoLAN', 'VLC', 'vlc.exe'),
        ]
        for path in potential_paths:
            if os.path.exists(path):
                return path

        # Try to find it in PATH
        try:
            result = subprocess.run(['where', 'vlc'], check=True, capture_output=True, text=True)
            paths = result.stdout.strip().split('\n')
            if paths and os.path.exists(paths[0]):
                return paths[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    elif sys.platform.startswith('darwin'):  # macOS
        potential_paths = [
            '/Applications/VLC.app/Contents/MacOS/VLC',
            os.path.expanduser('~/Applications/VLC.app/Contents/MacOS/VLC')
        ]
        for path in potential_paths:
            if os.path.exists(path):
                return path

    else:  # Linux and other Unix-like
        try:
            result = subprocess.run(['which', 'vlc'], check=True, capture_output=True, text=True)
            path = result.stdout.strip()
            if path and os.path.exists(path):
                return path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Common Linux paths
        potential_paths = [
            '/usr/bin/vlc',
            '/usr/local/bin/vlc'
        ]
        for path in potential_paths:
            if os.path.exists(path):
                return path

    return None


def format_time_for_vlc(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_duration_from_filesize(file_path, file_size_mb=None):
    """Estimate audio duration based on file size and type"""
    if file_size_mb is None:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # Check file extension to make better estimates
    ext = os.path.splitext(file_path)[1].lower()

    # Improved estimations
    if ext in ['.mp3']:
        # ~1MB per minute for 128kbps MP3
        estimated_duration = int(file_size_mb * 60)
    elif ext in ['.wav']:
        # Uncompressed audio - ~10MB per minute for 16-bit stereo 44.1kHz
        estimated_duration = int(file_size_mb * 6)
    elif ext in ['.m4a', '.aac']:
        # ~0.7MB per minute for typical AAC
        estimated_duration = int(file_size_mb * 85)
    else:
        # Generic fallback
        estimated_duration = int(file_size_mb * 60)

    print(f"Estimated duration based on file size: {estimated_duration} seconds")
    return estimated_duration


def read_in_transcript_file(transcript_file):
    """Read in a transcript file and extract the text, language, and time segments"""
    try:
        with open(transcript_file, 'r', encoding="utf-8") as file:
            content = file.read()

            # Extract the text from the "text" field
            text_pattern = re.compile(r"text=(?:\"|')(.*?)(?:\"|'), segments=", re.DOTALL)
            text_match = text_pattern.search(content)
            transcript_text = text_match.group(1) if text_match else ""

            # Extract the language from the "language" field
            language_pattern = re.compile(r"language=(?:\"|')(.*?)(?:\"|')")
            language_match = language_pattern.search(content)
            language = language_match.group(1) if language_match else "unknown"

            # Extract the segments
            segments_pattern = re.compile(
                r"TranscriptionSegment\((.*?)\), TranscriptionSegment|TranscriptionSegment\((.*?)\)\]", re.DOTALL)
            segments_matches = segments_pattern.findall(content)

            time_segments = []
            for match in segments_matches:
                segment = match[0] if match[0] else match[1]
                segment_dict = {}
                segment_parts = re.findall(r"(\w+)=(-?\d+(?:\.\d+)?|(?:\".*?\")|(?:'.*?')),", segment)
                for key, value in segment_parts:
                    # Remove quotes from string values
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1].strip()
                    elif value.startswith('"') and value.endswith('"'):
                        value = value[1:-1].strip()
                    # Convert numerical values to their appropriate types
                    elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    # Convert list values
                    elif value.startswith("[") and value.endswith("]"):
                        value = eval(value)
                    segment_dict[key] = value
                time_segments.append(segment_dict)

            return transcript_text, language, str(time_segments)
    except FileNotFoundError:
        print("The transcript file was not found. Please check again.")
        return None, None, None
    except Exception as e:
        print(f"Error reading transcript file: {e}")
        return None, None, None


def extract_sentences(segments):
    """Extract complete sentences from transcript segments"""
    sentences_with_times = []
    incomplete_sentence = ""
    incomplete_sentence_start_time = 0
    incomplete_sentence_words = 0

    for i, segment in enumerate(segments):
        segment_text = incomplete_sentence + segment['text']
        segment_start = incomplete_sentence_start_time if incomplete_sentence else segment['start']
        segment_end = segment['end']
        segment_words = incomplete_sentence_words + len(segment['text'].split())

        sentences = re.split(r'(?<=[.!?])\s+', segment_text)

        # Check if the last part of the segment is a complete sentence
        if sentences[-1][-1] not in '.!?':
            incomplete_sentence = sentences.pop()
            incomplete_sentence_start_time = segment_start + (
                    len(segment_text.split()) - len(incomplete_sentence.split())) * (
                                                     (segment_end - segment_start) / segment_words)
            incomplete_sentence_words = len(incomplete_sentence.split())
        else:
            incomplete_sentence = ""
            incomplete_sentence_start_time = 0
            incomplete_sentence_words = 0

        current_time = segment_start
        for sentence in sentences:
            num_words = len(sentence.split())
            sentence_duration = num_words * ((segment_end - segment_start) / segment_words)
            sentence_end_time = current_time + sentence_duration

            sentences_with_times.append({
                'text': sentence.strip(),
                'start_time': current_time,
                'end_time': sentence_end_time
            })

            current_time = sentence_end_time

    # If the last segment ends with an incomplete sentence, add it to the list
    if incomplete_sentence:
        sentences_with_times.append({
            'text': incomplete_sentence.strip(),
            'start_time': incomplete_sentence_start_time,
            'end_time': segments[-1]['end']
        })

    return sentences_with_times


def save_transcription_to_file(transcription, output_file):
    """Save transcription result to a file"""
    with open(output_file, 'w', encoding="utf-8") as f:
        f.write(str(transcription))
    print(f"Transcription saved to {output_file}")


def save_sentences_to_file(sentences, output_file):
    """Save extracted sentences to a file"""
    with open(output_file, 'w', encoding="utf-8") as out:
        for sen in sentences:
            out.write(f"Sentence: {sen['text']} \n(Start: {sen['start_time']} - End: {sen['end_time']})\n\n")
    print(f"Sentences saved to {output_file}")


if __name__ == "__main__":
    # Example usage:

    # For processing an audio file (automatically handles large files)
    # audio_file = "audio_file.mp3"
    # transcription = generate_speech_transcription(audio_file)
    # save_transcription_to_file(transcription, "transcript_output.txt")

    # For processing a file object
    # with open("audio_file.mp3", "rb") as audio_file:
    #     transcription = generate_speech_transcription(audio_file)
    #     save_transcription_to_file(transcription, "transcript_output.txt")

    # For processing a transcript that's already been generated
    transcript_file = "audio\\transcript_audio5.txt"
    transcript_text, language, time_segments = read_in_transcript_file(transcript_file)

    if transcript_text and time_segments:
        segments = ast.literal_eval(time_segments)
        sentences = extract_sentences(segments)
        save_sentences_to_file(sentences, "test_out2.txt")