import os
import shutil
import logging

# Setup logging
logging.basicConfig(
    filename='file_handler.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def save_uploaded_files(uploaded_files, submission_id, temp_root="main_data_test/temp_uploads"):
    """
    Saves uploaded files to a temporary directory.

    Args:
        uploaded_files (list): List of uploaded file objects.
        submission_id (str): Unique identifier for the submission.
        temp_root (str): Root directory for temporary uploads.

    Returns:
        list: List of saved file paths.
    """
    temp_dir = os.path.join(temp_root, submission_id)
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Created temporary directory at {temp_dir}")

    saved_paths = []
    for file in uploaded_files:
        # Sanitize the filename to prevent path traversal
        safe_filename = os.path.basename(file.name)
        file_path = os.path.join(temp_dir, safe_filename)
        try:
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(file_path)
            logger.info(f"Saved file {safe_filename} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save file {safe_filename}: {e}")
            raise e  # Re-raise exception after logging

    return saved_paths

def get_unique_folder_name(target_dir, desired_name):
    """
    Generates a unique folder name within the target directory.
    If the desired_name exists, appends a number to make it unique.

    Args:
        target_dir (str): The directory in which to create the folder.
        desired_name (str): The desired folder name.

    Returns:
        str: A unique folder name.
    """
    base_name = desired_name
    counter = 1
    unique_name = base_name

    while os.path.exists(os.path.join(target_dir, unique_name)):
        unique_name = f"{base_name}_{counter}"
        counter += 1

    return unique_name

def move_files(submission_id, topic, level, training_folder_name, temp_root="main_data_test/temp_uploads", target_root="main_data_test/DC_data"):
    """
    Moves files from the temporary directory to the structured main directory.

    Args:
        submission_id (str): Unique identifier for the submission.
        topic (str): Topic folder name.
        level (str): Level folder name.
        training_folder_name (str): Training folder name.
        temp_root (str): Root directory for temporary uploads.
        target_root (str): Root directory for main uploads.

    Raises:
        Exception: If moving files fails.
    """
    source = os.path.join(temp_root, submission_id)
    topic = topic.strip().replace(" ", "_")
    level = level.strip().replace(" ", "_")
    training_folder_name = training_folder_name.strip().replace(" ", "_")

    # Define the target directory
    target_dir = os.path.join(target_root, topic, level)
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Ensured target directory exists at {target_dir}")

    # Determine a unique training folder name
    training_folder = get_unique_folder_name(target_dir, training_folder_name)
    final_target_dir = os.path.join(target_dir, training_folder)
    os.makedirs(final_target_dir, exist_ok=True)
    logger.info(f"Training folder set to {final_target_dir}")

    # Move files with conflict resolution
    for item in os.listdir(source):
        if item == "metadata.json":  # Skip metadata file
            continue

        s = os.path.join(source, item)
        safe_filename = os.path.basename(item)
        dest = os.path.join(final_target_dir, safe_filename)

        # Handle existing files by appending a counter
        counter = 1
        while os.path.exists(dest):
            name, ext = os.path.splitext(safe_filename)
            dest = os.path.join(final_target_dir, f"{name}_{counter}{ext}")
            counter += 1

        try:
            shutil.move(s, dest)
            logger.info(f"Moved file {safe_filename} to {dest}")
        except Exception as e:
            logger.error(f"Failed to move file {safe_filename} to {dest}: {e}")
            raise e  # Re-raise exception after logging

    # Remove source directory after moving files
    try:
        shutil.rmtree(source)
        logger.info(f"Removed temporary directory {source}")
    except Exception as e:
        logger.error(f"Failed to remove temporary directory {source}: {e}")
        raise e  # Re-raise exception after logging

def get_existing_topics(main_folder, excluded_folders):
    """Retrieve existing main folder topics, excluding specified folders."""
    try:
        all_folders = [
            name for name in os.listdir(main_folder)
            if os.path.isdir(os.path.join(main_folder, name)) and name not in excluded_folders
        ]
        return sorted(all_folders)  # Sort alphabetically for better UX
    except Exception as e:
        logger.error(f"Error accessing main folder topics: {e}")
        return e
def move_files_rejected(submission_id, rejected_dir="main_data_test/rejected_documents"):
    """
    Moves files from the temporary directory to the rejected directory.

    Args:
        submission_id (str): Unique identifier for the submission.
        rejected_dir (str): Directory to store rejected submissions.

    Raises:
        Exception: If moving files fails.
    """
    source = os.path.join("main_data_test/temp_uploads", submission_id)
    training_folder_name = f"Rejected_{submission_id}"

    # Define the target directory
    target_dir = os.path.join(rejected_dir, training_folder_name)
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Ensured rejected directory exists at {target_dir}")

    # Move files with conflict resolution
    for item in os.listdir(source):
        if item == "metadata.json":  # Skip metadata file
            continue

        s = os.path.join(source, item)
        safe_filename = os.path.basename(item)
        dest = os.path.join(target_dir, safe_filename)

        # Handle existing files by appending a counter
        counter = 1
        while os.path.exists(dest):
            name, ext = os.path.splitext(safe_filename)
            dest = os.path.join(target_dir, f"{name}_{counter}{ext}")
            counter += 1

        try:
            shutil.move(s, dest)
            logger.info(f"Moved file {safe_filename} to {dest} (Rejected)")
        except Exception as e:
            logger.error(f"Failed to move file {safe_filename} to {dest} (Rejected): {e}")
            raise e  # Re-raise exception after logging

    # Remove source directory after moving files
    try:
        shutil.rmtree(source)
        logger.info(f"Removed temporary directory {source} after rejection")
    except Exception as e:
        logger.error(f"Failed to remove temporary directory {source} after rejection: {e}")
        raise e  # Re-raise exception after logging