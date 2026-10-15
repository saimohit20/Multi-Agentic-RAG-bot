import os
import streamlit as st
import requests
import json
from datetime import datetime
from utils.file_handler import save_uploaded_files

def send_teams_notification(submission_id, metadata):
    """
    Send a notification to Microsoft Teams using a webhook.

    Args:
        submission_id (str): The unique ID of the submission
        metadata (dict): The metadata for the submission

    Returns:
        bool: True if the notification was sent successfully, False otherwise
    """
    try:
        # Get the webhook URL from environment variables
        webhook_url = os.getenv('TEAMS_WEBHOOK_URL')

        # Format timestamp for better readability
        timestamp = datetime.strptime(metadata["timestamp"], "%Y%m%d_%H%M%S")
        formatted_time = timestamp.strftime("%B %d, %Y at %I:%M %p")

        # Create the Teams message card
        teams_message = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "0076D7",
            "summary": f"New Document Submission - {submission_id}",
            "sections": [
                {
                    "activityTitle": "📄 New Document Submission",
                    "activitySubtitle": f"Submitted on {formatted_time}",
                    "activityImage": "https://raw.githubusercontent.com/microsoft/vscode-codicons/main/src/icons/file.svg",
                    "facts": [
                        {
                            "name": "Submission ID:",
                            "value": submission_id
                        },
                        {
                            "name": "Number of Files:",
                            "value": str(len(metadata["file_paths"]))
                        },
                        {
                            "name": "Status:",
                            "value": "Pending Review"
                        },
                        {
                            "name": "Connection Info:",
                            "value": metadata["connection_info"] or "Not provided"
                        }
                    ],
                    "markdown": True
                },
                {
                    "title": "User Comments",
                    "text": metadata["user_comment"] or "No comments provided"
                }
            ],
            "potentialAction": [
                {
                    "@type": "OpenUri",
                    "name": "View in Admin Portal",
                    "targets": [
                        {
                            "os": "default",
                            "uri": os.getenv('ADMIN_PORTAL_URL', "https://available-deployment-eda4exdjd9f9byfs.swedencentral-01.azurewebsites.net/Review_Documents")
                        }
                    ]
                }
            ]
        }

        # Send the notification to Teams
        response = requests.post(
            webhook_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(teams_message)
        )

        # Check if the request was successful
        if response.status_code < 200 or response.status_code >= 300:
            print(f"Teams notification failed with status code {response.status_code}: {response.text}")
            return False

        print(f"Teams notification sent successfully for submission {submission_id}")
        return True

    except Exception as e:
        print(f"Error sending Teams notification: {e}")
        return False

def main():
    st.title("Document Upload Portal")

    with st.form("document_upload_form"):
        # File upload
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'docx', 'txt', 'xlsx'],
            accept_multiple_files=True
        )

        # User inputs
        user_comment = st.text_area("Add your comments")
        connection_info = st.text_input("Connection Information")

        submitted = st.form_submit_button("Submit Documents")

        if submitted and uploaded_files:
            # Create unique submission ID
            submission_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save files and metadata
            file_paths = save_uploaded_files(uploaded_files, submission_id)

            metadata = {
                "submission_id": submission_id,
                "timestamp": timestamp,
                "user_comment": user_comment,
                "connection_info": connection_info,
                "file_paths": file_paths,
                "status": "pending"
            }

            # Save metadata
            temp_dir = os.path.join("main_data_test/temp_uploads", submission_id)
            os.makedirs(temp_dir, exist_ok=True)
            with open(os.path.join(temp_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)

            # Send Teams notification
            if send_teams_notification(submission_id, metadata):
                st.success("Documents submitted successfully! An expert will review them shortly.")
            else:
                st.error("Submission succeeded but failed to send notification")

if __name__ == "__main__":
    main()