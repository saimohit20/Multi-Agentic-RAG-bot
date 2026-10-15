# AVAILABLE Chatbot

This project implements an intelligent chatbot application designed to provide information from various document types using Retrieval-Augmented Generation (RAG) and advanced summarization techniques. The application features a Streamlit-based frontend, multi-modal content processing (PDFs, PowerPoints, videos, emails), and language detection capabilities.

## Features

*   **Multi-modal Content Processing:** Ingests and processes information from diverse sources including PDF documents, PowerPoint presentations, video files (MP4), and email messages (MSG).
*   **Retrieval-Augmented Generation (RAG):** Utilizes a RAG system to retrieve relevant information from processed documents and generate accurate, context-aware responses to user queries.
*   **Advanced Summarization:** Employs various summarization techniques for different content types (text, images, videos) to extract key information and create concise overviews.
*   **Streamlit Frontend:** Provides an interactive and user-friendly web interface for chatbot interaction, document uploading, and review.
*   **AI Agents:** Incorporates AI agents for managing conversational flow, assessing user intent, and orchestrating information retrieval.
*   **Language Detection and Management:** Automatically detects the user's language and adapts responses accordingly.
*   **Document Trust Scoring:** Implements a system to assess and display the trustworthiness of documents.

## Project Structure

The project is organized into several key directories, each responsible for a specific part of the application:

*   `dc_available/`: The main application package containing core logic.
    *   `agents/`: Contains AI agents for handling different tasks like chat and RAG.
    *   `common/`: Provides common utility functions, data loading, data conversion, and language management.
    *   `config/`: Stores configuration settings for data paths, LLMs, and other application parameters.
    *   `frontend/`: Implements the Streamlit user interface, including pages and reusable components.
    *   `RAG/`: Contains the core Retrieval-Augmented Generation (RAG) logic, including data models, QA functions, and retrieval mechanisms.
    *   `summarization/`: Handles the summarization of various content types (images, texts, videos, slides) and trust scoring.
*   `deployment/`: Contains files related to deployment, such as Dockerfiles and Taskfiles.
*   `expriment/`: Contains experimental code or prototypes.
*   `resources/`: Stores static resources, including the available handbook and processed data.

## UML Documentation

### System Architecture Overview

The following diagram illustrates the high-level architecture of the project, showing the main packages and their dependencies.

```mermaid
graph TB
    subgraph "dc_available Package"
        subgraph "User Interface Layer"
            FE[Frontend]
            COMP[Components]
            PAGES[Pages]
            UTILS[Utils]
        end
        
        subgraph "Agent Layer"
            AGENTS[Agents]
            CHAT_AGENT[Chat Agent]
            RAG_AGENT[RAG Agent]
            CURR_AGENT[Curriculum Agent]
        end
        
        subgraph "Core Logic Layer"
            RAG[RAG Module]
            SUMM[Summarization]
            COMMON[Common Utils]
        end
        
        subgraph "Data & Config Layer"
            CONFIG[Configuration]
            DATA_CONFIG[Data Config]
            LLM_CONFIG[LLM Config]
        end
        
        subgraph "Processing Layer"
            VIDEO_IE[Video IE]
            DATA_CONV[Data Conversion]
            DATA_LOAD[Data Loading]
            LANG_MGR[Language Manager]
        end
    end
    
    %% Dependencies
    FE --> AGENTS
    FE --> RAG
    FE --> COMMON
    AGENTS --> CONFIG
    AGENTS --> COMMON
    RAG --> COMMON
    RAG --> CONFIG
    SUMM --> RAG
    SUMM --> COMMON
    SUMM --> CONFIG
    COMMON --> VIDEO_IE
    
    %% Styling
    classDef frontend fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef agents fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef core fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef config fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef processing fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class FE,COMP,PAGES,UTILS frontend
    class AGENTS,CHAT_AGENT,RAG_AGENT,CURR_AGENT agents
    class RAG,SUMM,COMMON core
    class CONFIG,DATA_CONFIG,LLM_CONFIG config
    class VIDEO_IE,DATA_CONV,DATA_LOAD,LANG_MGR processing
```

### RAG Module Architecture

This diagram details the classes within the `RAG` module, illustrating their responsibilities and interconnections.

```mermaid
classDiagram
    class AvailableDataloader {
        -Path _docs_path
        +load() List~Document~
        +lazy_load() Iterator~Document~
    }
    
    class AvailableMetaData {
        <<enumeration>>
        ID : str
        PAGE_ID : str
        DOCUMENT_TYPE : str
        FILE_TYPE : str
        TEXT : str
        LEVEL : str
        TRUST_SCORE : str
        PART_OF_TRAINING : str
        PART_OF_DOCUMENT : str
        IS_META_SUMMARY : str
        STORAGE_PATH : str
        IMAGE_PATH : str
        AUDIO_PATH : str
        VIDEO_PATH : str
        LINK_TO_PARENT : str
        LINK_TO_CHILDREN : str
        LINK_TO_NEXT : str
        LINK_TO_PREVIOUS : str
        LINK_TO_RELATED : str
        START_TIME : str
        END_TIME : str
    }
    
    class AvailableRetriever {
        +str id_key
        +FAISS vectorstore
        +bool format_outputs
        +add_documents(documents: List~Document~) void
        +_get_relevant_documents(query: str, k: int) List~Document~
        +format_training_with_paragraphs(docs: List~Document~) str
        +save_local(folder_path: str) void
        +load_local(folder_path: str) AvailableRetriever
    }
    
    class QAFunctions {
        <<service>>
        +search_question(question: str) List~Document~
        +ask_question(question: str, language: str) str
        +init_global_vars() void
        +show_qa() void
    }
    
    class RetrievalHelpers {
        <<utility>>
        +get_embeddings() Embeddings
        +get_retriever(refresh: bool) AvailableRetriever
        +prepare_path(path: str) Path
        +get_image_cache_name(image_path: str) str
        +prepare_image_cache(image_path: str) str
        +refresh_retriever() AvailableRetriever
    }
    
    %% Relationships
    AvailableDataloader --|> BaseLoader : extends
    AvailableRetriever --|> ParentDocumentRetriever : extends
    QAFunctions ..> AvailableMetaData : uses
    QAFunctions ..> RetrievalHelpers : uses
    QAFunctions ..> AvailableRetriever : uses
    RetrievalHelpers ..> AvailableRetriever : creates
    RetrievalHelpers ..> AvailableDataloader : uses
    RetrievalHelpers ..> AvailableMetaData : uses
    AvailableRetriever ..> AvailableMetaData : uses
```

### Summarization Module Architecture

This diagram shows the comprehensive summarization pipeline with different content processors.

```mermaid
classDiagram
    class ImageProcessor {
        +_create_prompt(image_path: str) str
        +_create_prompt_from_file(file_path: str) str
        +generate_image_summary(image_path: str) str
        +generate_image_summary_for_retrieval(image_path: str) Document
        +generate_sequential_image_summary_for_retrieval(images: List) List~Document~
    }
    
    class TextProcessor {
        +generate_text_summary(text: str, context: str) str
        +generate_meta_summary(documents: List~Document~) str
        +generate_meta_summary_as_document(documents: List~Document~) Document
        +generate_sequential_documents_summary(docs: List~Document~) str
        +generate_sequential_document_summary_for_retrieval(docs: List) List~Document~
        +load_pdf_pages_for_available(file_path: str) List~Document~
        +generate_pdf_summaries_for_retrieval(file_path: str) List~Document~
        +generate_msg_summary_for_retrieval(file_path: str) List~Document~
    }
    
    class VideoProcessor {
        -_load_video_parts(video_path: str) List~Document~
        -_split_and_summarize_video_parts(parts: List~Document~) List~Document~
        +generate_video_summaries_for_retrieval(video_path: str) List~Document~
    }
    
    class SlideProcessor {
        +extract_comments_from_pptx(file_path: str) List~str~
        +generate_powerpoint_summary_for_retrieval(file_path: str) List~Document~
        +generate_pdf_or_pptx_summary_for_retrieval(file_path: str) List~Document~
    }
    
    class TrainingProcessor {
        +summarize_single_file(file_path: str) List~Document~
        +check_for_new_files(directory: str) bool
        +summarize_training_for_retrieval(training_path: str) List~Document~
    }
    
    class TrustScorer {
        <<enumeration>>
        +AvailableTrustScore
        +check_for_trust_file(file_path: str) bool
        +get_trust_score(document: Document) float
        +get_meta_trust_score(documents: List~Document~) float
    }
    
    class MarkdownUtils {
        +GeneralSummarizer
        +TrainingSummarizer
        +general_parser(text: str) dict
        +training_parser(text: str) dict
        +escape_urls(text: str) str
        +to_markdown(document: Document) str
        +doc_to_markdown(documents: List~Document~) str
    }
    
    class SummaryUtils {
        +connect_parent_and_children_documents(docs: List~Document~) void
        +connect_related_documents(docs: List~Document~) void
        +Summarizer
        +parser(summary_text: str) dict
    }
    
    %% Relationships
    ImageProcessor ..> TextProcessor : collaborates
    VideoProcessor ..> TextProcessor : uses
    SlideProcessor ..> ImageProcessor : uses
    SlideProcessor ..> TextProcessor : uses
    TrainingProcessor ..> ImageProcessor : uses
    TrainingProcessor ..> TextProcessor : uses
    TrainingProcessor ..> VideoProcessor : uses
    TrainingProcessor ..> SlideProcessor : uses
    TrainingProcessor ..> TrustScorer : uses
    TrustScorer ..> TextProcessor : evaluates
    MarkdownUtils ..> SummaryUtils : uses
    SummaryUtils ..> TrainingProcessor : supports
```

### Frontend Architecture Flow

This diagram shows the user interaction flow and component relationships in the frontend.

```mermaid
flowchart TD
    USER[👤 User] --> STREAMLIT[Streamlit App]
    
    subgraph "Frontend Components"
        STREAMLIT --> ASSISTANT[Assistant.py]
        ASSISTANT --> PAGES{Page Router}
        
        PAGES --> QA_PAGE[🤖 QA Chat Page]
        PAGES --> HANDBOOK[📚 Handbook Page]
        PAGES --> UPLOAD[📤 Upload Documents]
        PAGES --> REVIEW[🔍 Review Documents]
        PAGES --> ABOUT[ℹ️ About Page]
        
        subgraph "Reusable Components"
            BACKGROUND[🎨 Background Image]
            FOOTER[📄 Footer]
            FILE_HANDLER[📁 File Handler]
            TEAMS_SENDER[💬 Teams Sender]
        end
        
        QA_PAGE --> BACKGROUND
        QA_PAGE --> FOOTER
        UPLOAD --> FILE_HANDLER
        REVIEW --> FILE_HANDLER
    end
    
    subgraph "Backend Integration"
        QA_PAGE --> CHAT_AGENT[Chat Agent]
        QA_PAGE --> RAG_SYSTEM[RAG System]
        UPLOAD --> SUMMARIZATION[Summarization]
        REVIEW --> TRUST_SCORING[Trust Scoring]
        
        CHAT_AGENT --> LANG_DETECT[Language Detection]
        RAG_SYSTEM --> RETRIEVAL[Document Retrieval]
        SUMMARIZATION --> MULTI_MODAL[Multi-modal Processing]
    end
    
    %% Styling
    classDef userStyle fill:#ff9999,stroke:#cc0000,stroke-width:2px
    classDef frontendStyle fill:#99ccff,stroke:#0066cc,stroke-width:2px
    classDef backendStyle fill:#99ff99,stroke:#009900,stroke-width:2px
    classDef componentStyle fill:#ffcc99,stroke:#ff6600,stroke-width:2px
    
    class USER userStyle
    class STREAMLIT,ASSISTANT,PAGES,QA_PAGE,HANDBOOK,UPLOAD,REVIEW,ABOUT frontendStyle
    class BACKGROUND,FOOTER,FILE_HANDLER,TEAMS_SENDER componentStyle
    class CHAT_AGENT,RAG_SYSTEM,SUMMARIZATION,TRUST_SCORING,LANG_DETECT,RETRIEVAL,MULTI_MODAL backendStyle
```

### Data Flow Architecture

This diagram illustrates how data flows through the system from input to output.

```mermaid
flowchart LR
    subgraph "Input Sources"
        PDF[📄 PDF Files]
        PPT[📊 PowerPoint]
        VIDEO[🎥 Video Files]
        EMAIL[📧 Email Messages]
        TEXT[📝 Text Documents]
    end
    
    subgraph "Processing Pipeline"
        INGESTION[Data Ingestion]
        CONVERSION[Format Conversion]
        
        subgraph "Content Extraction"
            TEXT_EXT[Text Extraction]
            IMAGE_EXT[Image Extraction]
            AUDIO_EXT[Audio Extraction]
            META_EXT[Metadata Extraction]
        end
        
        subgraph "AI Processing"
            SUMMARIZATION[Summarization]
            EMBEDDING[Embedding Generation]
            TRUST_EVAL[Trust Evaluation]
            LANG_DETECT[Language Detection]
        end
        
        VECTOR_STORE[(Vector Database)]
        DOCUMENT_STORE[(Document Store)]
    end
    
    subgraph "Query Processing"
        USER_QUERY[User Query]
        QUERY_ANALYSIS[Query Analysis]
        RETRIEVAL[Document Retrieval]
        CONTEXT_BUILD[Context Building]
        LLM_RESPONSE[LLM Response Generation]
        RESPONSE[Final Response]
    end
    
    %% Data Flow
    PDF --> INGESTION
    PPT --> INGESTION
    VIDEO --> INGESTION
    EMAIL --> INGESTION
    TEXT --> INGESTION
    
    INGESTION --> CONVERSION
    CONVERSION --> TEXT_EXT
    CONVERSION --> IMAGE_EXT
    CONVERSION --> AUDIO_EXT
    CONVERSION --> META_EXT
    
    TEXT_EXT --> SUMMARIZATION
    IMAGE_EXT --> SUMMARIZATION
    AUDIO_EXT --> SUMMARIZATION
    META_EXT --> TRUST_EVAL
    
    SUMMARIZATION --> EMBEDDING
    EMBEDDING --> VECTOR_STORE
    SUMMARIZATION --> DOCUMENT_STORE
    TRUST_EVAL --> DOCUMENT_STORE
    
    USER_QUERY --> QUERY_ANALYSIS
    QUERY_ANALYSIS --> LANG_DETECT
    LANG_DETECT --> RETRIEVAL
    RETRIEVAL --> VECTOR_STORE
    RETRIEVAL --> DOCUMENT_STORE
    RETRIEVAL --> CONTEXT_BUILD
    CONTEXT_BUILD --> LLM_RESPONSE
    LLM_RESPONSE --> RESPONSE
    
    %% Styling
    classDef inputStyle fill:#ffe6e6,stroke:#cc0000,stroke-width:2px
    classDef processStyle fill:#e6f3ff,stroke:#0066cc,stroke-width:2px
    classDef storageStyle fill:#e6ffe6,stroke:#009900,stroke-width:2px
    classDef queryStyle fill:#fff0e6,stroke:#ff6600,stroke-width:2px
    
    class PDF,PPT,VIDEO,EMAIL,TEXT inputStyle
    class INGESTION,CONVERSION,TEXT_EXT,IMAGE_EXT,AUDIO_EXT,META_EXT,SUMMARIZATION,EMBEDDING,TRUST_EVAL,LANG_DETECT processStyle
    class VECTOR_STORE,DOCUMENT_STORE storageStyle
    class USER_QUERY,QUERY_ANALYSIS,RETRIEVAL,CONTEXT_BUILD,LLM_RESPONSE,RESPONSE queryStyle
```

## Setup and Installation

To set up this project, you will typically need to:

1.  **Clone the repository:** `git clone <repository_url>`
2.  **Install dependencies:** This project uses Python. Install dependencies using `pip` or `poetry`:
    ```bash
    pip install -r requirements.txt
    # or
    poetry install
    ```
3.  **Configure environment variables:** Set up necessary environment variables for:
    - API keys (OpenAI/Azure OpenAI)
    - Data paths and storage locations
    - Configuration settings as specified in `config/llm_config.py` and `config/data_config.py`
4.  **Prepare data directories:** Ensure your data structure follows the expected format:
    ```
    resources/
    ├── available_handbook/
    ├── processed_data/
    └── uploads/
    ```
5.  **Run initial setup:** Execute any data processing or indexing scripts to prepare the vector database.

## Usage

### Starting the Application

1.  **Launch the Streamlit app:**
    ```bash
    streamlit run dc_available/frontend/Assistant.py
    ```

2.  **Access the web interface:** Open your browser to `http://localhost:8501`

### Key Features Usage

- **💬 Chat Interface:** Ask questions in natural language and receive context-aware responses
- **📤 Document Upload:** Upload PDFs, PowerPoint files, videos, or emails for processing
- **🔍 Document Review:** Review processed documents and their trust scores
- **📚 Handbook Access:** Browse the integrated knowledge base
- **🌐 Multi-language Support:** The system automatically detects and responds in your preferred language

### Supported File Types

| File Type | Extension | Processing Features |
|-----------|-----------|-------------------|
| PDF Documents | `.pdf` | Text extraction, image analysis, page-by-page processing |
| PowerPoint | `.pptx`, `.ppt` | Slide content, speaker notes, embedded images |
| Videos | `.mp4` | Speech-to-text, image frame analysis, temporal segmentation |
| Emails | `.msg` | Content extraction, attachment processing, metadata analysis |
| Text Files | `.txt`, `.md` | Direct text processing, formatting preservation |

This enhanced README provides a comprehensive overview of the project with improved UML documentation that better visualizes the system architecture, component relationships, and data flow patterns.