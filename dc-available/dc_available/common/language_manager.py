# Add this to your requirements.txt or install via pip
# pip install langdetect

from langdetect import detect, DetectorFactory
from typing import Optional
import streamlit as st

DetectorFactory.seed = 0


class LanguageManager:
    """Simple language detection and management for chatbot responses."""

    def __init__(self):
        self.current_language = None
        self.supported_languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish',
            'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish', 'cs': 'Czech',
            'hu': 'Hungarian', 'ro': 'Romanian', 'bg': 'Bulgarian', 'hr': 'Croatian',
            'sk': 'Slovak', 'sl': 'Slovenian', 'et': 'Estonian', 'lv': 'Latvian',
            'lt': 'Lithuanian', 'uk': 'Ukrainian', 'be': 'Belarusian', 'mk': 'Macedonian',
            'sq': 'Albanian', 'sr': 'Serbian', 'bs': 'Bosnian', 'mt': 'Maltese',
            'ga': 'Irish', 'cy': 'Welsh', 'eu': 'Basque', 'ca': 'Catalan',
            'gl': 'Galician', 'is': 'Icelandic', 'fo': 'Faroese', 'kl': 'Greenlandic'
        }

    def detect_language(self, text: str) -> Optional[str]:
        """Detect language of input text."""
        try:
            if len(text.strip()) < 10:
                return None
            detected_lang = detect(text)
            if detected_lang in self.supported_languages:
                return detected_lang
            return None
        except Exception as e:
            print(f"Language detection error: {e}")
            return None

    def update_language(self, user_input: str) -> str:
        """Update current language based on user input."""
        detected_lang = self.detect_language(user_input)
        if detected_lang:
            if self.current_language != detected_lang:
                print(f"Language changed from {self.current_language} to {detected_lang}")
                self.current_language = detected_lang
        return self.current_language or 'en'

    @staticmethod
    def get_language_instruction() -> str:
        """Get hidden instruction - not shown to user."""
        return ""  # Return empty - we'll handle language differently

    def get_conversation_context(self) -> str:
        """Get language context for system message only."""
        if not self.current_language or self.current_language == 'en':
            return ""

        lang_name = self.supported_languages.get(self.current_language, self.current_language)
        return f"\n\nConversation Language: {lang_name}. All participants should communicate in {lang_name}."

    def get_smart_qa_prompt(self, original_question: str) -> str:
        """
        Create intelligent prompt that handles translation within the LLM.
        This leverages the LLM's multilingual capabilities.
        """
        if not self.current_language or self.current_language == 'en':
            return original_question

        lang_name = self.supported_languages.get(self.current_language, self.current_language)

        # Smart prompt that instructs LLM to:
        # 1. Understand the question in any language
        # 2. Search using English keywords
        # 3. Respond in the original language
        smart_prompt = f"The user asked: {original_question}"

def get_language_manager():
    """Get or create language manager from session state."""
    if 'language_manager' not in st.session_state:
        st.session_state.language_manager = LanguageManager()
    return st.session_state.language_manager


def get_localized_system_message(base_message: str, language_manager: LanguageManager) -> str:
    """Add language instruction to system message."""
    return base_message + language_manager.get_conversation_context()

# Example: Modified agent creation functions
def create_topic_assessor_with_language():
    """Modified version of your create_topic_assessor function."""
    from dc_available.config.llm_config import get_autogen_llm_config, DEFAULT_MODEL
    from autogen import ConversableAgent

    llm_config = get_autogen_llm_config(model_name=DEFAULT_MODEL)
    language_manager = get_language_manager()

    base_system_message = """You are a topic assessor.
    Your goal is to figure out, what a user wants to learn. This is the only thing you care about.
    You only gather information, you do not answer the user's questions.
    You ask the user at least one clarifying question regarding the topic the user wants to learn about.
    Once you understand what the user wants to learn, you describe it in a detailed summary. This summary begins with the word 'SUMMARY:'."""

    system_message = get_localized_system_message(base_system_message, language_manager)

    topic_assessor = ConversableAgent(
        name="topic_assessor",
        description="Assesses what the user wants to learn.",
        system_message=system_message,
        human_input_mode="NEVER",
        llm_config=llm_config,
        max_consecutive_auto_reply=4,
        code_execution_config=False,
    )
    return topic_assessor


def create_knowledge_assessor_with_language():
    """Modified version of your create_knowledge_assessor function."""
    from dc_available.config.llm_config import get_autogen_llm_config, DEFAULT_MODEL
    from autogen import ConversableAgent

    llm_config = get_autogen_llm_config(model_name=DEFAULT_MODEL)
    language_manager = get_language_manager()

    # Your existing knowledge level definitions
    BEGINNER_KNOWLEDGE_LEVEL = """Beginner
    Understanding: Has no prior knowledge or understanding of the topic;
    Experience: Has not had any formal or informal exposure to the topic."""

    ADVANCED_KNOWLEDGE_LEVEL = """Advanced
    Understanding: Has a basic understanding of key concepts and terminology related to the topic.
    Experience: May have completed introductory courses, hands-on training, or have informal experience."""

    EXPERT_KNOWLEDGE_LEVEL = """Expert
    Understanding: Possesses deep, comprehensive knowledge and understanding of the topic, including advanced concepts and theories.
    Experience: Has extensive practical experience, including professional work or advanced academic study."""

    base_system_message = f"""You are a knowledge assessor.

    The three levels of knowledge are:

    {BEGINNER_KNOWLEDGE_LEVEL}

    {ADVANCED_KNOWLEDGE_LEVEL}

    {EXPERT_KNOWLEDGE_LEVEL}

    You ask questions, until you are confident that you have enough information to assess the user's knowledge level to be Beginner, Advanced or Expert.
    You only ask questions, you don't answer questions.
    Once you are sure about the user's knowledge level, you report the knowledge level.
    Your response options is one of the following three:

    1) "SUMMARY: {BEGINNER_KNOWLEDGE_LEVEL}",

    2) "SUMMARY: {ADVANCED_KNOWLEDGE_LEVEL}",

    3) "SUMMARY: {EXPERT_KNOWLEDGE_LEVEL}"."""

    system_message = get_localized_system_message(base_system_message, language_manager)

    knowledge_assessor = ConversableAgent(
        name="knowledge_assessor",
        description="Assesses a user's knowledge level with respect to a given topic.",
        system_message=system_message,
        human_input_mode="NEVER",
        llm_config=llm_config,
        max_consecutive_auto_reply=8,
        code_execution_config=False,
    )
    return knowledge_assessor


# Modified controlled_chat function
def controlled_chat_with_language(
        user,
        chat_manager,
        initial_message: str,
        stream: bool = True,
        context: Optional[str] = None,
        user_proxy=None,
        max_iterations: int = 20,
):
    """Modified version of your controlled_chat function with language detection."""
    language_manager = get_language_manager()
    iterations = 0

    while iterations < max_iterations:
        if stream:
            prompt = st.chat_input()
            if prompt is None and len(chat_manager.chat_messages.values()) > 0:
                return

        # Update language based on user input
        if stream and prompt:
            current_lang = language_manager.update_language(prompt)
            print(f"Current conversation language: {current_lang}")

            # Update all agent system messages with language instruction
            for agent in chat_manager.groupchat.agents:
                if hasattr(agent, 'system_message') and agent.name != 'user':
                    # Store original system message if not already stored
                    if not hasattr(agent, '_original_system_message'):
                        agent._original_system_message = agent.system_message

                    # Update with language instruction
                    agent.system_message = get_localized_system_message(
                        agent._original_system_message,
                        language_manager
                    )

        # Rest of your existing controlled_chat logic...
        if stream:
            from autogen import Cache
            agent_cache = Cache.disk(cache_seed=st.session_state["conversation_id"])
        else:
            agent_cache = None

        if len(chat_manager.chat_messages.values()) == 0:
            context = "\nContext: " + context if type(context) == str else ""
            intro_message = {
                "role": "user",
                "name": "system",
                "content": initial_message + context,
            }
        else:
            intro_message = {
                "role": "user",
                "name": "user",
                "content": prompt,
            }

        chat_result = user.initiate_chat(
            chat_manager,
            message=intro_message,
            summary_method="last_msg",
            clear_history=False,
            cache=agent_cache,
            is_termination_msg=lambda x: False,
            silent=False,
        )

        if (
                "SUMMARY" in chat_manager.groupchat.messages[-1]["content"]
                or "TERMINATE" in chat_manager.groupchat.messages[-1]["content"]
        ):
            return chat_manager.groupchat.messages[-1]["content"]
        else:
            if stream:
                return
            # Handle non-stream mode...

        iterations += 1

    return chat_manager.groupchat.messages[-1]["content"]