# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_config import logger

from pydantic import BaseModel, Field
from typing import Optional, Literal, TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END 

from dotenv import load_dotenv

load_dotenv()

# %%
llm = ChatGroq(model="llama-3.1-8b-instant")

# %%
class PrimaryFields(BaseModel):
    sender_name: Optional[str] = Field(None, description="Name of the person or entity who sent the email")
    topic: Optional[str] = Field(None, description="Main topic or subject discussed in the email")
    action_required: Optional[bool] = Field(None, description="Whether the recipient is expected to take any action")
    amount: Optional[str] = Field(None, description="Amount mentioned if related to a bill, receipt or payment")
    due_date: Optional[str] = Field(None, description="Deadline for any required action (ISO format)")
    location: Optional[str] = Field(None, description="Relevant location mentioned in the email, if any")

class PromotionsFields(BaseModel):
    product: Optional[str] = Field(None, description="Name or type of product being promoted")
    discount: Optional[str] = Field(None, description="Discount amount or percentage mentioned")
    valid_until: Optional[str] = Field(None, description="Expiration date of the promotion (ISO format)")
    vendor: Optional[str] = Field(None, description="Vendor or brand offering the promotion")
    promo_code: Optional[str] = Field(None, description="Promotional code provided, if any")
    urgency: Optional[str] = Field(None, description="Time-sensitive language such as 'limited time' or 'today only'")

class SocialFields(BaseModel):
    platform: Optional[str] = Field(None, description="Social media platform (e.g., Facebook, Twitter)")
    notification_type: Optional[str] = Field(None, description="Type of notification (e.g., friend request, comment)")
    from_user: Optional[str] = Field(None, description="User who triggered the notification")
    action_summary: Optional[str] = Field(None, description="Brief description of the social interaction")

class UpdatesFields(BaseModel):
    entity: Optional[str] = Field(None, description="Entity involved in the update (e.g., bank, utility)")
    amount: Optional[str] = Field(None, description="Amount billed or referenced in the update")
    due_date: Optional[str] = Field(None, description="Due date for the payment or action (ISO format)")
    statement_type: Optional[str] = Field(None, description="Type of statement or update (e.g., credit card, utility bill)")
    account_ref: Optional[str] = Field(None, description="Account number or reference")
    status: Optional[str] = Field(None, description="Current status or outcome (e.g., paid, overdue, closed)")

class ReplyDraft(BaseModel):
    to: Optional[str] = Field(None, description="Recipient email address")
    subject: Optional[str] = Field(None, description="Subject of the reply email")
    body: Optional[str] = Field(None, description="Main content of the reply email")


class Email(BaseModel):
    sender: str = Field(..., description="Email address of the sender")
    date: Optional[str] = Field(None, description="Date the email was sent")
    subject: Optional[str] = Field(None, description="Subject line of the email")
    body: Optional[str] = Field(None, description="Full text or HTML content of the email body")



# %%
# Graph state definition
class EmailGraphState(TypedDict, total=False):
    email: Email
    intent: Optional[Literal["primary", "promotions", "social", "updates"]] # Optional because when we first build state, we will have only email variable and all other fields will be populated later
    intent_reason: Optional[str]
    extracted_fields: Optional[BaseModel]
    summary: Optional[str]
    is_spam: Optional[bool]
    spam_reason: Optional[str]
    reply_needed: Optional[bool]
    reply_draft: Optional[ReplyDraft]
    event_present: Optional[bool]
    calendar_event: Optional[dict]
    messages: Optional[List[BaseMessage]]

    model_config = {
        "frozen": False  # Allows mutation
    }



# %% [markdown]
# # Intent Classification Node

# %%
class IntentOutput(BaseModel):
    intent: Literal["primary", "promotions", "social", "updates"] = Field(..., description="Email intent category: primary, promotions, social, or updates")
    reason: str = Field(..., description="Justification or explanation for the selected intent")


def classify_intent_node(state: EmailGraphState) -> EmailGraphState:
    email = state["email"]
    logger.info("state: ")
    logger.info(state)
    human_message = HumanMessage(content=(
        f"From: {email.sender}\n"
        f"Date: {email.date or ''}\n"
        f"Subject: {email.subject or ''}\n"
        f"Body:\n{email.body or ''}\n"
    ))

    system_message = SystemMessage(content=(
        """
        You are an email classification assistant. Your task is to classify incoming emails into one of the following categories:

        1. **Primary** - Personal conversations, bills, receipts, and important updates from services used.
        2. **Promotions** - Marketing emails, discount offers, sales, newsletters, and product recommendations.
        3. **Social** - Notifications from social networks like LinkedIn, Facebook, Twitter (X), Instagram, etc.
        4. **Updates** - Service-related updates, confirmations, statements, invoices, or other automated system messages.

        Return a structured JSON with the intent and a short reason.

        **Important Note** - Use only the content provided to you. Do not hallucinate values.
        """ 
    ))
    response = llm.with_structured_output(IntentOutput).invoke([system_message, human_message])
    logger.info(f"intent classification response: {response}")
    return EmailGraphState(
        **state,
        intent=response.intent,
        intent_reason=response.reason
    )



# %%
# # Directly invoke node
# intent_classification_state = classify_intent_node(initial_state)
# intent_classification_state

# %% [markdown]
# # Field Extractor Node

# %%
def extract_fields_node(state: EmailGraphState) -> EmailGraphState:
    logger.info("state: ")
    logger.info(state)
    email = state["email"]
    intent = state["intent"]

    schema_map = {
        "primary": (PrimaryFields, "Extract fields such as topic, amount, action required, due date, etc."),
        "promotions": (PromotionsFields, "Extract product name, discount, vendor, promo code, urgency, etc."),
        "social": (SocialFields, "Extract platform, notification type, user who interacted, and a summary."),
        "updates": (UpdatesFields, "Extract amount, due date, status, entity, account reference, etc."),
    }

    schema_class, instruction = schema_map[intent]

    system_msg = SystemMessage(content=f"""
    You are an email field extractor. Extract structured fields based on this email's content.
    {instruction}
    Always follow the structure exactly and fill as many fields as possible.
    """.strip())

    human_msg = HumanMessage(content=(
        f"From: {email.sender}\n"
        f"Date: {email.date or ''}\n"
        f"Subject: {email.subject or ''}\n"
        f"Body:\n{email.body or ''}"
    ))

    extracted = llm.with_structured_output(schema_class).invoke([system_msg, human_msg])
    logger.info(f"fields extractor node response: {extracted}")
    return EmailGraphState(
        **state,
        extracted_fields=extracted
    )

# %%
# fields_extractor_state = extract_fields_node(intent_classification_state)
# fields_extractor_state

# %% [markdown]
# # Summarizer Node

# %%
def summarizer_node(state: EmailGraphState) -> EmailGraphState:
    logger.info("state: ")
    logger.info(state)
    email = state["email"]
    intent = state["intent"]

    intent_summary_instructions = {
        "primary": "Summarize the key information and whether any action is required.",
        "promotions": "Summarize the offer, discount, and urgency if any.",
        "social": "Summarize the type of social interaction and who triggered it.",
        "updates": "Summarize the update type, amount, status, and any due date if applicable.",
    }

    instruction = intent_summary_instructions[intent]

    system_msg = SystemMessage(content=f"""
    You are an email summarization assistant. Your task is to summarize the email content in 1-2 lines.
    {instruction}
    The summary should be clear and helpful for a user skimming through their inbox.
    """)

    human_msg = HumanMessage(content=(
        f"From: {email.sender}\n"
        f"Date: {email.date or ''}\n"
        f"Subject: {email.subject or ''}\n"
        f"Body:\n{email.body or ''}"
    ))

    summary = llm.invoke([system_msg, human_msg]).content.strip()
    logger.info(f"summary: {summary}")
    return EmailGraphState(
        **state,
        summary=summary
    )


# %%
# summary_state = summarizer_node(fields_extractor_state)
# summary_state

# %% [markdown]
# # Spam Detection Node

# %%
class SpamOutput(BaseModel):
    is_spam: bool = Field(..., description="True if the email is considered spam")
    reason: str = Field(..., description="Brief explanation of why the email is or isn't spam")


def spam_detection_node(state: EmailGraphState) -> EmailGraphState:
    logger.info("state: ")
    logger.info(state)
    email = state["email"]

    system_msg = SystemMessage(content="""
    You are a spam detection assistant. Classify the email as spam or not spam.
    Use indicators such as: misleading offers, excessive promotions, phishing language, unknown senders, etc.
    Your response must include:
    - is_spam: true or false
    - reason: a brief explanation for your decision
    """)

    human_msg = HumanMessage(content=(
        f"From: {email.sender}\n"
        f"Date: {email.date or ''}\n"
        f"Subject: {email.subject or ''}\n"
        f"Body:\n{email.body or ''}"
    ))


    response = llm.with_structured_output(SpamOutput).invoke([system_msg, human_msg])
    logger.info(f"spam detection node response: {response}")
    return EmailGraphState(
        **state,
        is_spam=response.is_spam,
        spam_reason=response.reason
    )


# %%
# spam_detection_state = spam_detection_node(summary_state)
# spam_detection_state

# %% [markdown]
# # Reply Generator Node

# %%
class ReplyOutput(BaseModel):
    reply_needed: bool = Field(..., description="Whether a reply is needed to this email")
    reason: Optional[str] = Field(None, description="Justification for reply decision")
    draft: Optional[ReplyDraft] = Field(None, description="Structured draft to be used if a reply is needed")


def generate_reply_node(state: EmailGraphState) -> EmailGraphState:
    logger.info("state: ")
    logger.info(state)
    email = state["email"]

    messages = state.get("messages", [])
    logger.info(f"messages: {messages}")
    feedback = None
    old_draft = state.get("reply_draft")

    # Check for latest human feedback (if any)
    if messages and messages[-1].type == "human":
        content = messages[-1].content.strip()
        logger.info(f"feedback message from user: {content}")
        if content.lower() != "approved":
            feedback = content
    
    if feedback:
        system_msg = SystemMessage(content="""
        You are an email assistant.

        You will be provided with:
        - A message from a sender to the user.
        - A draft reply (if available) generated by an AI assistant.
        - Feedback from the user.

        Your task is to revise the existing draft using the feedback if it exists, or create a new draft if none is available. The final reply must contain:
        - reply_needed: true if a reply is needed, false if not.
        - reason: Justification for reply decision
        - 'to': recipient email
        - 'subject': subject line
        - 'body': the message body
        """)
        human_msg = HumanMessage(content=(
            "Original Email (This is the email sent by sender to the user):\n"
            f"From: {email.sender}\n"
            f"Date: {email.date or ''}\n"
            f"Subject: {email.subject or ''}\n"
            f"Body:\n{email.body or ''}\n"
            f"User feedback: {feedback}"
        ))
        if old_draft:
            draft_text = (
                f"To: {old_draft.to or ''}\n"
                f"Subject: {old_draft.subject or ''}\n"
                f"Body:\n{old_draft.body or ''}"
            )
        else:
            draft_text = "No previous draft available."
        ai_msg = AIMessage(content=(
            f"Current Draft (This is the draft done by AI Agent):\n{draft_text} "
        ))
        # Ask LLM for structured output
        response = llm.with_structured_output(ReplyOutput).invoke([system_msg, human_msg, ai_msg])
        logger.info(f"reply generator response: {response}")
        state.pop("is_reply_needed", None)
        state.pop("reply_reason", None)
        state.pop("reply_draft", None)

        return EmailGraphState(
            **state,
            is_reply_needed=response.reply_needed,
            reply_reason=response.reason,
            reply_draft=response.draft if response.reply_needed else None
        )
    else:
        system_msg = SystemMessage(content="""
        You are an email assistant. Determine if the email requires a reply.
        If a reply is needed, generate a response using the ReplyDraft format
        with 'to', 'subject', and 'body'. Otherwise, just say that no reply is needed.
        """)

        human_msg = HumanMessage(content=(
            f"From: {email.sender}\n"
            f"Date: {email.date or ''}\n"
            f"Subject: {email.subject or ''}\n"
            f"Body:\n{email.body or ''}"
        ))

        # Ask LLM for structured output
        response = llm.with_structured_output(ReplyOutput).invoke([system_msg, human_msg])
        logger.info(f"reply generator response: {response}")

        return EmailGraphState(
            **state,
            is_reply_needed=response.reply_needed,
            reply_reason=response.reason,
            reply_draft=response.draft if response.reply_needed else None
        )



# %%
# generate_reply_state = generate_reply_node(spam_detection_state)
# generate_reply_state

# %% [markdown]
# # Extract Event Node

# %%
class EventDetails(BaseModel):
    title: Optional[str] = Field(None, description="Title of the event or reminder")
    start_time: Optional[str] = Field(None, description="Start date and time of the event in ISO format")
    end_time: Optional[str] = Field(None, description="End date and time of the event in ISO format")
    location: Optional[str] = Field(None, description="Event location, if available")

class EventOutput(BaseModel):
    is_event: bool = Field(..., description="Whether the email is related to an event or reminder")
    reason: Optional[str] = Field(None, description="Justification for the decision")
    details: Optional[EventDetails] = Field(None, description="Details about the event if applicable")


def extract_event_node(state: EmailGraphState) -> EmailGraphState:
    logger.info("state: ")
    logger.info(state)
    email = state["email"]
    
    human_message = HumanMessage(content=(
        f"From: {email.sender}\n"
        f"Date: {email.date}\n"
        f"Subject: {email.subject}\n"
        f"Body:\n{email.body}\n"
    ))

    system_message = SystemMessage(content=(
        "You are an assistant that determines if an email is about an event, appointment, or deadline. "
        "If yes, extract event title, date/time, and location if available."
    ))

    response = llm.with_structured_output(EventOutput).invoke([system_message, human_message])
    logger.info(f"event response: {response}")
    return EmailGraphState(
        **state,
        event_present=response.is_event,
        calendar_event=response.details.model_dump() if response.details else None
    )

# %%
# extract_event_state = extract_event_node(generate_reply_state)
# extract_event_state

# %% [markdown]
# # Human Review Node

# %%
def human_review_node(state: EmailGraphState) -> EmailGraphState:
    logger.info("state: ")
    logger.info(state)
    
    logger.info(f"Reply Draft: {state["reply_draft"]}")

    feedback = input("Enter feedback on the draft (or 'approved'): ").strip()
    try:
        state["messages"].append(HumanMessage(content=feedback))
    except Exception as e:
        state["messages"] = [HumanMessage(content=feedback)]
    logger.info(f"state: {state}")
    return state


# %% [markdown]
# # Handle Feedback Condition 

# %%
def handle_feedback_condition(state: EmailGraphState) -> str:
    logger.info("state: ")
    logger.info(state)
    last_msg = state["messages"][-1].content.strip().lower()
    return "extract_event" if last_msg == "approved" else "generate_reply"


# %% [markdown]
# # Define Graph

# %%
# === Define the graph ===
graph = StateGraph(EmailGraphState)

graph.add_node("classify_intent", classify_intent_node)
graph.add_node("extract_fields", extract_fields_node)
graph.add_node("summarize", summarizer_node)
graph.add_node("detect_spam", spam_detection_node)
graph.add_node("generate_reply", generate_reply_node)
graph.add_node("extract_event", extract_event_node)
graph.add_node("human_review", human_review_node)

graph.set_entry_point("classify_intent")

graph.add_edge("classify_intent", "extract_fields")
graph.add_edge("extract_fields", "summarize")
graph.add_edge("summarize", "detect_spam")
graph.add_edge("detect_spam", "generate_reply")
graph.add_edge("generate_reply", "human_review")


graph.add_conditional_edges(
    "human_review",
    handle_feedback_condition,
    {
        "extract_event": "extract_event",
        "generate_reply": "generate_reply",
    },
)

graph.add_edge("extract_event", END)

# Compile the app
app = graph.compile()

# %% [markdown]
# # Invoke and get response

# %%
sample_email = Email(
    sender="jobs@example.com",
    subject="Interview Invitation",
    body="We are pleased to invite you...",
    date="2025-05-20"
)

initial_state = {"email": sample_email}
# %%
response = app.invoke(initial_state)
logger.info(response)

# %%
from pprint import pprint
pprint(response)

# %%



