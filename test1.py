def main():
    """Main function to demonstrate the customer support agent on Ubuntu"""
    if HAS_COLOR:
        print(f"{Fore.BLUE}{'='*50}")
        print(f"🤖 Customer Support AI Agent System")
        print(f"   Ubuntu Linux Optimized Version")
        print(f"{'='*50}{Style.RESET_ALL}\n")
    else:
        print("=== Customer Support AI Agent System ===")
        print("Ubuntu Linux Optimized Version\n")
    
    # System information
    #!/usr/bin/env python3

    import psutil
    from colorama import Fore, Style

    HAS_COLOR = True  # or detect if you want

    if psutil:
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        if HAS_COLOR:
            print(f"{Fore.CYAN}💻 System Info: {cpu_count} CPU cores, {memory_gb:.1f}GB RAM{Style.RESET_ALL}")
        else:
            print(f"System Info: {cpu_count} CPU cores, {memory_gb:.1f}GB RAM")

            
"""
Customer Support Ticket Classifier & Responder
An AI agent system using LangChain with local LLM integration

Optimized for Ubuntu Linux with enhanced error handling,
logging, and performance monitoring.

This system demonstrates:
- Ticket classification using LangChain chains
- Automated response generation
- Memory management per ticket
- Tool usage and decision making
- Integration with local LLM (Ollama/LlamaCpp)
- Ubuntu-specific optimizations
"""

import json
import logging
import os
import sys
import time
import signal
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Ubuntu-specific imports
try:
    import psutil  # For system monitoring
except ImportError:
    psutil = None

# Environment variable loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Terminal colors for Ubuntu
try:
    from colorama import init, Fore, Style
    init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    Fore = Style = type('MockColor', (), {'RED': '', 'GREEN': '', 'YELLOW': '', 'BLUE': '', 'RESET_ALL': ''})()

# LangChain imports with error handling
try:
    from langchain.llms import Ollama, LlamaCpp
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import LLMChain, ConversationChain
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.agents import Tool, initialize_agent, AgentType
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
except ImportError as e:
    print(f"Error importing LangChain: {e}")
    print("Please install with: pip install langchain")
    sys.exit(1)

# Set up logging with Ubuntu-friendly configuration
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
log_file = os.getenv('LOG_FILE', 'support_agent.log')

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration and Data Models
class TicketCategory(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT_ACCESS = "account_access"
    FEEDBACK = "feedback"
    UNKNOWN = "unknown"

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SupportTicket:
    ticket_id: str
    customer_email: str
    subject: str
    message: str
    category: Optional[TicketCategory] = None
    priority: Optional[TicketPriority] = None
    status: str = "open"
    created_at: str = None
    response: Optional[str] = None
    escalated: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

# LLM Configuration optimized for Ubuntu
class LLMConfig:
    """Configuration for local LLM setup on Ubuntu"""
    
    # Environment-based configuration
    USE_OLLAMA = os.getenv('USE_OLLAMA', 'true').lower() == 'true'
    
    # Ollama configuration
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral:7b')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    # LlamaCpp configuration
    LLAMACPP_MODEL_PATH = os.getenv('LLAMACPP_MODEL_PATH', './models/mistral-7b-instruct-v0.1.Q4_K_M.gguf')
    LLAMACPP_N_CTX = int(os.getenv('LLAMACPP_N_CTX', '2048'))
    LLAMACPP_TEMPERATURE = float(os.getenv('LLAMACPP_TEMPERATURE', '0.1'))
    
    # Performance settings
    MAX_MEMORY_TURNS = int(os.getenv('MAX_MEMORY_TURNS', '10'))
    RESPONSE_TIMEOUT = int(os.getenv('RESPONSE_TIMEOUT', '30'))
    
    @classmethod
    def validate_ubuntu_environment(cls):
        """Validate Ubuntu environment for LLM usage"""
        issues = []
        
        # Check memory
        if psutil:
            memory = psutil.virtual_memory()
            if memory.total < 8 * 1024**3:  # Less than 8GB
                issues.append(f"Low memory: {memory.total / 1024**3:.1f}GB (8GB+ recommended)")
        
        # Check Ollama availability
        if cls.USE_OLLAMA:
            try:
                import requests
                response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/version", timeout=5)
                if response.status_code != 200:
                    issues.append(f"Ollama not responding at {cls.OLLAMA_BASE_URL}")
            except Exception as e:
                issues.append(f"Cannot connect to Ollama: {e}")
        
        # Check LlamaCpp model file
        else:
            if not os.path.exists(cls.LLAMACPP_MODEL_PATH):
                issues.append(f"LlamaCpp model not found: {cls.LLAMACPP_MODEL_PATH}")
        
        return issues

class CustomerSupportAgent:
    """Main AI agent for customer support ticket processing"""
    
    def __init__(self, use_ollama: bool = True):
        """
        Initialize the support agent with local LLM
        
        Args:
            use_ollama: If True, use Ollama; if False, use LlamaCpp
        """
        self.use_ollama = use_ollama
        self.llm = self._setup_llm()
        self.memory_store: Dict[str, ConversationBufferWindowMemory] = {}
        self.ticket_store: Dict[str, SupportTicket] = {}
        
        # Initialize chains
        self.classification_chain = self._setup_classification_chain()
        self.response_chain = self._setup_response_chain()
        self.escalation_chain = self._setup_escalation_chain()
        
        # Initialize tools
        self.tools = self._setup_tools()
        
        logger.info(f"Customer Support Agent initialized with {'Ollama' if use_ollama else 'LlamaCpp'}")
    
    def _setup_llm(self):
        """Setup the local LLM based on configuration"""
        try:
            if self.use_ollama:
                llm = Ollama(
                    model=LLMConfig.OLLAMA_MODEL,
                    base_url=LLMConfig.OLLAMA_BASE_URL,
                    temperature=0.1,
                    num_ctx=2048
                )
                logger.info(f"Connected to Ollama model: {LLMConfig.OLLAMA_MODEL}")
            else:
                llm = LlamaCpp(
                    model_path=LLMConfig.LLAMACPP_MODEL_PATH,
                    n_ctx=LLMConfig.LLAMACPP_N_CTX,
                    temperature=LLMConfig.LLAMACPP_TEMPERATURE,
                    max_tokens=512,
                    verbose=False
                )
                logger.info(f"Loaded LlamaCpp model from: {LLMConfig.LLAMACPP_MODEL_PATH}")
            
            return llm
            
        except Exception as e:
            logger.error(f"Failed to setup LLM: {e}")
            raise
    
    def _setup_classification_chain(self) -> LLMChain:
        """Setup the ticket classification chain"""
        classification_prompt = PromptTemplate(
            input_variables=["subject", "message"],
            template="""
You are a customer support ticket classifier. Analyze the following support ticket and classify it into one of these categories:

Categories:
- billing: Issues related to payments, invoices, refunds, pricing
- technical: Technical problems, bugs, system issues, how-to questions
- account_access: Login problems, password resets, account lockouts
- feedback: General feedback, suggestions, complaints about service
- unknown: If the ticket doesn't fit clearly into other categories

Ticket Subject: {subject}
Ticket Message: {message}

Based on the content above, classify this ticket. Also determine the priority level:
- critical: System down, security issues, unable to access paid services
- high: Major functionality broken, billing disputes
- medium: Minor bugs, feature requests, general questions
- low: Feedback, suggestions, documentation requests

Respond in this exact format:
CATEGORY: [category]
PRIORITY: [priority]
REASONING: [brief explanation]
"""
        )
        
        return LLMChain(llm=self.llm, prompt=classification_prompt)
    
    def _setup_response_chain(self) -> LLMChain:
        """Setup the automated response generation chain"""
        response_prompt = PromptTemplate(
            input_variables=["category", "priority", "subject", "message", "customer_email"],
            template="""
You are a helpful customer support representative. Generate a professional and empathetic response to this support ticket.

Ticket Details:
- Category: {category}
- Priority: {priority}
- Customer Email: {customer_email}
- Subject: {subject}
- Message: {message}

Guidelines:
1. Be professional, empathetic, and helpful
2. Address the customer's specific concern
3. Provide clear next steps or solutions when possible
4. If you cannot resolve the issue, explain what will happen next
5. Keep the response concise but thorough
6. Use a friendly but professional tone

Generate a response:
"""
        )
        
        return LLMChain(llm=self.llm, prompt=response_prompt)
    
    def _setup_escalation_chain(self) -> LLMChain:
        """Setup the escalation decision chain"""
        escalation_prompt = PromptTemplate(
            input_variables=["category", "priority", "subject", "message"],
            template="""
You are a customer support supervisor. Determine if this ticket should be escalated to a human agent.

Ticket Details:
- Category: {category}
- Priority: {priority}
- Subject: {subject}
- Message: {message}

Escalation Criteria:
- ESCALATE if: Critical/High priority, complex technical issues, billing disputes, legal concerns, angry customers
- AUTO-RESPOND if: Low/Medium priority, simple questions, general feedback, standard requests

Respond with either "ESCALATE" or "AUTO_RESPOND" followed by a brief reason.

Decision:
"""
        )
        
        return LLMChain(llm=self.llm, prompt=escalation_prompt)
    
    def _setup_tools(self) -> List[Tool]:
        """Setup tools for the agent"""
        tools = [
            Tool(
                name="get_ticket_history",
                description="Get the conversation history for a specific ticket ID",
                func=self._get_ticket_history
            ),
            Tool(
                name="update_ticket_status",
                description="Update the status of a ticket (open, in_progress, resolved, escalated)",
                func=self._update_ticket_status
            ),
            Tool(
                name="search_knowledge_base",
                description="Search for solutions in the knowledge base",
                func=self._search_knowledge_base
            )
        ]
        return tools
    
    def _get_ticket_history(self, ticket_id: str) -> str:
        """Tool function to get ticket conversation history"""
        if ticket_id in self.memory_store:
            memory = self.memory_store[ticket_id]
            return f"Ticket {ticket_id} history: {memory.buffer}"
        return f"No history found for ticket {ticket_id}"
    
    def _update_ticket_status(self, ticket_id_status: str) -> str:
        """Tool function to update ticket status"""
        try:
            ticket_id, new_status = ticket_id_status.split(",")
            if ticket_id in self.ticket_store:
                self.ticket_store[ticket_id].status = new_status.strip()
                return f"Updated ticket {ticket_id} status to {new_status}"
            return f"Ticket {ticket_id} not found"
        except ValueError:
            return "Invalid format. Use: ticket_id,new_status"
    
    def _search_knowledge_base(self, query: str) -> str:
        """Tool function to search knowledge base (mock implementation)"""
        # In a real implementation, this would search a vector database or knowledge base
        knowledge_base = {
            "password reset": "To reset password, go to login page and click 'Forgot Password'",
            "billing": "For billing inquiries, check your account dashboard or contact billing@company.com",
            "technical": "For technical issues, try clearing cache and cookies first",
            "refund": "Refunds are processed within 5-7 business days after approval"
        }
        
        for key, value in knowledge_base.items():
            if key.lower() in query.lower():
                return f"Knowledge base result: {value}"
        
        return "No relevant information found in knowledge base"
    
    def get_or_create_memory(self, ticket_id: str) -> ConversationBufferWindowMemory:
        """Get or create conversation memory for a ticket"""
        if ticket_id not in self.memory_store:
            self.memory_store[ticket_id] = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 conversation turns
                return_messages=True
            )
        return self.memory_store[ticket_id]
    
    def classify_ticket(self, ticket: SupportTicket) -> Tuple[TicketCategory, TicketPriority]:
        """Classify a support ticket"""
        try:
            logger.info(f"Classifying ticket {ticket.ticket_id}")
            
            result = self.classification_chain.run(
                subject=ticket.subject,
                message=ticket.message
            )
            
            # Parse the classification result
            category = TicketCategory.UNKNOWN
            priority = TicketPriority.MEDIUM
            
            lines = result.strip().split('\n')
            for line in lines:
                if line.startswith('CATEGORY:'):
                    cat_str = line.split(':', 1)[1].strip().lower()
                    for cat in TicketCategory:
                        if cat.value == cat_str:
                            category = cat
                            break
                elif line.startswith('PRIORITY:'):
                    pri_str = line.split(':', 1)[1].strip().lower()
                    for pri in TicketPriority:
                        if pri.value == pri_str:
                            priority = pri
                            break
            
            logger.info(f"Classified ticket {ticket.ticket_id} as {category.value} with {priority.value} priority")
            return category, priority
            
        except Exception as e:
            logger.error(f"Error classifying ticket {ticket.ticket_id}: {e}")
            return TicketCategory.UNKNOWN, TicketPriority.MEDIUM
    
    def should_escalate(self, ticket: SupportTicket) -> bool:
        """Determine if ticket should be escalated to human"""
        try:
            result = self.escalation_chain.run(
                category=ticket.category.value if ticket.category else "unknown",
                priority=ticket.priority.value if ticket.priority else "medium",
                subject=ticket.subject,
                message=ticket.message
            )
            
            escalate = "ESCALATE" in result.upper()
            logger.info(f"Escalation decision for ticket {ticket.ticket_id}: {'ESCALATE' if escalate else 'AUTO_RESPOND'}")
            return escalate
            
        except Exception as e:
            logger.error(f"Error determining escalation for ticket {ticket.ticket_id}: {e}")
            return True  # Default to escalation on error
    
    def generate_response(self, ticket: SupportTicket) -> str:
        """Generate automated response for ticket"""
        try:
            logger.info(f"Generating response for ticket {ticket.ticket_id}")
            
            response = self.response_chain.run(
                category=ticket.category.value if ticket.category else "unknown",
                priority=ticket.priority.value if ticket.priority else "medium",
                subject=ticket.subject,
                message=ticket.message,
                customer_email=ticket.customer_email
            )
            
            # Update conversation memory
            memory = self.get_or_create_memory(ticket.ticket_id)
            memory.chat_memory.add_user_message(f"Subject: {ticket.subject}\nMessage: {ticket.message}")
            memory.chat_memory.add_ai_message(response)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response for ticket {ticket.ticket_id}: {e}")
            return "Thank you for contacting us. We're experiencing technical difficulties. A human agent will respond to your ticket shortly."
    
    def process_ticket(self, ticket: SupportTicket) -> Dict:
        """Main method to process a support ticket end-to-end with performance tracking"""
        start_time = time.time()
        
        if HAS_COLOR:
            print(f"\n{Fore.BLUE}🎫 Processing ticket {ticket.ticket_id}{Style.RESET_ALL}")
        
        logger.info(f"Processing ticket {ticket.ticket_id}")
        
        # Store ticket
        self.ticket_store[ticket.ticket_id] = ticket
        
        try:
            # Step 1: Classify ticket
            category, priority = self.classify_ticket(ticket)
            ticket.category = category
            ticket.priority = priority
            
            # Step 2: Determine if escalation is needed
            should_escalate = self.should_escalate(ticket)
            ticket.escalated = should_escalate
            
            # Step 3: Generate response or escalate
            if should_escalate:
                ticket.status = "escalated"
                response = f"Thank you for contacting us. Your {priority.value} priority {category.value} issue has been escalated to our specialized team. You will receive a response within 24 hours."
            else:
                response = self.generate_response(ticket)
                ticket.status = "responded"
            
            ticket.response = response
            
            # Update counters
            self.processed_tickets += 1
            processing_time = time.time() - start_time
            
            # Return processing result
            result = {
                "ticket_id": ticket.ticket_id,
                "category": category.value,
                "priority": priority.value,
                "escalated": should_escalate,
                "status": ticket.status,
                "response": response,
                "processing_time": processing_time,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            if HAS_COLOR:
                color = Fore.RED if should_escalate else Fore.GREEN
                print(f"{color}✅ Completed ticket {ticket.ticket_id} in {processing_time:.2f}s{Style.RESET_ALL}")
            
            logger.info(f"Completed processing ticket {ticket.ticket_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing ticket {ticket.ticket_id}: {e}")
            
            # Fallback response
            ticket.response = "We're experiencing technical difficulties. A human agent will respond to your ticket shortly."
            ticket.status = "error"
            ticket.escalated = True
            
            return {
                "ticket_id": ticket.ticket_id,
                "category": "unknown",
                "priority": "medium",
                "escalated": True,
                "status": "error",
                "response": ticket.response,
                "error": str(e),
                "processing_timestamp": datetime.now().isoformat()
            }

# Signal handlers for graceful shutdown on Ubuntu
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    if HAS_COLOR:
        print(f"\n{Fore.YELLOW}🛑 Shutting down gracefully...{Style.RESET_ALL}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Example usage and testing
def create_sample_tickets() -> List[SupportTicket]:
    """Create sample tickets for testing"""
    tickets = [
        SupportTicket(
            ticket_id="T001",
            customer_email="john.doe@email.com",
            subject="Cannot login to my account",
            message="I've been trying to log into my account for the past hour but keep getting 'invalid credentials' error. I'm sure my password is correct. Please help!"
        ),
        SupportTicket(
            ticket_id="T002",
            customer_email="jane.smith@email.com",
            subject="Billing issue - double charged",
            message="I was charged twice for my monthly subscription this month. I see two charges of $29.99 on my credit card statement. Please refund one of them immediately."
        ),
        SupportTicket(
            ticket_id="T003",
            customer_email="tech.user@email.com",
            subject="API returning 500 errors",
            message="Our production application is experiencing 500 internal server errors when calling your API endpoints. This started about 2 hours ago and is affecting our customers. Need urgent help!"
        ),
        SupportTicket(
            ticket_id="T004",
            customer_email="happy.customer@email.com",
            subject="Great service!",
            message="I just wanted to say thank you for the excellent customer service. The new features you released last month are fantastic and have really improved our workflow."
        ),
        SupportTicket(
            ticket_id="T005",
            customer_email="confused.user@email.com",
            subject="How to export data?",
            message="I need to export all my data from the platform but can't find the export option. Could you please guide me through the process?"
        )
    ]
    return tickets

def main():
    """Main function to demonstrate the customer support agent"""
    print("=== Customer Support AI Agent System ===\n")
    
    # Initialize the agent
    try:
        # Try Ollama first (make sure Ollama is running with: ollama serve)
        # and the model is pulled with: ollama pull mistral:7b
        agent = CustomerSupportAgent(use_ollama=True)
        print("✅ Successfully connected to Ollama")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("Please ensure Ollama is running and the model is available.")
        print("Run: ollama serve")
        print("Run: ollama pull mistral:7b")
        return
    
    # Create sample tickets
    sample_tickets = create_sample_tickets()
    
    # Process each ticket
    results = []
    for ticket in sample_tickets:
        print(f"\n--- Processing Ticket {ticket.ticket_id} ---")
        print(f"Subject: {ticket.subject}")
        print(f"From: {ticket.customer_email}")
        print(f"Message: {ticket.message[:100]}...")
        
        try:
            result = agent.process_ticket(ticket)
            results.append(result)
            
            print(f"✅ Category: {result['category']}")
            print(f"✅ Priority: {result['priority']}")
            print(f"✅ Escalated: {result['escalated']}")
            print(f"✅ Status: {result['status']}")
            print(f"✅ Response: {result['response'][:200]}...")
            
        except Exception as e:
            print(f"❌ Error processing ticket: {e}")
    
    # Summary
    print(f"\n=== Processing Summary ===")
    print(f"Total tickets processed: {len(results)}")
    print(f"Escalated: {sum(1 for r in results if r['escalated'])}")
    print(f"Auto-responded: {sum(1 for r in results if not r['escalated'])}")
    
    # Show ticket categories
    categories = {}
    for result in results:
        cat = result['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nTicket Categories:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
    
    # Show memory usage
    print(f"\nMemory Store: {len(agent.memory_store)} conversations tracked")
    
    # Example of accessing ticket history
    if "T001" in agent.memory_store:
        print(f"\nExample - Ticket T001 History:")
        history = agent._get_ticket_history("T001")
        print(f"  {history[:200]}...")

if __name__ == "__main__":
    main()