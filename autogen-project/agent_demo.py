import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Configuration for OpenAI
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": "your_api_key"   
    }
]

# Create two agents
agent_a = AssistantAgent(
    name="AgentA",
    system_message="You are Agent A. You are helpful and like to ask questions. Always respond briefly and then ask a follow-up question.",
    llm_config={"config_list": config_list}
)

agent_b = AssistantAgent(
    name="AgentB", 
    system_message="You are Agent B. You are knowledgeable and like to provide information. Always respond briefly and then ask a follow-up question.",
    llm_config={"config_list": config_list}
)

# Create a user proxy to initiate the conversation
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",  # No human input needed
    max_consecutive_auto_reply=10,  # Allow up to 10 back-and-forth messages
    llm_config={"config_list": config_list}
)

def main():
    print("🤖 Starting A2A (Agent-to-Agent) Communication Demo")
    print("=" * 50)
    
    # Start the conversation between the two agents
    user_proxy.initiate_chat(
        agent_a,
        message="Hello Agent A! I'd like you to start a conversation with Agent B. Please introduce yourself and ask Agent B a question."
    )
    
    # Now let Agent A chat with Agent B
    agent_a.initiate_chat(
        agent_b,
        message="Hello Agent B! I'm Agent A. How are you today? What's your favorite topic to discuss?"
    )

if __name__ == "__main__":
    main() 