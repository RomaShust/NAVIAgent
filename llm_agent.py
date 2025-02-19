# llm_agent.py

import requests
import json
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from entity_extraction import extract_entities
from langchain_openai import ChatOpenAI


def search_npi_registry(location: str, state: str, specialty: str):
    """
    Searches the NPI Registry API with given criteria.
    Uses location as a zip code if numeric and 5 digits; otherwise, treats it as a city.
    Returns up to 10 physician records with name, address, specialty, and phone number.
    Only includes records where a valid name (non-empty and not "None") is provided.
    """
    base_url = "https://npiregistry.cms.hhs.gov/api/"
    params = {
        "version": "2.1",
        "limit": 20,
    }
    # Determine if location is a zip code
    if location.isdigit() and len(location) == 5:
        params["postal_code"] = location
    else:
        params["city"] = location

    if state:
        params["state"] = state
    if specialty:
        # The API expects a taxonomy description string
        params["taxonomy_description"] = specialty

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return None  # Signal an error

    data = response.json()
    results = []
    if "results" in data:
        for result in data["results"]:
            basic = result.get("basic", {})
            # Use .get() with fallback to empty string and then strip
            first_name = basic.get("first_name") or ""
            last_name = basic.get("last_name") or ""
            name = f"{first_name.strip()} {last_name.strip()}".strip()
            # Skip if the name is empty or appears to be "None"
            if not name or name.lower() in ["none", "none none"]:
                continue

            addresses = result.get("addresses", [])
            # Look for the "LOCATION" address
            address = "N/A"
            for addr in addresses:
                if addr.get("address_purpose", "").upper() == "LOCATION":
                    address = f"{addr.get('address_1', '').strip()}, {addr.get('city', '').strip()}, {addr.get('state', '').strip()} {addr.get('postal_code', '').strip()}"
                    break

            # Attempt to find a phone number
            phone = "N/A"
            for addr in addresses:
                if addr.get("telephone_number"):
                    phone = addr.get("telephone_number").strip()
                    break

            specialty_value = result.get("taxonomies", [{}])[0].get("desc", "N/A")
            results.append({
                "name": name,
                "address": address,
                "specialty": specialty_value,
                "phone": phone
            })
    return results



def npi_tool_func(query: str, extraction_llm: ChatOpenAI) -> str:
    """
    Uses the LLM-based entity extraction to parse the query, then performs the NPI Registry search.
    """
    entities = extract_entities(query, extraction_llm)
    location = entities["location"]
    state = entities["state"]
    specialty = entities["specialty"]


    print("Extracted Entities:", location, state, specialty)  # Debug output
    results = search_npi_registry(location, state, specialty)
    print("Results from NPI search:", results)  # Debug output
    
    if results is None:
        return "Error retrieving data from the NPI Registry API."
    if len(results) == 0:
        crit = []
        if specialty:
            crit.append(specialty)
        crit.append(state if state else location)
        criteria_str = " in ".join(crit)
        return f"It seems that no physicians were found for the search criteria of a {criteria_str}. Please try adjusting your query."
    
    output = ""
    for idx, res in enumerate(results, start=1):
        output += f"**Physician {idx}:**\n"
        output += f"- **Name:** {res['name']}\n"
        output += f"- **Address:** {res['address']}\n"
        output += f"- **Specialty:** {res['specialty']}\n"
        output += f"- **Phone:** {res['phone']}\n\n"
        
    # Append debugging information for the user
    #debug_info = f"\n**DEBUG:** Search parameters used: location = '{location}', state = '{state}', specialty = '{specialty}'"
    #output += debug_info
        
    return json.dumps({"output": output})


def create_agent(openai_api_key: str, model_name: str = "gpt-4o", temperature: float = 0.0):
    """
    Creates the main agent for conversation,
    plus a second LLM for entity extraction.
    """
    # This LLM is used by the agent for conversation
    conversation_llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key
    )

    # This separate LLM is used specifically for entity extraction
    extraction_llm = ChatOpenAI(
        model_name=model_name, 
        temperature=0.0,
        openai_api_key=openai_api_key
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    # Add a system prompt for the main conversation agent
    system_prompt = """
 You are a helpful chatbot that assists users in finding medical providers.

 **Key Behaviors**:

 1. **One-time introduction**: At the very start of the conversation, greet the user with:
 “I am a chatbot, here to help you find a medical provider for your current needs. 
     How can I help you today?”

 - After the initial greeting, **do not** repeat the exact same introduction again.

 2. **Emergency rule**: If a user mentions urgent or life-threatening symptoms (e.g., strong chest pain):
 - **Immediately** advise them to call 911 (or local emergency services) or go to the nearest ER.
 - Acknowledge that you are not a medical professional and cannot diagnose.
 - If they recover or come back with a separate, non-emergency request (e.g., “help me find an ophthalmologist”), 
 **do** help them with that request without reintroducing yourself.

 3. **Referral check**: 
 - Before you perform a search for a medical provider, **ask once** if the user has a referral from their family 
 doctor (e.g. “Did you get a referral from your family doctor?”).
 - If the user says “no” or “I don’t need one,” **do not** ask again. Proceed to the search.
 - If the user says “yes,” note that they have a referral and still proceed with the search if they want.

 4. **No repeated intros**: **Never** restate the line “I am a chatbot, here to help you…” after you have already 
 done it once. Once the user answers about a referral, do not re-ask.

 5. **No medical diagnosis**: 
 - You are **not** a medical professional and cannot provide a diagnosis or treatment.
 - Always remind the user that you are not diagnosing them and are simply providing names or addresses of providers.

 6. **Mental health referrals**: 
 - Only mention mental health professionals if the user explicitly requests it or reveals mental distress.

 7. **Final output**:
 - When the user asks for a provider (e.g., an ophthalmologist in Portland), provide a short disclaimer 
 (e.g., “I’m not a medical professional, but here is some information…”), output debug search parameters from 
 the NPI search tool, and list the provider results: name, address, specialty, phone.
 - If no providers are found, politely say so.
 
 - **Do not** reintroduce yourself or re-ask the referral question repeatedly.x
 
 
 **Example 1**:
 User:  “I have strong chest pain,” 
 Agent: “I’m not a medical professional, but chest pain can be serious. Call 911 or go to an ER. Let me know if you need 
 non-emergency info or a cardiologist referral afterward.”


 **Example 2**:
 User:“ Help me find an ophthalmologist in Portland,” 
 Agent: “Did you get a referral from your family doctor?”  
 User: “No” 
 Agent:“Okay, here are some ophthalmologists in Portland: [list]. Don't forget to ask for a referal from your doctor.
 Please make sure to contact them directly to confirm their availability and whether they accept your insurance.”  
        
    """
    memory.chat_memory.add_message(SystemMessage(content=system_prompt))

    # Wrap npi_tool_func so the agent can call it by name
    def npi_tool_wrapper(query: str) -> str:
        return npi_tool_func(query, extraction_llm)

    # Create a single "Tool" that uses the NPI search
    tool = Tool(
        name="NPI_Registry_Search",
        func=npi_tool_wrapper,
        description="Searches for physicians using the NPI Registry API. "
                    "Provide a natural language request (e.g. 'I need a cardiologist in LA')."
    )

    # Initialize the agent with the conversation LLM and the new tool
    agent = initialize_agent(
        [tool],
        conversation_llm,
        agent='conversational-react-description',
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
                "system_message": system_prompt
            }
    )
    return agent
