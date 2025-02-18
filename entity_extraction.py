# entity_extraction.py

import json
import re
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

def extract_entities(query: str, extraction_llm: ChatOpenAI) -> dict:
    """
    Calls a separate LLM with strict instructions to:
      1) Extract location, state, and specialty from user text.
      2) Map the user's specialty request to the closest match in a known list.
      3) Return ONLY valid JSON in the format:
         {"location":"...", "state":"...", "specialty":"..."}
      4) If an entity is missing, use an empty string.
    """

    # ----------------------------------------------------------------
    # The official list of specialties you want the LLM to match against:
    # ----------------------------------------------------------------
    specialty_list = (
        "General Practice, Surgery, Allergy & Immunology, Otolaryngology, "
        "Anesthesiology, Internal Medicine, Cardiovascular Disease, Dermatology,"
        "Family Practice, Pain Medicine, Interventional Pain Medicine,"
        "Gastroenterology, Internal Medicine, Neuromusculoskeletal Medicine & OMM, Psychiatry & Neurology, Neurology,"
        "Neurological Surgery, Obstetrics & Gynecology, Ophthalmology, Oral & Maxillofacial Surgery, "
        "Orthopedic Surgery, Clinical Pathology, Plastic,"
        "Physical Medicine & Rehabilitation, Psychiatry, Colon & Rectal Surgery,"
        "Pulmonary Disease, Diagnostic Radiology, Anesthesiologist Assistant, Thoracic Surgery, "
        "Urology, Chiropractor, Nuclear Medicine, Pediatrics, Geriatric Medicine, Nephrology, "
        "Surgery of the Hand, Optometrist, Midwife, Nurse Anesthetist, "
        "Infectious Disease, Radiology, Mammography, Endocrinology, Physiological Laboratory,"
        "Podiatrist, Ambulatory Surgical, Nurse Practitioner, Orthotist, Ambulance," 
        "Prosthetist, Public Health or Welfare, Voluntary or Charitable, Psychologist,"
        "Audiologist, Physical Therapist, Rheumatology, Occupational Therapist, Psychologist/Clinical,"
        "Dietitian, Pain Medicine, Oncology, Surgery, Vascular Surgery, Addiction Medicine,"
        "Critical Care Medicine, Hematology, Hematology & Oncology, Preventive Medicine,"
        "Public Health & General Preventive Medicine, Oral & Maxillofacial Surgery, Psychiatry & Neurology,"
        "Clinical Nurse Specialist, Medical Oncology, Surgical Oncology, Radiation Oncology,"
        "Emergency Medicine, Vascular & Interventional Radiology, Optician, Obstetrics & Gynecology,"
        "Gynecologic Oncology"
    )

    # ----------------------------------------------------------------
    # System prompt: Tells the LLM to use that list to find a best match.
    # ----------------------------------------------------------------
    system_msg = SystemMessage(
        content=(
 "You are an assistant that extracts 3 fields from user text: location, state, and specialty.\n"
 "1) Parse the user's query to identify location (city or ZIP), state, and specialty."
 "The state should be ALWAYS abbreviated to two letters. \n"
 "2) For specialty, map the user's request to the closest match from the following official list:\n\n"
 f"{specialty_list}\n\n"
 "For example, if the user says 'cardiologist', return 'Cardiology'; if they say 'eye doctor', "
 "you might return 'Otolaryngology', etc.\n"
 "Return ONLY valid JSON in this format:\n"
 '{"location":"...", "state":"...", "specialty":"..."}\n'
 "No extra text, disclaimers, or keys."
        )
    )

    # The user message is just their raw query
    user_msg = HumanMessage(content=query)

    # ----------------------------------------------------------------
    # Call the extraction LLM with these messages
    # ----------------------------------------------------------------
    try:
        ai_message = extraction_llm.invoke([system_msg, user_msg])
        result_text = ai_message.content
        print("LLM raw text for entity extraction:", repr(result_text))
    except Exception as e:
        print("Error calling extraction LLM:", e)
        result_text = "{}"  # fallback if something goes wrong

    # If the LLM wrapped its JSON in ```...``` fences, remove them
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", result_text, re.DOTALL)
    if match:
        result_text = match.group(1).strip()

    # Try to parse the LLM's output as JSON
    try:
        data = json.loads(result_text)
    except Exception as e:
        print("Error parsing JSON for entity extraction:", e)
        data = {}

    # Normalize fields
    location = data.get("location", "").strip()
    state = data.get("state", "").strip()
    specialty = data.get("specialty", "").strip()

    return {
        "location": location,
        "state": state,
        "specialty": specialty
    }
