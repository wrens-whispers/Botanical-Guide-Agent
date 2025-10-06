import os
import re
import pandas as pd
import sys
import random
import json
from openai import OpenAI
from typing import Literal
import streamlit as st
import httpx # Required for custom timeout configuration

# ======================================================================
# 1. ENVIRONMENT AND API CONFIGURATION
# ======================================================================

# Using the user-specified model
MODEL_ID = "google/gemma-3-27b-it:free"

# The global client will be initialized in run_streamlit_app
client: OpenAI = None

# ======================================================================
# 2. CONSTANTS AND DATA MAPPING
# ======================================================================

PLANT_SEQUENCE = [
    'cacao', 'tea', 'mate', 'lemon', 'cardamom',
    'ginger', 'black pepper', 'osmanthus', 'frankincense', 'bay leaf'
]
VOICE_OPTIONS = ['elder', 'child', 'evidence']
VOICE_MAPPING = {
    'elder': 'Elder Herbalist',
    'child': 'Child Herbalist',
    'evidence': 'Evidence-Based Herbalist'
}

# The document content (omitted for brevity, assume it is unchanged from previous versions)
DOC_TEXT = """
---
1. Cacao
Latin Name: Theobroma cacao
Native Region: Central and South America
Plant Part Used: Seed (commonly called the “bean”); fruit pulp occasionally used
Contraindications: Contains caffeine and theobromine; may not be suitable for people with caffeine sensitivity or certain heart conditions
Elder Herbalist: A sacred plant of the heart. The cacao bean is not just a treat, it is a ceremony—used to uplift mood and spirit, energize the body, and connect with others. The pulp surrounding the bean is also nourishing, though less commonly used.
Child Herbalist: Cacao is what chocolate is made from! It’s a bean from a fruit and has a yummy smell and taste. Sometimes it’s made into sweet drinks or even used in special ceremonies.
Evidence-Based Herbalist: Contains theobromine and flavonoids that may support mood, circulation, and cardiovascular health. Also a source of magnesium and antioxidants. May mildly increase blood pressure or heart rate.
---
2. Tea (Camellia)
Latin Name: Camellia sinensis
Native Region: East Asia
Plant Part Used: Leaves and buds
Contraindications: Contains caffeine; may interfere with iron absorption; should be used with caution in pregnancy or insomnia
Elder Herbalist: A leaf of clarity and peace. Depending on how it is processed, tea may be light and uplifting (green), grounding (black), or mellow (white). When used externally, it cools and clears skin eruptions.
Child Herbalist: Tea leaves can make green or brown drinks, depending on how how they’re made. Green tea can help your skin feel calm, and tea helps you stay awake during the day!
Evidence-Based Herbalist: Contains caffeine, theanine, and catechins like EGCG. Green tea is antioxidant-rich and may support cardiovascular, metabolic, and cognitive health. Topical application has shown anti-inflammatory effects.
---
3. Yerba Maté
Latin Name: Ilex paraguariensis
Native Region: South America (especially Argentina, Paraguay, and Brazil)
Plant Part Used: Leaves
Contraindications: Contains caffeine-like compounds; avoid excessive use in individuals with anxiety, insomnia, or high blood pressure
Elder Herbalist: This smoky, bitter leaf is revered for endurance and energy. It awakens the mind while providing nutrients to the blood. Traditionally shared in community through the gourd and bombilla.
Child Herbalist: Maté is like a super strong tea that people in South America love. It helps you feel awake and strong, and they drink it with a special straw.
Evidence-Based Herbalist: Contains caffeine, theobromine, and antioxidants. Shown to improve alertness and may have cholesterol-lowering effects. Overuse may increase cancer risk when consumed at high temperatures over time.
---
4. Lemon
Latin Name: Citrus limon
Native Region: Asia (likely India and China)
Plant Part Used: Fruit (juice and peel), leaves occasionally
Contraindications: Acidic; may aggravate ulcers or sensitive stomachs in excess
Elder Herbalist: Bright and purifying, lemon clears dampness and stimulates digestion. The peel is aromatic and warming, while the juice refreshes and detoxifies.
Child Herbalist: Lemons are super sour fruits that make lemonade! They’re also great for cleaning and have lots of vitamin C to help keep you healthy.
Evidence-Based Herbalist: Rich in vitamin C, citric acid, and flavonoids like hesperidin and quercetin. Shown to support immunity and digestion. Peel oil contains limonene with antimicrobial properties.
---
5. Cardamom
Latin Name: Elettaria cardamomum
Native Region: Southern India
Plant Part Used: Seed pods (and seeds inside)
Contraindications: Generally safe in culinary doses; avoid concentrated use in gallstone conditions
Elder Herbalist: Warming and harmonizing, cardamom moves stagnant food and awakens the Spleen Qi. Often used to balance heavy foods and settle the stomach.
Child Herbalist: Cardamom smells sweet and spicy! It helps with tummy aches and makes your food smell super good, especially in chai tea.
Evidence-Based Herbalist: Used to reduce bloating, gas, and support digestive enzymes. Contains essential oils (cineole, terpinyl acetate) with antimicrobial and anti-inflammatory activity.
---
6. Ginger
Latin Name: Zingiber officinale
Native Region: Southeast Asia
Plant Part Used: Rhizome (root)
Contraindications: May interact with blood-thinning medications. Use with caution in those with gallstones.
Elder Herbalist: Ginger is a revered warming root, used to kindle the digestive fire and dispel internal cold. We use it fresh for surface conditions, such as colds and chills, and dried when deeper warmth is needed. It invigorates the blood and aids in moving stagnant Qi.
Child Herbalist: Ginger is like a spicy superhero root! It helps your tummy feel better when it's upset or if you're feeling carsick. Some people even drink ginger tea to help them feel warm and cozy when they're cold.
Evidence-Based Herbalist: Ginger has been shown to be effective in reducing nausea from motion sickness and pregnancy. It exhibits anti-inflammatory properties and may assist in reducing joint pain. Research suggests benefits for gastrointestinal motility and digestion.
---
7. Black Pepper
Latin Name: Piper nigrum
Native Region: South India
Plant Part Used: Dried unripe fruit (peppercorn)
Contraindications: Generally safe in food amounts. Large quantities may irritate gastrointestinal tract.
Elder Herbalist: A pungent spice used to awaken the senses and open the channels. It can be combined with herbs like turmeric to enhance absorption and move internal cold.
Child Herbalist: Black pepper is the spicy specks you see on food! It adds flavor and can help your body stay warm. A little goes a long way!
Evidence-Based Herbalist: Piperine, the active compound in black pepper, increases the bioavailability of nutrients such as curcumin. It is used in formulations for digestion and metabolic support.
---
8. Osmanthus
Latin Name: Osmanthus fragrans
Native Region: China and the Himalayas
Plant Part Used: Flower
Contraindications: No known major contraindications; generally used in culinary amounts.
Elder Herbalist: Fragrant flower of the autumn season, Osmanthus uplifts the spirit and soothes the heart. In traditional practice, it is used to harmonize the stomach and promote joy.
Child Herbalist: Osmanthus smells like happiness in a flower! Some teas and treats are made with it, and it's like nature’s perfume.
Evidence-Based Herbalist: Osmanthus fragrans has been found to contain essential oils with potential antioxidant and mild sedative effects. It is primarily used for its fragrance in teas and confections.
---
9. Frankincense
Latin Name: Boswellia serrata (or Boswellia sacra, depending on species)
Native Region: Middle East and Northern Africa
Plant Part Used: Resin
Contraindications: May interact with anti-inflammatory medications. Avoid during pregnancy unless supervised.
Elder Herbalist: Sacred resin brought by wise ones. Used in ceremonies and for healing the pain of the joints. Often combined with myrrh for deeper blood-moving effect.
Child Herbalist: Frankincense is a shiny golden sap that smells really nice. People used it a long time ago in special ceremonies, and even now it's used in oils to help sore spots feel better.
Evidence-Based Herbalist: Boswellia resin contains boswellic acids, which have demonstrated anti-inflammatory effects in osteoarthritis and other inflammatory conditions. Used topically and internally in some traditional systems.
---
10. Bay Leaf
Latin Name: Laurus nobilis
Native Region: Mediterranean region
Plant Part Used: Leaf
Contraindications: Generally safe in culinary amounts. Large doses may cause drowsiness or affect blood sugar.
Elder Herbalist: A gentle leaf added to broths and stews to harmonize the flavors and aid digestion. Also used in warming salves for the joints.
Child Herbalist: Bay leaf is like a magic flavor leaf! It doesn’t taste good by itself, but when you cook it in soup or sauce, it makes everything taste better.
Evidence-Based Herbalist: Bay leaf (Laurus nobilis) contains essential oils and flavonoids. Some research supports mild antimicrobial and anti-inflammatory properties. Commonly used in culinary medicine to support digestion.
"""

# ======================================================================
# 3. DATA LOADING & PARSING
# ======================================================================

def load_and_structure_plant_data(doc_text: str, sequence: list, voice_map: dict) -> pd.DataFrame:
    """Parses the text, extracts details, and creates a DataFrame with one row per plant/voice combination."""

    plant_blocks = re.split(r'\n---\n\s*\d+\.\s', doc_text)[1:]
    data_list = []

    try:
        for block in plant_blocks:
            lines = block.strip().split('\n')

            plant_name_line = lines[0].split('(')[0].strip().lower()
            plant_name = next((p for p in sequence if p in plant_name_line), plant_name_line.split(' ')[0])
            if plant_name == 'yerba': plant_name = 'mate'

            # Extract fixed info
            info = {line.split(':')[0].strip(): line.split(':')[-1].strip()
                            for line in lines if ':' in line and not line.startswith(tuple(voice_map.values()))}

            # Extract specific notes for each voice
            for short_voice, long_voice in voice_map.items():
                try:
                    note_start_index = next(i for i, line in enumerate(lines) if line.startswith(long_voice))
                    single_note = lines[note_start_index].split(':', 1)[-1].strip()
                except StopIteration:
                    single_note = "ERROR: Note not found in document."

                # Store one row for each (plant, voice) combination
                data_list.append({
                    'plant': plant_name,
                    'voice': short_voice,
                    'note': single_note, # This is the short note we'll expand
                    'latin_name': info.get('Latin Name', 'N/A'),
                    'origin': info.get('Native Region', 'N/A'),
                    'parts_used': info.get('Plant Part Used', 'N/A'),
                    'contraindications': info.get('Contraindications', 'None known'),
                })

        return pd.DataFrame(data_list)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to load plant data. Check DOC_TEXT structure.")
        print(f"Details: {e}")
        sys.exit(1)


# ======================================================================
# 4. AGENT CLASS (LOGIC & FLOW)
# ======================================================================

class BotanicalGuideAgent:

    def __init__(self, plant_data: pd.DataFrame, sequence: list, voice_options: list):
        self.plant_data = plant_data
        self.plant_sequence = sequence
        self.voice_options = voice_options

        # State Tracking
        self.current_voice = None
        self.current_plant = sequence[0]
        self.current_plant_index = 0

        # State Tracking for reading parts
        self.current_reading_step = 0
        self.expanded_readings: list[str] = []

        self.REDIRECT_RESPONSES = [
            "Interesting thought! These plants adapt in their own ways too. Let's return to {plant}.",
            "Ha, that's a good one. Speaking of growth, {plant} has its own story…",
            "Noted. Our tour is focused on the plants—here's more about {plant}."
        ]

    def _build_system_prompt(self, current_plant_row, user_input: str) -> str:
        """
        Builds the system prompt to enforce a structured prose output (Part 1, Part 2, Part 3)
        and guides the LLM on how to handle the user's input while maintaining guardrails.
        """

        plant_name = current_plant_row['plant'].capitalize()
        target_voice = current_plant_row['voice']

        # Format the core instruction and data
        prompt = (
            f"You are a Botanical Garden Tour Guide for the plant **{plant_name}**. "
            f"Your persona is the **{target_voice.upper()}** herbalist. "
            f"Your role is to deliver a three-part scripted reading based on your 'notecards'. "
            f"Your sole purpose is to provide structured information on the current plant.\n\n"

            f"DATA:\n"
            f"Latin Name: {current_plant_row['latin_name']}\n"
            f"Region: {current_plant_row['origin']}\n"
            f"Parts Used: {current_plant_row['parts_used']}\n"
            f"Contraindications: {current_plant_row['contraindications']}\n"
            f"Short Note: {current_plant_row['note']}\n\n"

            f"USER INPUT: '{user_input}'\n\n"

            f"INSTRUCTIONS:\n"
            f"1. **Primary Guardrail:** Your response MUST stay focused on the plant. If the USER INPUT is off-topic (e.g., about movies, weather, or pricing), give a very brief, gentle acknowledgment, and immediately proceed to the structured reading.\n"
            f"2. **Flow Control:** If the USER INPUT contains commands related to state change (like 'next plant', 'ginger', or another 'voice'), **ignore those commands** as the main application handles navigation. Only focus on interpreting general questions.\n"
            f"3. **Your response MUST follow this exact, labeled structure** and contain ONLY the three parts of the reading:\n"
            f"    - **Part 1: History and Origin**\n"
            f"    - **Part 2: Key Features and Uses**\n"
            f"    - **Part 3: Scientific Details and Context**\n"
            f"4. **Do not include any other text** (no ending summary, no markdown wrappers)."
        )

        return prompt

    def _get_expanded_reading(self, user_input: str) -> tuple[str, dict]:
        """
        Collects data, sends the prompt, and splits the LLM's prose response
        into three parts based on the labeled structure.
        """

        try:
            plant_row = self.plant_data[
                (self.plant_data['plant'] == self.current_plant) &
                (self.plant_data['voice'] == self.current_voice)
            ].iloc[0]
        except IndexError:
            return "Error: Plant data not found for the current plant and voice.", {}

        # 1. Generate the LLM System Prompt with the user's raw input
        system_prompt = self._build_system_prompt(plant_row, user_input)

        # 2. Call the LLM to generate the structured readings (raw prose string)
        prose_string_raw = generate_llm_response(system_prompt)

        # --- ROBUST PROSE PARSING FIX ---
        # Look for the predefined Part labels to split the response
        parts = re.split(r'(?:\*\*|)\s*Part\s+\d+:\s+.*?(?:\*\*|)', prose_string_raw.strip(), re.DOTALL)

        # Fallback list for error cases
        self.expanded_readings = []
        reading_text = ""

        # Check if the split produced the expected number of sections
        if len(parts) >= 4:
            # Store the three main parts (indices 1, 2, 3)
            self.expanded_readings = [p.strip() for p in parts[1:4]]

            if self.expanded_readings[0]:
                reading_text = self.expanded_readings[0] # Part 1 content
                self.current_reading_step = 1 # Set to start at the first reading
            else:
                reading_text = f"[LLM STRUCTURE ERROR] The guide generated structure but Part 1 was empty. Raw output begins: {prose_string_raw[:200]}..."
        else:
            # The model failed to adhere to the Part structure, or returned an error/empty content
            reading_text = f"[LLM STRUCTURE ERROR] The guide failed to generate the reading correctly. Raw output begins: {prose_string_raw[:200]}..."

        # Extract fixed info for local formatting
        fixed_info = {
            'Latin Name': plant_row['latin_name'],
            'Region of Origin': plant_row['origin'],
            'Parts Used': plant_row['parts_used'],
            'Contraindications': plant_row['contraindications'],
        }

        return reading_text, fixed_info

    def _get_next_reading_part(self) -> str:
        """Retrieves and increments the reading part counter."""

        if self.current_reading_step < len(self.expanded_readings):
            next_part_content = self.expanded_readings[self.current_reading_step]
            self.current_reading_step += 1
            return next_part_content
        else:
            # All parts have been read
            self.current_reading_step = 0 # Reset state
            self.expanded_readings = []
            return f"That concludes the full reading on **{self.current_plant.capitalize()}**. **Ready for the next plant?**"


    def _generate_reading(self, user_input: str) -> str:
        """Calls the LLM data fetcher and formats the first part of the reading."""

        narrative, fixed_info = self._get_expanded_reading(user_input)

        # Build the formatted response with fixed info
        response = ""
        response += f"***\n"
        response += f"**Latin Name:** {fixed_info.get('Latin Name', 'N/A')} | "
        response += f"**Region:** {fixed_info.get('Region of Origin', 'N/A')} | "
        response += f"**Parts Used:** {fixed_info.get('Parts Used', 'N/A')}\n"
        response += f"***\n\n"

        # Check if an error occurred during expansion
        if "[LLM STRUCTURE ERROR]" in narrative:
            response += narrative
            response += "\n\n**Please try another command or quit.**"
            return response

        # If successful, format the first part
        response += f"**[Expanded Reading Part 1/3 - {self.current_voice.upper()} Persona]**\n"
        response += narrative

        # We now ask for the NEXT reading.
        if len(self.expanded_readings) > 1:
            response += "\n\n**Continue reading this plant's story?**"
        else:
            response += "\n\n**Ready for the next plant?**"

        return response

    # --- VOICE SELECTION ---
    def _handle_select_voice(self, user_input: str) -> str:
        """Sets the voice, resets step counter, and then performs a reading."""

        new_voice = next((v for v in self.voice_options if v in user_input.lower()), None)

        if not new_voice:
            # Since the LLM handles interpretation, any non-voice command is treated as a redirect/question
            return self._handle_redirect(user_input)

        # Reset reading state whenever voice or plant changes
        self.current_reading_step = 0
        self.expanded_readings = []

        # The LLM prompt is now responsible for incorporating the user's conversational intent

        if self.current_voice is None:
            self.current_voice = new_voice
            return f"Great choice! I'm excited to share our garden with you from the perspective of a **{new_voice}** herbalist. Let's begin with **{self.current_plant.capitalize()}**.\n\n" + self._generate_reading(user_input)

        elif new_voice != self.current_voice:
            self.current_voice = new_voice
            return f"Voice switched to **{new_voice.upper()}** persona. Here is the expanded note on **{self.current_plant.capitalize()}**:\n\n" + self._generate_reading(user_input)

        else:
            # If same voice is selected, regenerate the reading for the current plant (or continue if reading is active)
            if self.current_reading_step == 0:
                return f"The voice is already set to **{self.current_voice}**. Let's start the reading on **{self.current_plant.capitalize()}**.\n\n" + self._generate_reading(user_input)
            else:
                return self._handle_continue_reading(user_input)


    # --- NAVIGATION ---
    def _handle_plant_navigation(self, user_input: str) -> str:
        """Handles 'next plant' or 'plant by name', and then performs a reading."""

        if self.current_voice is None:
            return f"Please select your preferred herbalist persona: {', '.join(self.voice_options)}."

        # 1. Check for 'next reading' command
        if self.current_reading_step > 0 and any(w in user_input.lower() for w in ["next reading", "continue", "tell me more"]):
            return self._handle_continue_reading(user_input)

        # 2. Check for 'next plant'
        if "next plant" in user_input.lower():
            if self.current_plant_index == len(self.plant_sequence) - 1:
                return "We've completed the tour of all 10 plants. Please select a plant by name if you wish to re-visit one."

            # Reset reading state when moving to a new plant
            self.current_reading_step = 0
            self.expanded_readings = []

            self.current_plant_index = (self.current_plant_index + 1) % len(self.plant_sequence)
            self.current_plant = self.plant_sequence[self.current_plant_index]

            return f"Moving to the next plant, **{self.current_plant.capitalize()}**.\n\n" + self._generate_reading(user_input)

        # 3. Check for 'plant by name'
        named_plant = next((p for p in self.plant_sequence if p in user_input.lower()), None)
        if named_plant:
            # Reset reading state when switching to a named plant
            self.current_reading_step = 0
            self.expanded_readings = []

            self.current_plant = named_plant
            self.current_plant_index = self.plant_sequence.index(named_plant)
            return f"Switching focus to **{self.current_plant.capitalize()}**.\n\n" + self._generate_reading(user_input)

        # If input was not a recognized command, it's a redirect.
        return self._handle_redirect(user_input)


    def _handle_continue_reading(self, user_input: str) -> str:
        """Retrieves and formats the next part of the expanded reading."""

        # If no reading is active, assume the user is asking a general question about the current plant
        if not self.expanded_readings or self.current_reading_step == 0:
            return self._generate_reading(user_input)

        # Retrieve the next part of the story
        next_reading = self._get_next_reading_part()

        if self.current_reading_step == 0:
            return next_reading # Returns the "That concludes the full reading..." message

        response = f"**[Expanded Reading Part {self.current_reading_step}/3 - {self.current_voice.upper()} Persona]**\n"
        response += next_reading

        if self.current_reading_step < 3:
            response += "\n\n**Continue reading this plant's story?**"
        else:
            response += "\n\nThat concludes the full reading on this plant. **Ready for the next plant?**"
            self.current_reading_step = 0
            self.expanded_readings = []

        return response

    def _handle_redirect(self, user_input: str) -> str:
        """
        Handles inputs that are not clear navigation or voice commands.
        Triggers a new reading but allows the LLM to interpret the user's question first.
        """
        if self.current_voice is None:
            return f"Welcome! Please select your preferred herbalist persona: {', '.join(self.voice_options)}."

        # Any conversational input that isn't a direct command is treated as a question about the current plant.
        # This forces the LLM to process it (e.g., "what's the weather") and redirect, then deliver the script.
        return self._generate_reading(user_input)


    def respond(self, user_input: str) -> str:
        """The main interaction method that executes the logic."""
        user_input_lower = user_input.lower().strip()

        # 0. Initial Greeting / Persona Selection Prompt
        if self.current_voice is None:
            if any(v in user_input_lower for v in self.voice_options):
                return self._handle_select_voice(user_input)
            return f"Welcome! Please select your preferred herbalist persona: {', '.join(self.voice_options)}."


        # 1. COMMAND: Continue Reading (High Priority)
        if self.current_reading_step > 0 and any(w in user_input_lower for w in ["continue", "next reading", "more", "tell me more"]):
            return self._handle_continue_reading(user_input)

        # 2. COMMAND: Change Voice
        if any(v in user_input_lower for v in self.voice_options):
            return self._handle_select_voice(user_input)

        # 3. COMMAND: Next Plant or Plant by Name
        plant_commands = ["next plant"] + self.plant_sequence
        if any(p in user_input_lower for p in plant_commands):
            return self._handle_plant_navigation(user_input)

        # 4. Handle all other inputs (Redirect/Vague Question)
        # This now routes ALL conversational inputs back to _generate_reading
        # via _handle_redirect, allowing the LLM prompt to manage the guardrail and content delivery.
        return self._handle_redirect(user_input)

# ======================================================================
# 5. API INTERACTION FUNCTION
# ======================================================================

def generate_llm_response(system_prompt_content: str) -> str:
    """Sends the system prompt to the chosen LLM for prose generation (structured expansion)."""

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt_content},
            ],
            temperature=0.5,
            max_tokens=1024
        )

        # Log success (only visible in Streamlit Cloud logs)
        print(f"DEBUG A: LLM API call SUCCESS.")

        raw_content = response.choices[0].message.content

        # --- FAIL-SAFE CHECK ---
        if not raw_content or raw_content.isspace():
            print(f"[LLM CONTENT FAIL] Model returned empty or blank content for prompt.")
            return "[EMPTY CONTENT ERROR] The LLM returned a blank response."
        # ---------------------------

        # Return the raw content
        return raw_content

    except Exception as e:
        # Critical failure if the LLM can't generate the reading
        print(f"\n[CRITICAL LLM ERROR] Failed to expand reading: {type(e).__name__} - {e}")
        return f"[CRITICAL LLM ERROR] Failed to expand reading: {type(e).__name__} - {e}"

# ======================================================================
# 6. STREAMLIT UI RUNNER
# ======================================================================

def run_streamlit_app():
    # 1. Configuration and Title
    st.set_page_config(page_title="Botanical Guide Agent", layout="centered")
    st.title("🌿 Interactive Botanical Guide")
    st.markdown("---")

    # 2. API Key Loading (using Streamlit's secrets for security)
    try:
        openrouter_key = st.secrets["OPENROUTER_API_KEY"]
    except KeyError:
        st.error("API Key not found in Streamlit Secrets. Please configure `OPENROUTER_API_KEY`.")
        return

    # 3. Initialization and Agent Setup
    if 'agent' not in st.session_state:

        # CRITICAL: Initialize the global client with the loaded secret
        global client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            # Increased request timeout to 120 seconds to handle complex generation
            timeout=httpx.Timeout(120.0, connect=30.0)
        )

        # Initialize the rest of the agent's state
        PLANT_DATA = load_and_structure_plant_data(DOC_TEXT, PLANT_SEQUENCE, VOICE_MAPPING)
        st.session_state.agent = BotanicalGuideAgent(PLANT_DATA, PLANT_SEQUENCE, VOICE_OPTIONS)
        st.session_state.messages = []
        st.session_state.messages.append(
            {"role": "agent", "content": st.session_state.agent.respond("")}
        )

    # 4. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. Handle User Input
    if prompt := st.chat_input("Enter your command (e.g., 'elder', 'continue', 'next plant', 'ginger')..."):
        # Add user prompt to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("agent"):
            # The agent's respond method handles all the logic and LLM calls
            response = st.session_state.agent.respond(prompt)
            st.markdown(response)

            # Update history with agent's response
            st.session_state.messages.append({"role": "agent", "content": response})

# Run the app function when the script starts
if __name__ == "__main__":
    run_streamlit_app()
