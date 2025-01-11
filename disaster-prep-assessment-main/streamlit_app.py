import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize WatsonX credentials
try:
    from ibm_watson_machine_learning.foundation_models import Model
    from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
    
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False
    st.warning("WatsonX integration not available. Installing in demo mode.")

def init_watsonx():
    """Initialize the WatsonX model with credentials and parameters"""
    if not WATSONX_AVAILABLE:
        return None

    credentials = {
        "url": os.getenv("WATSONX_URL"),
        "apikey": os.getenv("WATSONX_APIKEY")
    }

    model_params = {
        "decoding_method": "greedy",
        "max_new_tokens": 500,
        "min_new_tokens": 50,
        "temperature": 0.7
    }
    projectid = os.getenv("projectID")
    spaceid = os.getenv("spaceID")
    return Model(
        model_id=ModelTypes.GRANITE_13B_CHAT_V2,
        credentials=credentials,
        params=model_params,
        space_id=spaceid,
        project_id=projectid,
    )

def display_demo_results():
    """Display sample results when WatsonX is not available or fails"""
    st.info("Demo Mode: Would normally send to WatsonX for processing")
    st.write("### Sample Assessment Results")
    st.write("""
    Safety Scores:
    - Shelter: 75/100
    - People: 85/100
    - Supplies: 60/100
    
    Summary:
    Based on the provided information, this is a moderate-risk situation. Your shelter provides basic protection but has some vulnerabilities that should be addressed. The number of people and their specific needs are well-documented, which helps in planning. However, supply levels could be improved for better preparedness.
    
    Recommended Precautions:
    1. Strengthen shelter vulnerabilities
    2. Increase water supplies (recommend 1 gallon per person per day)
    3. Add emergency communication devices
    4. Create an evacuation plan
    5. Stock additional non-perishable food items
    """)

def format_data_display(data):
    """Format the session data in a readable way"""
    output = []
    
    if 'disaster' in data:
        output.append("**Disaster Information:**")
        output.append(f"- Type: {data['disaster'].get('type', 'N/A')}")
        output.append(f"- Intensity: {data['disaster'].get('intensity', 'N/A')}")
        output.append(f"- In Disaster Zone: {'Yes' if data['disaster'].get('currently_in') else 'No'}")
        output.append("")
    
    if 'shelter' in data:
        output.append("**Shelter Information:**")
        output.append(f"- Floors: {data['shelter'].get('floors', 'N/A')}")
        output.append(f"- Rooms: {data['shelter'].get('rooms', 'N/A')}")
        output.append(f"- Material: {data['shelter'].get('material', 'N/A')}")
        output.append(f"- Vulnerabilities: {', '.join(data['shelter'].get('vulnerabilities', ['None']))}")
        output.append("")
    
    if 'people' in data:
        output.append("**People Information:**")
        output.append(f"- Total Count: {data['people'].get('count', 'N/A')}")
        output.append(f"- Special Circumstances: {', '.join(data['people'].get('special_circumstances', ['None']))}")
        if data['people'].get('has_pets'):
            output.append(f"- Pets: {data['people'].get('pets_info', 'N/A')}")
        output.append("")
    
    if 'supplies' in data:
        output.append("**Supplies Information:**")
        output.append(f"- Food Supply: {data['supplies'].get('food_days', 'N/A')} days")
        output.append(f"- Water Supply: {data['supplies'].get('water_liters', 'N/A')} liters")
        output.append(f"- Available Gear: {', '.join(data['supplies'].get('gear', ['None']))}")
    
    return "\n".join(output)

def process_assessment():
    """Process the assessment data and generate recommendations"""
    # Check if each category in form_data exists before trying to access it
    disaster_data = st.session_state.form_data.get('disaster', {})
    shelter_data = st.session_state.form_data.get('shelter', {})
    people_data = st.session_state.form_data.get('people', {})
    supplies_data = st.session_state.form_data.get('supplies', {})
    
    # Format the prompt based on available data
    prompt = f"""
    Disaster Assessment:
    Type: {disaster_data.get('type', 'N/A')}
    Intensity: {disaster_data.get('intensity', 'N/A')}
    Currently In Disaster: {disaster_data.get('currently_in', 'N/A')}
    
    Shelter Details:
    Floors: {shelter_data.get('floors', 'N/A')}
    Rooms: {shelter_data.get('rooms', 'N/A')}
    Material: {shelter_data.get('material', 'N/A')}
    Vulnerabilities: {', '.join(shelter_data.get('vulnerabilities', []))}
    
    People:
    Total Count: {people_data.get('count', 'N/A')}
    Special Circumstances: {', '.join(people_data.get('special_circumstances', []))}
    Pets: {people_data.get('pets_info', 'None')}
    
    Supplies:
    Food Supply: {supplies_data.get('food_days', 'N/A')} days
    Water Supply: {supplies_data.get('water_liters', 'N/A')} liters
    Available Gear: {', '.join(supplies_data.get('gear', []))}

    Based on this information, please provide:
    1. Safety scores for shelter, people, and supplies (out of 100)
    2. A detailed summary of the risk level and situation
    3. Specific recommended precautions or actions, prioritized by importance
    4. Any additional considerations based on the special circumstances or pets
    """
    
    if WATSONX_AVAILABLE:
        try:
            model = init_watsonx()
            if model is None:
                raise Exception("Failed to initialize WatsonX model")
                
            # Generate response from WatsonX
            response = model.generate_text(prompt)
            
            # Check if response is a dictionary with results
            if isinstance(response, dict) and 'results' in response:
                generated_text = response['results'][0]['generated_text']
            else:
                generated_text = response
            
            # Create a container for the full-width display
            with st.container():
                # Display the assessment in a highlighted box
                st.markdown("""<style>
                    .assessment-box {
                        padding: 20px;
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        margin: 10px 0;
                    }
                    </style>""", unsafe_allow_html=True)
                
                st.markdown('<div class="assessment-box">', unsafe_allow_html=True)
                st.write("### 📊 Assessment Results")
                st.write(generated_text)
                st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing assessment: {str(e)}")
            st.info("Falling back to demo mode...")
            display_demo_results()
    else:
        display_demo_results()

def handle_navigation(direction):
    """Handle navigation between form steps"""
    if direction == "next":
        st.session_state.step += 1
    elif direction == "prev":
        st.session_state.step -= 1
    st.rerun()

def main():
    """Main application function"""
    st.title("Disaster Preparedness Assessment")
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    # Constants
    DISASTER_TYPES = ['Hurricane', 'Earthquake', 'Flood', 'Wildfire', 'Tornado']
    BUILDING_MATERIALS = ['Wood', 'Concrete', 'Brick', 'Steel']
    SPECIAL_CIRCUMSTANCES = ['Pregnant', 'Elderly', 'Underage', 'Disabled', 'Medical Conditions']
    GEAR_LIST = ['First Aid Kit', 'Flashlight', 'Batteries', 'Radio', 'Tools',
                 'Blankets', 'Medications', 'Important Documents', 'Cash', 'Clothing',
                 'Cell Phone Charger', 'Multi-tool', 'Emergency Contact List', 'Smoke Alarm', 'Fire Extinguisher', 'Generator']
    VULNERABILITIES = ['Structural damage', 'Poor insulation', 'Weak foundation', 
                       'Old wiring', 'Leaky roof', 'Large windows']

    # Progress bar
    progress = (st.session_state.step - 1) * 25
    st.progress(progress)
    
    # Step 1: Disaster Information
    if st.session_state.step == 1:
        st.subheader("Step 1: Disaster Information")
        
        disaster_type = st.selectbox("Type of Disaster", DISASTER_TYPES)
        intensity = st.slider("Intensity of Disaster (1-10)", min_value=1, max_value=10)
        currently_in = st.radio("Are you currently in the disaster zone?", ("Yes", "No"))
        
        st.session_state.form_data['disaster'] = {
            'type': disaster_type,
            'intensity': intensity,
            'currently_in': currently_in == "Yes"
        }
        
        if st.button("Next"):
            handle_navigation("next")
    
    # Step 2: Shelter Information
    elif st.session_state.step == 2:
        st.subheader("Step 2: Shelter Information")
        
        floors = st.number_input("Number of Floors", min_value=1, step=1)
        rooms = st.number_input("Number of Rooms", min_value=1, step=1)
        material = st.selectbox("Building Material", BUILDING_MATERIALS)
        vulnerabilities = st.multiselect("Vulnerabilities", VULNERABILITIES)
        
        st.session_state.form_data['shelter'] = {
            'floors': floors,
            'rooms': rooms,
            'material': material,
            'vulnerabilities': vulnerabilities
        }
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                handle_navigation("prev")
        with col2:
            if st.button("Next"):
                handle_navigation("next")
    
    # Step 3: People Information
    elif st.session_state.step == 3:
        st.subheader("Step 3: People Information")
        
        total_count = st.number_input("Total Number of People", min_value=1, step=1)
        special_circumstances = st.multiselect("Special Circumstances", SPECIAL_CIRCUMSTANCES)
        has_pets = st.checkbox("Are there pets?")
        
        pets_info = ""
        if has_pets:
            pets_info = st.text_input("Describe the pets (type and count)")
        
        st.session_state.form_data['people'] = {
            'count': total_count,
            'special_circumstances': special_circumstances,
            'has_pets': has_pets,
            'pets_info': pets_info
        }
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                handle_navigation("prev")
        with col2:
            if st.button("Next"):
                handle_navigation("next")
    
    # Step 4: Supplies Information
    elif st.session_state.step == 4:
        st.subheader("Step 4: Supplies Information")
        
        food_days = st.number_input("Food Supply in Days", min_value=1, step=1)
        water_liters = st.number_input("Water Supply in Liters", min_value=1, step=1)
        gear = st.multiselect("Available Gear", GEAR_LIST)
        
        st.session_state.form_data['supplies'] = {
            'food_days': food_days,
            'water_liters': water_liters,
            'gear': gear
        }
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                handle_navigation("prev")
        with col2:
            if st.button("Submit"):
                process_assessment()

if __name__ == "__main__":
    main()
