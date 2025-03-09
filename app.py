import os
import streamlit as st
import requests
from dotenv import load_dotenv
from PIL import Image
import io
from ai_modules import TextAnalysis, CodeAssistant, AcademicResearch, DataAnalysis, MachineLearning, DataVisualization, NaturalLanguageProcessing

# Load environment variables
load_dotenv()

# Get API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DeepSeek API key not found. Please check your .env file.")

class DeepSeekAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, prompt, max_tokens=500, temperature=0.7):
        """
        Generate text using DeepSeek's API
        """
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(endpoint, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
            return None
    
    def analyze_image(self, image_data, prompt):
        """
        Analyze an image using DeepSeek's API
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        # Prepare the image for multimodal analysis
        image_base64 = image_data.decode('utf-8') if isinstance(image_data, bytes) else image_data
        
        payload = {
            "model": "deepseek-vision",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        response = requests.post(endpoint, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
            return None

def display_api_response(response):
    """Helper function to display API responses"""
    if response:
        try:
            result = response['choices'][0]['message']['content']
            st.markdown(result)
        except KeyError:
            st.error("Unexpected response format from the API.")

def main():
    st.set_page_config(page_title="DeepSeek AI Project", layout="wide")
    
    st.title("All-in-One AI Assistant")
    st.subheader("Final Year Computer Science Project")
    
    # Initialize DeepSeek API client
    client = DeepSeekAPI(DEEPSEEK_API_KEY)
    
    # Initialize specialized modules
    text_analysis = TextAnalysis(client)
    code_assistant = CodeAssistant(client)
    academic_research = AcademicResearch(client)
    data_analysis = DataAnalysis(client)
    machine_learning = MachineLearning(client)
    data_visualization = DataVisualization(client)
    nlp = NaturalLanguageProcessing(client)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Text Generation", 
        "Image Analysis", 
        "Text Analysis", 
        "Code Assistant",
        "Academic Research",
        "Machine Learning",
        "Data Visualization",
        "Advanced NLP"
    ])
    
    with tab1:
        st.header("AI Text Generation")
        
        prompt = st.text_area("Enter your prompt:", height=150)
        col1, col2 = st.columns(2)
        
        with col1:
            max_tokens = st.slider("Maximum tokens:", 100, 2000, 500)
        
        with col2:
            temperature = st.slider("Temperature (creativity):", 0.1, 1.0, 0.7)
        
        if st.button("Generate Text"):
            if prompt:
                with st.spinner("Generating text..."):
                    response = client.generate_text(prompt, max_tokens, temperature)
                    st.markdown("### Generated Text:")
                    display_api_response(response)
            else:
                st.warning("Please enter a prompt.")
    
    with tab2:
        st.header("AI Image Analysis")
        
        uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
        analysis_prompt = st.text_input("What do you want to know about this image?", 
                                       "Describe this image in detail and identify the main elements.")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    # Convert image to base64
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG")
                    image_bytes = buffer.getvalue()
                    import base64
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    response = client.analyze_image(image_base64, analysis_prompt)
                    st.markdown("### Analysis Result:")
                    display_api_response(response)
    
    with tab3:
        st.header("Text Analysis Tools")
        
        text_analysis_type = st.selectbox(
            "Select analysis type:",
            ["Sentiment Analysis", "Text Summarization", "Entity Extraction"]
        )
        
        text_to_analyze = st.text_area("Enter text to analyze:", height=150)
        
        if st.button("Analyze Text"):
            if text_to_analyze:
                with st.spinner("Analyzing text..."):
                    if text_analysis_type == "Sentiment Analysis":
                        response = text_analysis.analyze_sentiment(text_to_analyze)
                        st.markdown("### Sentiment Analysis Result:")
                        display_api_response(response)
                    
                    elif text_analysis_type == "Text Summarization":
                        max_length = st.slider("Summary length (words):", 50, 300, 100)
                        response = text_analysis.summarize_text(text_to_analyze, max_length)
                        st.markdown("### Text Summary:")
                        display_api_response(response)
                    
                    elif text_analysis_type == "Entity Extraction":
                        response = text_analysis.extract_entities(text_to_analyze)
                        st.markdown("### Extracted Entities:")
                        display_api_response(response)
            else:
                st.warning("Please enter text to analyze.")
    
    with tab4:
        st.header("Code Assistant")
        
        code_task = st.selectbox(
            "Select task:",
            ["Generate Code", "Explain Code", "Optimize Code"]
        )
        
        if code_task == "Generate Code":
            language = st.selectbox(
                "Programming language:",
                ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "SQL", "PHP"]
            )
            task_description = st.text_area("Describe what you want the code to do:", height=150)
            
            if st.button("Generate Code"):
                if task_description:
                    with st.spinner("Generating code..."):
                        response = code_assistant.generate_code(language, task_description)
                        st.markdown(f"### Generated {language} Code:")
                        display_api_response(response)
                else:
                    st.warning("Please describe the task.")
        
        elif code_task in ["Explain Code", "Optimize Code"]:
            language = st.selectbox(
                "Programming language:",
                ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "SQL", "PHP"]
            )
            code = st.text_area("Enter code:", height=300)
            
            if st.button("Process Code"):
                if code:
                    with st.spinner("Processing code..."):
                        if code_task == "Explain Code":
                            response = code_assistant.explain_code(code, language)
                            st.markdown("### Code Explanation:")
                            display_api_response(response)
                        else:  # Optimize Code
                            response = code_assistant.optimize_code(code, language)
                            st.markdown("### Optimized Code:")
                            display_api_response(response)
                else:
                    st.warning("Please enter code to process.")
    
    with tab5:
        st.header("Academic Research Assistant")
        
        research_task = st.selectbox(
            "Select task:",
            ["Literature Review", "Research Question Generator", "Methodology Design"]
        )
        
        if research_task == "Literature Review":
            topic = st.text_input("Research topic:")
            num_papers = st.slider("Number of papers to include:", 3, 10, 5)
            
            if st.button("Generate Literature Review"):
                if topic:
                    with st.spinner("Generating literature review..."):
                        response = academic_research.literature_review(topic, num_papers)
                        st.markdown("### Literature Review:")
                        display_api_response(response)
                else:
                    st.warning("Please enter a research topic.")
        
        elif research_task == "Research Question Generator":
            topic = st.text_input("Research area:")
            field = st.text_input("Specific field:", "computer science")
            
            if st.button("Generate Research Questions"):
                if topic:
                    with st.spinner("Generating research questions..."):
                        response = academic_research.research_question_generator(topic, field)
                        st.markdown("### Potential Research Questions:")
                        display_api_response(response)
                else:
                    st.warning("Please enter a research area.")
        
        elif research_task == "Methodology Design":
            research_question = st.text_area("Enter your research question:", height=100)
            
            if st.button("Design Methodology"):
                if research_question:
                    with st.spinner("Designing methodology..."):
                        response = academic_research.methodology_design(research_question)
                        st.markdown("### Research Methodology:")
                        display_api_response(response)
                else:
                    st.warning("Please enter a research question.")

    with tab6:
        st.header("Machine Learning Assistant")
        
        ml_task = st.selectbox(
            "Select task:",
            ["Model Selection", "Feature Engineering", "Hyperparameter Tuning", 
             "Model Evaluation", "ML Pipeline Generation", "Explain ML Concept"]
        )
        
        if ml_task == "Model Selection":
            problem_description = st.text_area("Describe your machine learning problem:", 
                                               height=100,
                                               placeholder="E.g., Predict customer churn based on usage patterns and demographic data")
            data_description = st.text_area("Describe your dataset:", 
                                            height=100,
                                            placeholder="E.g., 10,000 records with 20 features including categorical and numerical data, 5% missing values")
            
            if st.button("Recommend Models"):
                if problem_description and data_description:
                    with st.spinner("Analyzing problem and recommending models..."):
                        response = machine_learning.model_selection(problem_description, data_description)
                        st.markdown("### Recommended Machine Learning Models:")
                        display_api_response(response)
                else:
                    st.warning("Please describe both your problem and dataset.")
                    
        elif ml_task == "Feature Engineering":
            data_description = st.text_area("Describe your dataset:", 
                                            height=100,
                                            placeholder="E.g., Customer transaction data with timestamp, amount, location, etc.")
            target_variable = st.text_input("What are you trying to predict?",
                                           placeholder="E.g., Fraudulent transactions")
            
            if st.button("Suggest Feature Engineering Techniques"):
                if data_description and target_variable:
                    with st.spinner("Generating feature engineering suggestions..."):
                        response = machine_learning.feature_engineering(data_description, target_variable)
                        st.markdown("### Feature Engineering Suggestions:")
                        display_api_response(response)
                else:
                    st.warning("Please describe both your dataset and target variable.")
                    
        elif ml_task == "Hyperparameter Tuning":
            model_type = st.text_input("Model type:", placeholder="E.g., Random Forest, XGBoost, Neural Network")
            dataset_characteristics = st.text_area("Dataset characteristics:", 
                                                  height=100,
                                                  placeholder="E.g., 5000 samples, imbalanced classes, high dimensionality")
            
            if st.button("Generate Tuning Strategy"):
                if model_type and dataset_characteristics:
                    with st.spinner("Developing hyperparameter tuning strategy..."):
                        response = machine_learning.hyperparameter_tuning(model_type, dataset_characteristics)
                        st.markdown("### Hyperparameter Tuning Strategy:")
                        display_api_response(response)
                else:
                    st.warning("Please provide both model type and dataset characteristics.")
                    
        elif ml_task == "Model Evaluation":
            model_type = st.text_input("Model type:", placeholder="E.g., Logistic Regression, Random Forest")
            evaluation_results = st.text_area("Evaluation metrics and results:", 
                                             height=150,
                                             placeholder="E.g., Accuracy: 0.85, Precision: 0.78, Recall: 0.92, AUC: 0.88, F1: 0.84")
            
            if st.button("Analyze Results"):
                if model_type and evaluation_results:
                    with st.spinner("Analyzing evaluation results..."):
                        response = machine_learning.model_evaluation(model_type, evaluation_results)
                        st.markdown("### Model Evaluation Analysis:")
                        display_api_response(response)
                else:
                    st.warning("Please provide both model type and evaluation results.")
                    
        elif ml_task == "ML Pipeline Generation":
            task_description = st.text_area("Describe your ML task:", 
                                           height=100,
                                           placeholder="E.g., Build a classification model to identify customer segments")
            data_description = st.text_area("Describe your dataset:", 
                                           height=100,
                                           placeholder="E.g., CSV file with 15 columns including categorical and numerical features")
            
            if st.button("Generate ML Pipeline"):
                if task_description and data_description:
                    with st.spinner("Generating complete ML pipeline..."):
                        response = machine_learning.generate_ml_pipeline(task_description, data_description)
                        st.markdown("### Machine Learning Pipeline Code:")
                        display_api_response(response)
                else:
                    st.warning("Please describe both your ML task and dataset.")
                    
        elif ml_task == "Explain ML Concept":
            concept = st.text_input("Enter a machine learning concept:", 
                                   placeholder="E.g., Random Forest, Gradient Descent, Cross-validation")
            
            if st.button("Explain Concept"):
                if concept:
                    with st.spinner("Generating explanation..."):
                        response = machine_learning.explain_ml_concept(concept)
                        st.markdown(f"### Explanation of {concept}:")
                        display_api_response(response)
                else:
                    st.warning("Please enter a machine learning concept to explain.")

    with tab7:
        st.header("Data Visualization Assistant")
        
        viz_task = st.selectbox(
            "Select task:",
            ["Recommend Visualizations", "Generate Visualization Code", 
             "Visualization Storytelling", "Dashboard Design", "Critique Visualization"]
        )
        
        if viz_task == "Recommend Visualizations":
            data_description = st.text_area("Describe your data:", 
                                           height=100,
                                           placeholder="E.g., Time series sales data across 5 product categories over 2 years with seasonal patterns")
            analysis_goal = st.text_area("What insights are you trying to uncover?", 
                                        height=100,
                                        placeholder="E.g., Understand seasonal trends and identify top-performing product categories")
            
            if st.button("Recommend Visualizations"):
                if data_description and analysis_goal:
                    with st.spinner("Analyzing data and recommending visualizations..."):
                        response = data_visualization.recommend_visualization(data_description, analysis_goal)
                        st.markdown("### Recommended Visualization Approaches:")
                        display_api_response(response)
                else:
                    st.warning("Please describe both your data and analysis goals.")
                    
        elif viz_task == "Generate Visualization Code":
            data_description = st.text_area("Describe your data:", 
                                           height=100,
                                           placeholder="E.g., Customer demographic data including age, income, location, and purchase history")
            visualization_type = st.text_input("What type of visualization do you need?",
                                              placeholder="E.g., Scatter plot, heatmap, choropleth map, interactive dashboard")
            language = st.selectbox("Programming language:", ["Python", "R", "JavaScript"])
            
            if st.button("Generate Code"):
                if data_description and visualization_type:
                    with st.spinner("Generating visualization code..."):
                        response = data_visualization.generate_visualization_code(data_description, visualization_type, language)
                        st.markdown(f"### {visualization_type} Code ({language}):")
                        display_api_response(response)
                else:
                    st.warning("Please describe your data and specify a visualization type.")
                    
        elif viz_task == "Visualization Storytelling":
            data_insights = st.text_area("What insights have you gathered from your data?", 
                                        height=100,
                                        placeholder="E.g., Customer retention has decreased by 15% in the last quarter, with highest churn among 18-25 year olds")
            target_audience = st.text_input("Who is your target audience?",
                                           placeholder="E.g., Executive team, marketing department, technical team, general public")
            
            if st.button("Create Data Story"):
                if data_insights and target_audience:
                    with st.spinner("Crafting data storytelling narrative..."):
                        response = data_visualization.visualization_storytelling(data_insights, target_audience)
                        st.markdown("### Data Storytelling Plan:")
                        display_api_response(response)
                else:
                    st.warning("Please describe your data insights and specify your target audience.")
                    
        elif viz_task == "Dashboard Design":
            data_sources = st.text_area("What data sources will your dashboard use?", 
                                       height=100,
                                       placeholder="E.g., Sales database, website analytics, CRM system, social media metrics")
            user_requirements = st.text_area("What are the requirements for this dashboard?", 
                                           height=100,
                                           placeholder="E.g., Track KPIs, allow filtering by date range and region, show comparisons to previous periods")
            
            if st.button("Design Dashboard"):
                if data_sources and user_requirements:
                    with st.spinner("Creating dashboard design..."):
                        response = data_visualization.interactive_dashboard_design(data_sources, user_requirements)
                        st.markdown("### Interactive Dashboard Design Plan:")
                        display_api_response(response)
                else:
                    st.warning("Please describe your data sources and dashboard requirements.")
                    
        elif viz_task == "Critique Visualization":
            visualization_description = st.text_area("Describe the visualization you want critiqued:", 
                                                   height=150,
                                                   placeholder="Please describe in detail the visualization, including chart type, data represented, colors used, labels, etc.")
            
            if st.button("Get Expert Critique"):
                if visualization_description:
                    with st.spinner("Analyzing visualization..."):
                        response = data_visualization.critique_visualization(visualization_description)
                        st.markdown("### Visualization Critique:")
                        display_api_response(response)
                else:
                    st.warning("Please describe the visualization you want to have critiqued.")

    with tab8:
        st.header("Advanced NLP Assistant")
        
        nlp_task = st.selectbox(
            "Select task:",
            ["Semantic Search", "Language Translation", "Named Entity Recognition", 
             "Topic Modeling", "Text Classification", "Advanced Sentiment Analysis"]
        )
        
        if nlp_task == "Semantic Search":
            corpus = st.text_area("Enter the text corpus to search within:", 
                                 height=200,
                                 placeholder="Paste the collection of text documents you want to search within...")
            query = st.text_input("Enter your search query:",
                                 placeholder="What information are you looking for?")
            
            if st.button("Perform Semantic Search"):
                if corpus and query:
                    with st.spinner("Performing semantic search..."):
                        response = nlp.semantic_search(corpus, query)
                        st.markdown("### Search Results:")
                        display_api_response(response)
                else:
                    st.warning("Please provide both a text corpus and a search query.")
                    
        elif nlp_task == "Language Translation":
            text_to_translate = st.text_area("Enter text to translate:", 
                                            height=150,
                                            placeholder="Enter the text you want to translate...")
            col1, col2 = st.columns(2)
            with col1:
                source_language = st.text_input("Source language:",
                                              placeholder="e.g., English, Spanish, French")
            with col2:
                target_language = st.text_input("Target language:",
                                              placeholder="e.g., German, Japanese, Russian")
            
            if st.button("Translate Text"):
                if text_to_translate and source_language and target_language:
                    with st.spinner(f"Translating from {source_language} to {target_language}..."):
                        response = nlp.language_translation(text_to_translate, source_language, target_language)
                        st.markdown(f"### Translation ({source_language} → {target_language}):")
                        display_api_response(response)
                else:
                    st.warning("Please provide text to translate, source language, and target language.")
                    
        elif nlp_task == "Named Entity Recognition":
            text_for_ner = st.text_area("Enter text for entity recognition:", 
                                       height=200,
                                       placeholder="Enter text containing named entities (people, organizations, locations, etc.)...")
            
            if st.button("Extract Entities"):
                if text_for_ner:
                    with st.spinner("Identifying named entities..."):
                        response = nlp.named_entity_recognition(text_for_ner)
                        st.markdown("### Recognized Entities:")
                        display_api_response(response)
                else:
                    st.warning("Please provide text for entity recognition.")
                    
        elif nlp_task == "Topic Modeling":
            documents = st.text_area("Enter documents for topic modeling:", 
                                    height=250,
                                    placeholder="Enter multiple documents or text sections separated by line breaks or document indicators...")
            
            if st.button("Extract Topics"):
                if documents:
                    with st.spinner("Performing topic modeling..."):
                        response = nlp.topic_modeling(documents)
                        st.markdown("### Identified Topics:")
                        display_api_response(response)
                else:
                    st.warning("Please provide documents for topic modeling.")
                    
        elif nlp_task == "Text Classification":
            text_to_classify = st.text_area("Enter text to classify:", 
                                           height=150,
                                           placeholder="Enter the text you want to classify...")
            
            categories_input = st.text_input("Enter categories (comma-separated):",
                                           placeholder="e.g., Sports, Politics, Technology, Entertainment, Science")
            
            if st.button("Classify Text"):
                if text_to_classify and categories_input:
                    categories = [cat.strip() for cat in categories_input.split(",")]
                    with st.spinner("Classifying text..."):
                        response = nlp.text_classification(text_to_classify, categories)
                        st.markdown("### Classification Results:")
                        display_api_response(response)
                else:
                    st.warning("Please provide both text to classify and categories.")
                    
        elif nlp_task == "Advanced Sentiment Analysis":
            text_for_sentiment = st.text_area("Enter text for sentiment analysis:", 
                                            height=150,
                                            placeholder="Enter text to analyze for sentiment and emotions...")
            
            if st.button("Analyze Sentiment"):
                if text_for_sentiment:
                    with st.spinner("Performing advanced sentiment analysis..."):
                        response = nlp.sentiment_analysis_advanced(text_for_sentiment)
                        st.markdown("### Sentiment Analysis Results:")
                        display_api_response(response)
                else:
                    st.warning("Please provide text for sentiment analysis.")

if __name__ == "__main__":
    main()
