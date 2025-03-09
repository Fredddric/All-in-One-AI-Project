"""
Advanced AI modules for specialized tasks using the DeepSeek API
"""

class TextAnalysis:
    """Text analysis capabilities including sentiment analysis, 
    summarization, and entity extraction"""
    
    def __init__(self, deepseek_client):
        self.client = deepseek_client
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment of a text passage"""
        prompt = f"""Analyze the sentiment of the following text. 
        Categorize it as positive, negative, or neutral, and provide a 
        confidence score (0-100%). Also highlight key phrases that influenced 
        your decision.
        
        Text: {text}
        
        Format your response as:
        Sentiment: [sentiment]
        Confidence: [score]%
        Key phrases: [phrases]
        Explanation: [brief explanation]
        """
        return self.client.generate_text(prompt, max_tokens=200, temperature=0.3)
    
    def summarize_text(self, text, max_length=100):
        """Generate a concise summary of longer text"""
        prompt = f"""Summarize the following text in no more than {max_length} words 
        while preserving the key points and main message:
        
        {text}
        """
        return self.client.generate_text(prompt, max_tokens=300, temperature=0.4)
    
    def extract_entities(self, text):
        """Extract named entities from text (people, organizations, locations, etc.)"""
        prompt = f"""Extract all named entities from the following text. 
        Categorize each entity as PERSON, ORGANIZATION, LOCATION, DATE, or OTHER.
        Format the output as a structured list:
        
        Text: {text}
        """
        return self.client.generate_text(prompt, max_tokens=500, temperature=0.3)


class CodeAssistant:
    """Code generation and analysis capabilities"""
    
    def __init__(self, deepseek_client):
        self.client = deepseek_client
    
    def generate_code(self, language, task_description):
        """Generate code in the specified programming language based on description"""
        prompt = f"""Generate {language} code for the following task. 
        Include comments to explain the solution:
        
        Task: {task_description}
        
        Only output the code without any additional explanation.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.2)
    
    def explain_code(self, code, language):
        """Explain what a piece of code does"""
        prompt = f"""Explain the following {language} code in clear, concise terms. 
        Break down the explanation by sections and highlight any important patterns, 
        algorithms, or potential issues:
        
        ```{language}
        {code}
        ```
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.4)
    
    def optimize_code(self, code, language):
        """Suggest optimizations for a given code snippet"""
        prompt = f"""Optimize the following {language} code for better performance, 
        readability, and maintainability. Explain the improvements you've made:
        
        ```{language}
        {code}
        ```
        
        Return the optimized code followed by an explanation of the changes.
        """
        return self.client.generate_text(prompt, max_tokens=1000, temperature=0.3)


class AcademicResearch:
    """AI assistance for academic research tasks"""
    
    def __init__(self, deepseek_client):
        self.client = deepseek_client
    
    def literature_review(self, topic, num_papers=5):
        """Generate a mock literature review on a topic"""
        prompt = f"""Generate a brief literature review for a computer science research 
        paper on the topic of "{topic}". Include references to {num_papers} important 
        papers in this field (provide fictional citations in IEEE format). For each paper, 
        mention:
        - The main contribution
        - Key findings
        - How it relates to the topic
        
        Format the review as a proper academic section.
        """
        return self.client.generate_text(prompt, max_tokens=1000, temperature=0.6)
    
    def research_question_generator(self, topic, field="computer science"):
        """Generate potential research questions on a topic"""
        prompt = f"""Suggest 5 specific, focused research questions related to "{topic}" 
        in the field of {field}. For each question:
        
        1. Formulate the question clearly
        2. Explain why this question is significant
        3. Suggest a possible methodology to investigate this question
        4. Identify potential challenges in answering this question
        
        Format each question and its details separately.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.7)
    
    def methodology_design(self, research_question):
        """Generate a research methodology for a given research question"""
        prompt = f"""Design a detailed research methodology to address the following 
        research question in computer science:
        
        "{research_question}"
        
        Include:
        - Research approach (qualitative, quantitative, mixed)
        - Data collection methods
        - Analysis techniques
        - Ethical considerations
        - Limitations of the methodology
        
        Format this as a methodology section for an academic paper.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.5)


class DataAnalysis:
    """AI assistance for data analysis tasks"""
    
    def __init__(self, deepseek_client):
        self.client = deepseek_client
    
    def generate_analysis_code(self, data_description, analysis_goal, language="python"):
        """Generate code for data analysis"""
        prompt = f"""Create {language} code to perform data analysis with the following details:
        
        Data description: {data_description}
        Analysis goal: {analysis_goal}
        
        Include code for:
        1. Data loading and preprocessing
        2. Exploratory data analysis
        3. Main analysis to achieve the goal
        4. Visualization of results
        5. Interpretation of findings
        
        Provide comments throughout the code to explain each step.
        """
        return self.client.generate_text(prompt, max_tokens=1000, temperature=0.3)
    
    def interpret_results(self, results_description):
        """Generate an interpretation of analysis results"""
        prompt = f"""Provide a detailed interpretation of the following data analysis results:
        
        {results_description}
        
        Include:
        - Summary of key findings
        - Possible implications
        - Limitations of the analysis
        - Suggestions for further analysis
        
        Format this as a results interpretation section for a technical report.
        """
        return self.client.generate_text(prompt, max_tokens=600, temperature=0.4)


class MachineLearning:
    """Machine learning capabilities using the DeepSeek API"""
    
    def __init__(self, deepseek_client):
        self.client = deepseek_client
    
    def model_selection(self, problem_description, data_description):
        """Recommend suitable machine learning models for a given problem"""
        prompt = f"""Given the following problem and data description, recommend the 
        most appropriate machine learning models. For each model:
        
        1. Explain why it's suitable for this problem
        2. List its advantages and limitations
        3. Provide implementation considerations
        4. Suggest evaluation metrics
        
        Problem description: {problem_description}
        Data description: {data_description}
        
        Return a structured analysis of at least 3 recommended models.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.4)
    
    def feature_engineering(self, data_description, target_variable):
        """Generate feature engineering suggestions"""
        prompt = f"""Suggest effective feature engineering techniques for the following 
        machine learning scenario:
        
        Data description: {data_description}
        Target variable: {target_variable}
        
        For each suggestion:
        1. Describe the technique in detail
        2. Explain why it would be beneficial
        3. Provide pseudo-code for implementation
        4. Mention potential pitfalls to avoid
        
        Focus on techniques that would improve model performance for this specific problem.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.4)
    
    def hyperparameter_tuning(self, model_type, dataset_characteristics):
        """Generate hyperparameter tuning strategy"""
        prompt = f"""Develop a comprehensive hyperparameter tuning strategy for a {model_type} 
        model with the following dataset characteristics:
        
        {dataset_characteristics}
        
        Include:
        1. Key hyperparameters to tune
        2. Recommended search ranges for each parameter
        3. Suggested tuning approach (grid search, random search, Bayesian optimization, etc.)
        4. Python code template for implementing the tuning process
        5. Best practices for avoiding overfitting during tuning
        
        Format the response as a detailed tuning plan.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.3)
    
    def model_evaluation(self, model_type, evaluation_results):
        """Analyze model evaluation results"""
        prompt = f"""Provide a detailed analysis of the following evaluation results for a {model_type} model:
        
        {evaluation_results}
        
        Include:
        1. Interpretation of each metric
        2. Assessment of model performance
        3. Potential issues identified (overfitting, underfitting, bias, etc.)
        4. Recommendations for model improvement
        5. Comparison to typical performance benchmarks
        
        Format your response as a professional model evaluation report.
        """
        return self.client.generate_text(prompt, max_tokens=700, temperature=0.3)
    
    def generate_ml_pipeline(self, task_description, data_description):
        """Generate a complete machine learning pipeline code"""
        prompt = f"""Create a complete Python machine learning pipeline for the following task and data:
        
        Task: {task_description}
        Data: {data_description}
        
        The pipeline should include:
        1. Data preprocessing (handling missing values, encoding, scaling, etc.)
        2. Feature selection/engineering
        3. Model selection and training
        4. Hyperparameter tuning
        5. Evaluation
        6. Prediction functionality
        
        Use scikit-learn, pandas, and other common ML libraries. Include detailed comments.
        """
        return self.client.generate_text(prompt, max_tokens=1200, temperature=0.3)
    
    def explain_ml_concept(self, concept):
        """Explain a machine learning concept in detail"""
        prompt = f"""Explain the machine learning concept of "{concept}" in detail. Include:
        
        1. Clear definition and explanation
        2. Mathematical formulation where relevant
        3. Real-world applications
        4. Advantages and limitations
        5. Related concepts
        6. Simple example to illustrate the concept
        
        Make the explanation suitable for a computer science student with basic ML knowledge.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.4)


class DataVisualization:
    """Data visualization capabilities using the DeepSeek API"""
    
    def __init__(self, deepseek_client):
        self.client = deepseek_client
    
    def recommend_visualization(self, data_description, analysis_goal):
        """Recommend appropriate visualization techniques for specific data and goals"""
        prompt = f"""Based on the following data description and analysis goal, recommend 
        the most appropriate data visualization techniques. For each recommended visualization:
        
        1. Explain why it's suitable for this specific data and goal
        2. Describe how to interpret the visualization
        3. Mention key design considerations for clarity and effectiveness
        4. Suggest appropriate color schemes and layout options
        5. Identify potential limitations or misinterpretations to avoid
        
        Data description: {data_description}
        Analysis goal: {analysis_goal}
        
        Return a structured analysis of at least 3 recommended visualization approaches.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.4)
    
    def generate_visualization_code(self, data_description, visualization_type, language="python"):
        """Generate code for creating data visualizations"""
        prompt = f"""Generate {language} code to create a {visualization_type} visualization 
        for the following data:
        
        Data description: {data_description}
        
        The code should:
        1. Include sample data creation that matches the description (if real data isn't provided)
        2. Perform any necessary data preparation and transformation
        3. Create a professional-quality {visualization_type} using appropriate libraries
        4. Include customizations for colors, labels, titles, and annotations
        5. Add brief comments explaining key parts of the code
        
        Use popular visualization libraries appropriate for {language} (e.g., Matplotlib, Seaborn, 
        Plotly for Python).
        """
        return self.client.generate_text(prompt, max_tokens=1000, temperature=0.3)
    
    def visualization_storytelling(self, data_insights, target_audience):
        """Create a data storytelling narrative around visualizations"""
        prompt = f"""Craft a data storytelling narrative that effectively communicates 
        the following insights to the specified audience. Include recommendations for a 
        sequence of visualizations that would best tell this data story:
        
        Data insights: {data_insights}
        Target audience: {target_audience}
        
        Your response should include:
        1. An engaging introduction that sets context and highlights importance
        2. A logical flow that builds the narrative using data points
        3. Specific visualization recommendations for each part of the story
        4. Key messages to emphasize in each visualization
        5. A conclusion with actionable insights
        
        Format this as a comprehensive data storytelling plan.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.5)
    
    def interactive_dashboard_design(self, data_sources, user_requirements):
        """Generate a design plan for an interactive data dashboard"""
        prompt = f"""Design a comprehensive interactive dashboard based on the following 
        data sources and user requirements:
        
        Data sources: {data_sources}
        User requirements: {user_requirements}
        
        Your design plan should include:
        1. Overall dashboard layout and organization
        2. Key visualizations to include (with justification)
        3. Interactive elements and filtering capabilities
        4. Color scheme and design principles
        5. User journey and interaction flow
        6. Performance considerations
        
        Include a mockup description of the dashboard and explain how each component 
        fulfills the user requirements.
        """
        return self.client.generate_text(prompt, max_tokens=900, temperature=0.4)
    
    def critique_visualization(self, visualization_description):
        """Provide expert critique of a data visualization"""
        prompt = f"""Provide an expert critique of the following data visualization:
        
        {visualization_description}
        
        Your critique should address:
        1. Effectiveness in communicating the intended message
        2. Appropriate use of visualization type for the data
        3. Visual design elements (color, layout, labeling, etc.)
        4. Potential for misinterpretation or bias
        5. Accessibility considerations
        6. Specific recommendations for improvement
        
        Balance constructive criticism with recognition of effective elements.
        """
        return self.client.generate_text(prompt, max_tokens=700, temperature=0.4)


class NaturalLanguageProcessing:
    """Advanced NLP capabilities using the DeepSeek API"""
    
    def __init__(self, deepseek_client):
        self.client = deepseek_client
    
    def semantic_search(self, corpus, query):
        """Perform semantic search on a corpus of text"""
        prompt = f"""Perform a semantic search for the following query within the provided text corpus.
        Rank the most relevant passages and explain why each is relevant to the query.
        
        Query: {query}
        
        Corpus:
        {corpus}
        
        Return:
        1. The top 3-5 most semantically relevant passages
        2. For each passage, provide a relevance score (0-100%)
        3. A brief explanation of why each passage is relevant to the query
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.3)
    
    def language_translation(self, text, source_language, target_language):
        """Translate text between languages"""
        prompt = f"""Translate the following text from {source_language} to {target_language}.
        Ensure the translation preserves the original meaning, tone, and cultural nuances as much as possible.
        
        Text to translate ({source_language}):
        {text}
        
        Please provide:
        1. The translation in {target_language}
        2. Notes on any cultural adaptations made or idioms that needed special handling
        """
        return self.client.generate_text(prompt, max_tokens=1000, temperature=0.4)
    
    def named_entity_recognition(self, text):
        """Identify and classify named entities in text"""
        prompt = f"""Perform Named Entity Recognition (NER) on the following text.
        Identify all named entities and classify them into appropriate categories such as:
        - Person
        - Organization
        - Location
        - Date/Time
        - Quantity
        - Event
        - Product
        - Other (specify)
        
        Text:
        {text}
        
        For each entity identified, provide:
        1. The entity text
        2. Its category
        3. A brief note on its role or significance in the text (if applicable)
        
        Format the results in a structured, easy-to-read format.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.3)
    
    def topic_modeling(self, documents):
        """Extract main topics from a collection of documents"""
        prompt = f"""Perform topic modeling on the following collection of documents.
        Identify the key topics discussed across these documents and provide a summary of each topic.
        
        Documents:
        {documents}
        
        For each identified topic:
        1. Provide a descriptive name
        2. List the key terms/phrases associated with this topic
        3. Summarize what this topic represents
        4. Indicate which documents most strongly relate to this topic
        
        Aim to identify 3-7 distinct topics, depending on the diversity of the content.
        """
        return self.client.generate_text(prompt, max_tokens=1000, temperature=0.5)
    
    def text_classification(self, text, categories):
        """Classify text into predefined categories"""
        categories_str = ", ".join(categories)
        prompt = f"""Classify the following text into one or more of these categories: {categories_str}
        
        Text to classify:
        {text}
        
        For each relevant category:
        1. Indicate a confidence score (0-100%)
        2. Provide a brief explanation for why this category applies
        3. Highlight key phrases that influenced the classification
        
        If multiple categories apply, rank them by relevance.
        """
        return self.client.generate_text(prompt, max_tokens=600, temperature=0.3)
    
    def sentiment_analysis_advanced(self, text):
        """Perform advanced sentiment analysis with emotion detection"""
        prompt = f"""Perform an advanced sentiment analysis on the following text.
        Go beyond basic positive/negative classification to identify specific emotions,
        intensity, and potential underlying sentiments.
        
        Text:
        {text}
        
        Please provide:
        1. Overall sentiment (positive, negative, neutral, mixed)
        2. Sentiment intensity score (-100 to +100)
        3. Primary emotions detected (joy, sadness, anger, fear, surprise, etc.)
        4. Any detected sarcasm, irony, or implicit sentiments
        5. Key phrases that influenced this analysis
        6. Confidence level in this analysis (0-100%)
        
        Support your analysis with specific evidence from the text.
        """
        return self.client.generate_text(prompt, max_tokens=800, temperature=0.4)
