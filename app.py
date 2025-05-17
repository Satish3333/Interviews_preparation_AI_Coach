import os
import streamlit as st
from crewai import Agent, Task, Crew
from crewai.llm import LLM  
from crewai_tools import (
    FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool
)


# --- Sidebar Inputs ---
st.title("AI Job Coach ")
st.sidebar.header("🔐 API Keys")

azure_api_key = st.sidebar.text_input("Azure API Key", type="password")
azure_api_base = st.sidebar.text_input("Azure API Base")
azure_api_version = st.sidebar.text_input("Azure API Version")

if st.sidebar.button("🔧 Set API Keys"):
    os.environ['SERPER_API_KEY'] = "SERPER_API_KEY"
    os.environ['AZURE_API_KEY'] = "AZURE_API_KEY"
    os.environ['AZURE_API_BASE'] = "AZURE_API_VERSION"
    os.environ['AZURE_API_VERSION'] =  "AZURE_API_VERSION"
    st.success("API keys configured successfully!")

# --- User Inputs ---
st.subheader("🔗 Job Details")
job_posting_url = st.text_input("Job Posting URL")
github_url = st.text_input("GitHub Profile URL")
personal_writeup = st.text_area("Personal Summary")

uploaded_resume = st.file_uploader("📄 Upload Resume (Markdown format only)", type="md")

if st.button(" Run AI Job Coach"):
    if not all([job_posting_url, github_url, personal_writeup, uploaded_resume]):
        st.error("Please fill in all fields and upload a resume.")
    else:
        # Save uploaded resume
        resume_path = "./uploaded_resume.md"
        with open(resume_path, "wb") as f:
            f.write(uploaded_resume.read())

        # Setup LLM
        llm = LLM(model="azure/gpt-4o-mini", temperature=0.7)

        # Tools
        search_tool = SerperDevTool()
        scrape_tool = ScrapeWebsiteTool()
        read_resume = FileReadTool(file_path=resume_path)
        semantic_search_resume = MDXSearchTool(mdx=resume_path, config={
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "BAAI/bge-small-en-v1.5"},
            },
        })

        # Agents
        researcher = Agent(
            role="Tech Job Researcher",
            goal="Analyze job postings to help candidates",
            tools=[scrape_tool, search_tool],
            backstory="You extract skills and qualifications from job postings.",
            llm=llm, max_iter=5, allow_delegation=True, verbose=True
        )
        profiler = Agent(
            role="Personal Profiler",
            goal="Research candidates based on GitHub and summary",
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            backstory="You create a comprehensive candidate profile.",
            llm=llm, max_iter=5, allow_delegation=True, verbose=True
        )
        resume_strategist = Agent(
            role="Resume Strategist",
            goal="Tailor resume for job applications",
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            backstory="You refine resumes to match job postings.",
            llm=llm, max_iter=5, allow_delegation=True, verbose=True
        )
        interview_preparer = Agent(
            role="Interview Preparer",
            goal="Generate interview questions",
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            backstory="You generate questions to prepare candidates for interviews.",
            llm=llm, max_iter=5, allow_delegation=True, verbose=True
        )

        # Tasks
        research_task = Task(
            description=f"Analyze the job posting at {job_posting_url} to extract required skills and qualifications.",
            expected_output="Structured list of job requirements.",
            agent=researcher, async_execution=True
        )
        profile_task = Task(
            description=f"Compile candidate profile from GitHub ({github_url}) and summary.",
            expected_output="Detailed candidate profile.",
            agent=profiler, async_execution=True
        )
        resume_strategy_task = Task(
            description="Use job requirements and profile to rewrite resume.",
            expected_output="Tailored resume.",
            output_file="tailored_resume.md",
            context=[research_task, profile_task],
            agent=resume_strategist
        )
        interview_preparation_task = Task(
            description="Create interview questions from tailored resume and job requirements.",
            expected_output="Interview questions and talking points.",
            output_file="interview_materials.md",
            context=[research_task, profile_task, resume_strategy_task],
            agent=interview_preparer
        )

        # Run Crew
        crew = Crew(
            agents=[researcher, profiler, resume_strategist, interview_preparer],
            tasks=[research_task, profile_task, resume_strategy_task, interview_preparation_task],
            verbose=True
        )

        inputs = {
            'job_posting_url': job_posting_url,
            'github_url': github_url,
            'personal_writeup': personal_writeup
        }

        result = crew.kickoff(inputs=inputs)

        # Show Output
        st.success(" AI Job Coach Run Completed")

        # Display Results
        st.subheader(" Tailored Resume")
        with open("tailored_resume.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())

        st.subheader(" Interview Preparation")
        with open("interview_materials.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
