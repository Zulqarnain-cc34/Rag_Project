import re
from groq import Groq

client = Groq()  # api key is loaded in at GROQ_API_KEY by default

system_instructions = """
# System Instructions for Proposition Extraction from Resumes (Simplified for Smaller Models)

**Introduction:**  
You are tasked with extracting facts from a resume and turning them into propositions. Each proposition should be a clear, standalone statement based directly on the resume. You will also create variations of these propositions to help match potential job requirements. Follow these instructions carefully.

**1. Fact Extraction:**  
- **Separate Propositions:** Each distinct fact from the resume must be its own proposition.  
- **Variations:** For each fact, YOU MUST CREATE 3 different ways to say the same thing. These variations should be based directly on the resume and not add new information.  
  - **How to create variations:** Try rephrasing the proposition using synonyms or different sentence structures, but keep the meaning the same.  
  - **Example 1:** Resume says "Candidate is a Tech Leader." Variations: "Candidate leads development teams." or "Candidate has leadership experience in technology."  
  - **Example 2:** Resume says "Candidate worked on AI projects." Variations: "Candidate has experience with AI." or "Candidate contributed to AI initiatives."  
  - Think about the greater context too, if they talked about GenAI and OpenAI and say Open Source Models, you can fairly assume they are talking about Large Language Models even if they didn't mention it explicitly.
- **Important:** Do not make up details. Stick to what the resume says.

**2. Full Context:**  
- Each proposition must include all relevant details from the resume, like dates, company names, or specific skills.  
- **Example:** Instead of "Candidate worked at a company," say "Candidate worked at XYZ from 2018 to 2020 as a Software Engineer."

**3. Lists of Skills or Technologies:**  
- If the resume lists multiple skills or technologies together, make each one a separate proposition.  
- **Example:** For "CI/CD (GitHub Actions, Gitlab Pipeline, Azure DevOps)," create:  
  - Candidate knows CI/CD.  
  - Candidate has experience with CI/CD.
  - Candidate has worked with CI/CD tools.
  - Candidate knows GitHub Actions.  
  - Candidate has experience with GitHub Actions.
  - Candidate has worked with GitHub Actions.
  - Candidate knows Gitlab Pipeline.  
  - Candidate has experience with Gitlab Pipeline.
  - Candidate has worked with Gitlab Pipeline.
  - Candidate knows Azure DevOps.
  - Candidate has experience with Azure DevOps.
  - Candidate has worked with Azure DevOps.

**4. Candidate Name:**  
- Replace the candidate's name with "Candidate" in all propositions.  
- **Example:** "Matt Tech Leadership" becomes "Candidate is a technical leader." and additional variations like "Candidate has Tech Leadership skills." and "Candidate has lead development teams."

**5. PII (Personal Information):**  
- **Do not include:** full name, phone number, email address, or links to personal websites.  
- **Allowed:** school names, degrees, and previous job experiences.

**6. Output Format:**  
- Put all propositions inside a single `<propositions>` XML tag.  
- Start each proposition with a bullet (`-`).  
- **Example:**  
  <propositions>  
  - Candidate knows React.
  - Candidate knows TypeScript.  
  - Candidate worked at XYZ from 2018 to 2020.  
  </propositions>

**Additional Tips:**  
- Read the resume carefully and extract all key facts.  
- For each fact, think of one or two ways to rephrase it without changing the meaning.  
- Make sure each proposition is complete and includes all necessary details.  
- Double-check that you haven't included any personal information like name or contact details.  
- Ensure the output is correctly formatted with XML tags and bullets.
"""

candidate_resume = """
# Matthew Groff

Orlando, FL | [mattlgroff@gmail.com](mailto:mattlgroff@gmail.com) | (619) 312-5617  
[GitHub](https://github.com/mattlgroff) | [LinkedIn](https://www.linkedin.com/in/mattgroff/) | [Dev Blog](https://groff.dev)

As an AI Capability Leader, I foster a culture of innovation, guiding the development of scalable, enterprise-grade AI products. My leadership encourages risk-taking and creativity, ensuring the delivery of advanced AI solutions like RAG ChatBots and Marketing Content Generation. I focus on practical applications of AI, optimizing open-source models for efficiency, and positioning our projects at the forefront of the Generative AI field.

---

## SKILLS

### Technical Skills:
- **Generative AI**: OpenAI, Open Source Models, Fine-tuning, Function Calling, Agents, ChatBots  
- **Full-Stack**: React, Typescript, Node, Bun, Python FastAPI, GraphQL  
- **Database & Data Modeling**: PostgreSQL, Vector Databases  
- **Deployment & Operations**: Docker, Linux, CI/CD (GitHub Actions, Gitlab Pipeline, Azure DevOps)  

### Non-Technical Skills:
- Tech Leadership  
- Mentoring and training junior engineers  
- Client Interaction  
- Scrum & Agile  
- Creating organizational wide AI policy  

---

## EXPERIENCE

### **Principal AI Engineer @ Umbrage**  
*Houston, TX (Remote) — Dec 2019 - Present*  
*(Previous roles: Director of Engineering, Senior Web Engineer)*  

### **Web Developer @ Accenture**  
*Houston, TX — Jun 2018 - Dec 2019*  

### **Self Employed App Developer**  
*San Diego, CA — Mar 2013 - Jun 2018*  

---

## EDUCATION

- **Graph Developer - Professional Certification** — Apollo GraphQL — Jan 2023  
- **Graph Developer - Associate Certification** — Apollo GraphQL — May 2022  
- **UCSD Extension** — Coding Bootcamp in Web Development — 2017 - 2018  
- **UCSD Extension** — UNIX and Linux Systems Administration Certification — 2013 - 2014 
"""

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": system_instructions,
        },
        {
            "role": "user",
            "content": candidate_resume,
        },
    ],
    model="deepseek-r1-distill-qwen-32b",
)

# print(chat_completion)

# First extract the content between <propositions></propositions> XML tags
xml_content = re.search(
    r"<propositions>(.*?)</propositions>",
    chat_completion.choices[0].message.content,
    re.DOTALL,
)
if xml_content:
    # Then extract the propositions with dashes from within the XML content
    propositions = re.findall(r"- (.*)", xml_content.group(1))
    print(propositions)
else:
    print("No propositions found within XML tags")
